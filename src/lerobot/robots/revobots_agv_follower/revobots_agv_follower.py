#!/usr/bin/env python
"""Revobots AGV follower robot: UDP velocity I/O plus UDP GPS telemetry."""

import json
import logging
import socket
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_revobots_agv_follower import RevobotsAGVFollowerConfig

logger = logging.getLogger(__name__)

# Action feature keys
ACTION_LINEAR_VEL = "lin_x"
ACTION_ANGULAR_VEL = "ang_z"

# Observation feature keys
OBS_LINEAR_VEL = "lin_x"
OBS_ANGULAR_VEL = "ang_z"
OBS_LATITUDE = "lat"
OBS_LONGITUDE = "long"
OBS_ORIENTATION = "orientation"


class _VelocityUdpReader:
    """Background UDP listener caching the latest {lin_x, ang_z} JSON packet."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest = {"lin_x": 0.0, "ang_z": 0.0}

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.5)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, _ = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                logger.debug(f"Dropped malformed velocity packet: {e}")
                continue
            with self._lock:
                self._latest["lin_x"] = float(payload.get("lin_x", self._latest["lin_x"]))
                self._latest["ang_z"] = float(payload.get("ang_z", self._latest["ang_z"]))

    def get(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sock = None
        self._thread = None


class RevobotsAGVFollower(Robot):
    config_class = RevobotsAGVFollowerConfig
    name = "revobots_agv_follower"

    def __init__(self, config: RevobotsAGVFollowerConfig):
        super().__init__(config)
        self.config = config

        self.cameras = make_cameras_from_configs(config.cameras)

        self._velocity_reader: _VelocityUdpReader | None = None
        self._write_sock: socket.socket | None = None
        self._gps = None

        self._latest_lat = 0.0
        self._latest_long = 0.0
        self._latest_orientation = 0.0

        self._is_connected = False

    # ------------------------------------------------------------------
    # Feature specs (required by Robot base class)
    # ------------------------------------------------------------------

    @property
    def _cameras_ft(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Observation space for dataset recording.

        - lin_x / ang_z: float - Live velocity feedback
        - lat / long: float - GPS coordinates
        - orientation: float - Heading in degrees
        - camera keys: (height, width, 3) - RGB frames
        """
        return {
            OBS_LINEAR_VEL: float,
            OBS_ANGULAR_VEL: float,
            OBS_LATITUDE: float,
            OBS_LONGITUDE: float,
            OBS_ORIENTATION: float,
            **self._cameras_ft,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action space: lin_x / ang_z target velocities."""
        return {
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from LAB.sensors import GpsReader

        self._velocity_reader = _VelocityUdpReader(
            self.config.cmd_vel_read_host, self.config.cmd_vel_read_port
        )
        self._velocity_reader.start()

        self._write_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._gps = GpsReader(udp_host=self.config.gps_udp_host, udp_port=self.config.gps_udp_port)
        self._gps.start()

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        self.configure()

        logger.info(
            f"{self.name} connected — cmd_vel read {self.config.cmd_vel_read_host}:"
            f"{self.config.cmd_vel_read_port}, write {self.config.cmd_vel_write_host}:"
            f"{self.config.cmd_vel_write_port}, gps {self.config.gps_udp_host}:{self.config.gps_udp_port}"
        )

    def calibrate(self) -> None:
        logger.info(f"Calibration not required for {self.name}")

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.send_action({ACTION_LINEAR_VEL: 0.0, ACTION_ANGULAR_VEL: 0.0})
        except Exception as e:
            logger.warning(f"Failed to send stop command during disconnect: {e}")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass

        if self._velocity_reader is not None:
            self._velocity_reader.stop()
            self._velocity_reader = None

        if self._write_sock is not None:
            self._write_sock.close()
            self._write_sock = None

        if self._gps is not None:
            self._gps.stop()
            self._gps = None

        self._is_connected = False
        logger.info(f"Disconnected {self.name}")

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------

    def _update_gps(self) -> None:
        gps_dict = self._gps.get()
        if gps_dict:
            self._latest_lat = gps_dict.get("gps_latitude", 0.0)
            self._latest_long = gps_dict.get("gps_longitude", 0.0)
            self._latest_orientation = gps_dict.get("orientation", 0.0)

    @check_if_not_connected
    def get_observation(self) -> dict[str, Any]:
        self._update_gps()
        velocity = self._velocity_reader.get()

        obs_dict: dict[str, Any] = {
            OBS_LINEAR_VEL: velocity["lin_x"],
            OBS_ANGULAR_VEL: velocity["ang_z"],
            OBS_LATITUDE: self._latest_lat,
            OBS_LONGITUDE: self._latest_long,
            OBS_ORIENTATION: self._latest_orientation,
        }

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        lin_x = float(action.get(ACTION_LINEAR_VEL, 0.0))
        ang_z = float(action.get(ACTION_ANGULAR_VEL, 0.0))

        payload = json.dumps({"lin_x": lin_x, "ang_z": ang_z}).encode("utf-8")
        self._write_sock.sendto(payload, (self.config.cmd_vel_write_host, self.config.cmd_vel_write_port))

        return {
            ACTION_LINEAR_VEL: lin_x,
            ACTION_ANGULAR_VEL: ang_z,
        }

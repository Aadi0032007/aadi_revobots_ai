#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import select
import socket
import time

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_koch_leader_remote import KochLeaderRemoteConfig

logger = logging.getLogger(__name__)

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# A six joint action serializes to ~200 bytes, so a single datagram always holds a whole action.
MAX_PACKET_BYTES = 4096

# How long a single `get_action` poll waits on the socket before looping again.
POLL_TIMEOUT_S = 0.1


class KochLeaderRemote(Teleoperator):
    """
    Receiving half of a network-split Koch leader arm.

    The physical arm lives on another machine running
    :pymod:`~lerobot.teleoperators.koch_leader_remote.koch_leader_remote_client`, which streams its joint
    positions here over UDP. This class owns no hardware: it binds a socket and hands the freshest action it
    has received to the teleoperation loop.

    UDP fits because every packet carries *absolute* joint positions, so a dropped packet needs no recovery
    and is simply superseded by the next one. Two safeguards keep that from turning into silently wrong
    actions:

    - packets carry a sequence number, so a reordered datagram can never overwrite a newer pose;
    - freshness is measured from the packet's *arrival* time here, since the sender's clock is unrelated to
      ours. An action older than `max_action_age_s` is treated as no action at all.
    """

    config_class = KochLeaderRemoteConfig
    name = "koch_leader_remote"

    def __init__(self, config: KochLeaderRemoteConfig):
        super().__init__(config)
        self.config = config
        self._sock: socket.socket | None = None
        self._action: dict[str, float] | None = None
        self._action_rx_t: float = 0.0
        self._session: str | None = None
        self._last_seq: int = -1
        self._warned_about_keys = False

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTORS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.config.host, self.config.port))
        except OSError as e:
            sock.close()
            raise ConnectionError(
                f"Failed to bind {self.config.host}:{self.config.port} for {self}: {e}"
            ) from e

        self._sock = sock
        self._action = None
        self._action_rx_t = 0.0
        self._session = None
        self._last_seq = -1
        logger.info(f"{self} listening for leader actions on {self.config.host}:{self.config.port}.")

    @property
    def is_calibrated(self) -> bool:
        # The arm this teleoperator mirrors is calibrated on the client machine, against its own motors.
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def _accept_packet(self, data: bytes) -> None:
        """Validate one datagram and, if it is newer than what we hold, adopt it as the current action."""
        try:
            packet = json.loads(data.decode("utf-8"))
            session = str(packet["session"])
            seq = int(packet["seq"])
            action = {key: float(val) for key, val in packet["action"].items()}
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as e:
            logger.debug(f"{self} dropped a malformed action packet: {e}")
            return

        if not action:
            logger.debug(f"{self} dropped an empty action packet.")
            return

        if session != self._session:
            # The client restarted and its sequence numbers began again from zero.
            logger.info(f"{self} picked up leader arm session {session}.")
            self._session = session
            self._last_seq = -1
        elif seq <= self._last_seq:
            logger.debug(f"{self} dropped out-of-order packet {seq} (holding {self._last_seq}).")
            return

        if not self._warned_about_keys and set(action) != set(self.action_features):
            logger.warning(
                f"{self} received action keys {sorted(action)}, expected {sorted(self.action_features)}. "
                "Passing them through as-is; check that both machines run the same motor configuration."
            )
            self._warned_about_keys = True

        self._last_seq = seq
        self._action = action
        self._action_rx_t = time.perf_counter()

    def _receive_latest(self, timeout_s: float) -> None:
        """Wait up to `timeout_s` for traffic, then consume every datagram already queued."""
        ready = select.select([self._sock], [], [], timeout_s)[0]
        while ready:
            try:
                data, _ = self._sock.recvfrom(MAX_PACKET_BYTES)
            except ConnectionResetError:
                # Windows reports an earlier send's ICMP "port unreachable" on the next read. A listening
                # socket has nothing to recover from, so keep draining.
                logger.debug(f"{self} ignored an ICMP unreachable notification.")
            except OSError as e:
                logger.debug(f"{self} failed to read from its socket: {e}")
                return
            else:
                self._accept_packet(data)
            ready = select.select([self._sock], [], [], 0)[0]

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        waiting_since: float | None = None
        last_log_t = 0.0

        while True:
            self._receive_latest(POLL_TIMEOUT_S)

            now = time.perf_counter()
            if self._action is not None and now - self._action_rx_t <= self.config.max_action_age_s:
                if waiting_since is not None:
                    logger.info(f"{self} resumed after {now - waiting_since:.1f}s without a fresh action.")
                return dict(self._action)

            if waiting_since is None:
                waiting_since = now
                last_log_t = now
                logger.info(f"Waiting for actions from the remote leader arm on port {self.config.port}...")
            elif now - last_log_t >= self.config.waiting_log_period_s:
                last_log_t = now
                logger.info(
                    f"Still waiting for the remote leader arm ({now - waiting_since:.0f}s so far)..."
                )

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self._sock.close()
        self._sock = None
        self._action = None
        logger.info(f"{self} disconnected.")

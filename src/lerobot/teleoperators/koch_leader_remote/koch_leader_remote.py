# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 07:45:37 2026

@author: Aadi
"""

import logging
import socket
import json
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..teleoperator import Teleoperator
from .config_koch_leader_remote import KochLeaderRemoteConfig

logger = logging.getLogger(__name__)

class KochLeaderRemote(Teleoperator):
    """
    Remote Koch Leader that listens for joint data over a persistent TCP socket.
    """
    config_class = KochLeaderRemoteConfig
    name = "koch_leader_remote"

    def __init__(self, config: KochLeaderRemoteConfig):
        super().__init__(config)
        self.config = config
        self.server_sock = None
        self.client_conn = None
        self.motor_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex", 
            "wrist_flex", "wrist_roll", "gripper"
        ]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.server_sock is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return

        port = self.config.port
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind(("0.0.0.0", port))
            self.server_sock.listen(1)
            self.server_sock.settimeout(0.1) # Non-blocking accept
            logger.info(f"KochLeaderRemote listening on 0.0.0.0:{port}...")
        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to start server: {e}")

    def _wait_for_client(self):
        """Internal helper to accept a new client connection if one isn't active."""
        try:
            conn, addr = self.server_sock.accept()
            conn.settimeout(1.0)
            self.client_conn = conn
            logger.info(f"Client connected from {addr}")
        except socket.timeout:
            pass 

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        while True:
            # 1. If no client is connected, try to accept one
            if self.client_conn is None:
                print("waiting for data from remote teleoperator")
                self._wait_for_client()
                if self.client_conn is None:
                    continue  # Keep looping until a client connects

            try:
                # 2. Request data from the client
                self.client_conn.sendall(b"GET_ACTION\n")
                data = self.client_conn.recv(1024).decode('utf-8').strip()

                # 3. Check if data is empty or connection closed
                if not data:
                    print("waiting for data from remote teleoperator")
                    self.client_conn = None  # Reset connection to trigger reconnection logic
                    continue

                # 4. Parse JSON and ensure it contains action data
                action = json.loads(data)
                if action:  # Only return if the dictionary is not empty
                    return action
                
                # If JSON was valid but empty (e.g. {}), print and loop again
                print("waiting for data from remote teleoperator")

            except (ConnectionError, socket.timeout, Exception):
                # 5. Handle crashes or timeouts by resetting the connection
                print("waiting for data from remote teleoperator")
                self.client_conn = None
                # The loop will automatically attempt to reconnect via _wait_for_client()

    def calibrate(self) -> None: pass
    def configure(self) -> None: pass
    def setup_motors(self) -> None: pass
    def send_feedback(self, feedback: dict[str, float]) -> None: pass

    @property
    def is_calibrated(self) -> bool: return True

    def disconnect(self) -> None:
        if self.client_conn: self.client_conn.close()
        if self.server_sock: self.server_sock.close()
        self.client_conn = None
        self.server_sock = None
#!/usr/bin/env python
"""Configuration for the Revobots AGV follower robot."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("revobots_agv_follower")
@dataclass
class RevobotsAGVFollowerConfig(RobotConfig):
    # Shared cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # GPS telemetry, read over UDP
    gps_udp_host: str = "127.0.0.1"
    gps_udp_port: int = 57002

    # Live velocity feedback (e.g. from a teleop/joystick node), read over UDP
    cmd_vel_read_host: str = "127.0.0.1"
    cmd_vel_read_port: int = 57003

    # Velocity command output to the AGV's low-level motor controller, sent over UDP
    cmd_vel_write_host: str = "127.0.0.1"
    cmd_vel_write_port: int = 57004

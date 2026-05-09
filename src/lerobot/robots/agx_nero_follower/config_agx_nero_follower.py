# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:57:52 2026

@author: Aadi
"""

import platform
from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig

from ..config import RobotConfig


def _detect_can_defaults() -> tuple[str, str]:
    """Return (interface, channel) based on the current OS."""
    system = platform.system()
    if system == "Windows":
        return ("agx_cando", "0")
    if system == "Linux":
        return ("socketcan", "can0")
    if system == "Darwin":
        return ("slcan", "/dev/ttyACM0")
    raise RuntimeError(
        f"Unsupported platform '{system}'. "
        "pyAgxArm supports Linux (socketcan), Windows (agx_cando), and macOS (slcan)."
    )


@RobotConfig.register_subclass("agx_nero_follower")
@dataclass
class AgxNeroFollowerConfig(RobotConfig):
    # None → auto-detected from the current OS and available CAN ports
    interface: str | None = None
    channel: str | None = None
    # Firmware version — "default" (<=1.10) | "v111" (>=1.11)
    firmware_version: str = "v111"
    bitrate: int = 1_000_000
    # Motion speed 0-100%
    speed_percent: int = 20
    # Max per-step joint movement (degrees). None = unlimited.
    max_relative_target: float | dict[str, float] | None = None
    disable_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.interface is None or self.channel is None:
            detected_interface, detected_channel = _detect_can_defaults()
            if self.interface is None:
                self.interface = detected_interface
            if self.channel is None:
                self.channel = detected_channel

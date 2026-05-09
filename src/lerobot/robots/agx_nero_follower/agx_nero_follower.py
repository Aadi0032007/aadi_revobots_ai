# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:57:52 2026

@author: Aadi
"""

import logging
import math
import time
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_agx_nero_follower import AgxNeroFollowerConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "forearm_roll",
    "wrist_pitch",
    "wrist_roll",
    "wrist_twist",
]

# Software joint limits in degrees (mirrors pyAgxArm constants.py for NERO)
JOINT_LIMITS_DEG = {
    "shoulder_pan":  (-155.0,  155.0),
    "shoulder_lift": (-100.0,  100.0),
    "elbow_flex":    (-158.0,  158.0),
    "forearm_roll":  ( -58.0,  123.0),
    "wrist_pitch":   (-158.0,  158.0),
    "wrist_roll":    ( -42.0,   55.0),
    "wrist_twist":   ( -90.0,   90.0),
}


class AgxNeroFollower(Robot):
    config_class = AgxNeroFollowerConfig
    name = "agx_nero_follower"

    def __init__(self, config: AgxNeroFollowerConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self.logs: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Feature specs (required by Robot base class)
    # ------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and self._arm.is_connected()

    @property
    def is_calibrated(self) -> bool:
        # Nero uses absolute encoders with factory calibration — no user calibration needed
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

        fw_map = {"default": NeroFW.DEFAULT, "v111": NeroFW.V111}
        fw = fw_map.get(self.config.firmware_version, NeroFW.DEFAULT)

        arm_cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=fw,
            interface=self.config.interface,
            channel=self.config.channel,
            bitrate=self.config.bitrate,
        )
        self._arm = AgxArmFactory.create_arm(arm_cfg)
        self.gripper = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._arm.connect()
        self._arm.set_normal_mode()
        self.gripper.move_gripper_deg(value=0.0, force=3.0)

        # Enable all joints (retry until confirmed)
        for _ in range(50):
            if self._arm.enable():
                break
            time.sleep(0.02)
        else:
            logger.warning("%s: joint enable did not confirm within timeout", self.name)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        logger.info("Connected %s on %s/%s", self.name, self.config.interface, self.config.channel)

    def configure(self) -> None:
        if self._arm is None:
            return
        self._arm.set_speed_percent(self.config.speed_percent)
        self._arm.set_joint_limits_enabled(True)
        self._arm.set_auto_set_motion_mode_enabled(True)
        
        if hasattr(self, "gripper") and self.gripper is not None:
            self.gripper.set_gripper_teaching_pendant_param(
                teaching_range_per=200,
                max_range_config=0.1,
                teaching_friction=10,
            )
            print("AGX gripper connected")

    def calibrate(self) -> None:
        # Nero arm has factory-calibrated absolute encoders; nothing to do here.
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass

        if self._arm is not None:
            # if self.config.disable_on_disconnect:
            #     self._arm.disable(255)
            self._arm.disconnect()
            self._arm = None

        logger.info("Disconnected %s", self.name)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")

        obs: dict[str, Any] = {}

        t0 = time.perf_counter()
        joint_angles_rad = self._read_joint_angles_rad()
        for i, name in enumerate(JOINT_NAMES):
            obs[f"{name}.pos"] = math.degrees(joint_angles_rad[i])
        self.logs["read_pos_dt_s"] = time.perf_counter() - t0

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected.")
    
        t0 = time.perf_counter()
    
        # 1. Extract keys and values directly from the action dict (ignoring JOINT_NAMES)
        # We assume the order in the dictionary is the correct joint order
        joint_keys = [k for k in action.keys() if k.endswith(".pos")]
        goal_deg = [float(action[k]) for k in joint_keys]
        
        goal_deg[2] = -goal_deg[2] + 90
        gripper = min(goal_deg[5] * 2, 95)
        goal_deg.pop(5)
        # 2. Force indices 2 and 5 to zero
        goal_deg.insert(2, 0.0)
        goal_deg.insert(5, 0.0)
        goal_deg[1] = goal_deg[1] + 43.7466
        goal_deg[4], goal_deg[6] = goal_deg[6], goal_deg[4]
    
        # 4. Apply max_relative_target safety cap if configured
        if self.config.max_relative_target is not None:
            # Read current angles to compare
            present_deg = [math.degrees(v) for v in self._read_joint_angles_rad()]
            
            # Create the mapping for the safety checker: { 'joint.pos': (goal, present) }
            goal_present = {
                k: (goal_deg[i], present_deg[i] if i < len(present_deg) else goal_deg[i]) 
                for i, k in enumerate(joint_keys)
            }
            
            safe = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            # Update goal_deg with safe values
            goal_deg = [safe[k] for k in joint_keys]
    
        # 5. Convert degrees → radians and send to arm
        # print(goal_deg)
        self.gripper.move_gripper_deg(value=gripper)
        self._arm.move_j([math.radians(v) for v in goal_deg])
        
        
        self.logs["write_pos_dt_s"] = time.perf_counter() - t0
    
        # 6. Return the action actually sent
        return {k: goal_deg[i] for i, k in enumerate(joint_keys)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_joint_angles_rad(self) -> list[float]:
        result = self._arm.get_joint_angles()
        if result is None:
            return [0.0] * 7
        return list(result.msg)

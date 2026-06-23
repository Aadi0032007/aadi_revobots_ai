# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:57:52 2026

@author: Aadi
"""

# -*- coding: utf-8 -*-
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


class AgxNeroFollower(Robot):
    config_class = AgxNeroFollowerConfig
    name = "agx_nero_follower"

    # 1. Define the arm's physical joints exactly ONCE here
    FOLLOWER_JOINTS = (
        "shoulder_pan",
        "shoulder_lift",
        "forearm_roll",
        "elbow_flex",
        "wrist_roll",
        "wrist_pitch",
        "wrist_flex",
        "gripper"
    )

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
        # Dynamically build the dictionary based on the FOLLOWER_JOINTS tuple
        return {f"{name}.pos": float for name in self.FOLLOWER_JOINTS}

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
        self.gripper.move_gripper_deg(value=0.0, force=0.5)

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
        
        # 1. Map hardware array to software named space
        # NOTE: Apply your REVERSE offsets/math here if your hardware requires it.
        # (e.g., obs["shoulder_pan.pos"] = math.degrees(joint_angles_rad[0]) + 5.0)
        if len(joint_angles_rad) >= 7:
            obs["shoulder_pan.pos"] = math.degrees(joint_angles_rad[0])
            obs["shoulder_lift.pos"] = math.degrees(joint_angles_rad[1])
            obs["forearm_roll.pos"] = math.degrees(joint_angles_rad[2])
            obs["elbow_flex.pos"] = math.degrees(joint_angles_rad[3])
            obs["wrist_roll.pos"] = math.degrees(joint_angles_rad[4])
            obs["wrist_pitch.pos"] = math.degrees(joint_angles_rad[5])
            obs["wrist_flex.pos"] = math.degrees(joint_angles_rad[6])
        else:
            # Fallback in case of hardware read failure
            for name in self.FOLLOWER_JOINTS:
                if name != "gripper":
                    obs[f"{name}.pos"] = 0.0

        # 2. Gripper read
        gs = self.gripper.get_gripper_status()
        # Apply reverse scaling if necessary (e.g., raw_gripper / 2.0)
        raw_gripper = float(gs.msg.value) if gs is not None else 0.0
        obs["gripper.pos"] = raw_gripper
        
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
    
        # 1. Generic Extraction
        # Automatically grabs all 8 joints perfectly, defaulting missing ones to 0.0
        raw_goals = {
            name: float(action.get(f"{name}.pos", 0.0)) 
            for name in self.FOLLOWER_JOINTS
        }

        # 2. Apply max_relative_target safety cap
        if self.config.max_relative_target is not None:
            present_rads = self._read_joint_angles_rad()
            
            if len(present_rads) >= 7:
                # Map the hardware array back to our named software space
                present_deg = {
                    "shoulder_pan": math.degrees(present_rads[0]),
                    "shoulder_lift": math.degrees(present_rads[1]),
                    "forearm_roll": math.degrees(present_rads[2]),
                    "elbow_flex": math.degrees(present_rads[3]),
                    "wrist_roll": math.degrees(present_rads[4]),
                    "wrist_pitch": math.degrees(present_rads[5]),
                    "wrist_flex": math.degrees(present_rads[6]),
                    "gripper": float(self.gripper.get_gripper_status().msg.value) if self.gripper.get_gripper_status() else 0.0
                }
            else:
                present_deg = {name: 0.0 for name in self.FOLLOWER_JOINTS}
            
            goal_present = {name: (raw_goals[name], present_deg[name]) for name in self.FOLLOWER_JOINTS}
            safe_goals = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            raw_goals.update(safe_goals)

        # 3. Hardware Math & Command Construction
        # NOTE: Apply your hardware-specific math/inversions here 
        # (e.g., pan = raw_goals["shoulder_pan"] - 5.0)
        goal_deg_hardware = [
            raw_goals["shoulder_pan"] - 5.0,   # Index 0
            raw_goals["shoulder_lift"] + 43.7466,  # Index 1
            raw_goals["forearm_roll"],     # Index 2
            raw_goals["elbow_flex"] + 90.0,   # Index 3
            raw_goals["wrist_roll"],    # Index 4
            raw_goals["wrist_pitch"],     # Index 5
            raw_goals["wrist_flex"]      # Index 6
        ]
        
        # Calculate gripper (apply scaling or caps here if needed)
        gripper_cmd = min(raw_goals["gripper"] * 2, 95)

        # 4. Convert to radians and send to arm
        self.gripper.move_gripper_deg(value=gripper_cmd, force=0.5)
        self._arm.move_j([math.radians(v) for v in goal_deg_hardware])
        
        self.logs["write_pos_dt_s"] = time.perf_counter() - t0
    
        # 5. Return the software-space action that was actually applied
        return {f"{name}.pos": raw_goals[name] for name in self.FOLLOWER_JOINTS}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_joint_angles_rad(self) -> list[float]:
        result = self._arm.get_joint_angles()
        if result is None:
            return [0.0] * 7
        return list(result.msg)
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 18:25:46 2026

@author: Aadi

Minimal policy-inference CLI: loads a trained policy, runs it on a real robot
at a fixed control rate. Dataset metadata (from the training dataset) supplies
the feature spec and the normalization stats.

Example:

```shell
python -m lerobot.scripts.lerobot_inference \
    --robot.type=revobots_agv_follower \
    --robot.id=agv \
    --policy.path=outputs/train/checkpoints/last/pretrained_model \
    --dataset_repo_id=<user>/<training_dataset> \
    --single_task="drive to the bin" \
    --fps=30 \
    --display_data=true
```

Press ESC (or `q`) to stop.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from lerobot.common.control_utils import predict_action
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# ---------------------------------------------------------------------
# CLI Config
# ---------------------------------------------------------------------
@dataclass
class InferenceConfig:
    robot: RobotConfig

    # Dataset on Hub (or local, via --root) used only for: ds features +
    # normalization stats + policy wiring
    dataset_repo_id: str

    # Task text passed into predict_action(...)
    single_task: str

    # Set through `--policy.path=<dir or hub id>`; see __get_path_fields__ below.
    policy: PreTrainedConfig | None = None

    # Control loop
    fps: int = 30

    control_time_s: float | None = None

    # Visualization (rerun)
    display_data: bool = False

    # Local dataset directory (skips the Hub lookup for dataset_repo_id)
    root: str | Path | None = None

    # Optional observation key renaming (same idea as record)
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """
        IMPORTANT:
        The ACT positional embedding mismatch happens when we instantiate a default ACT
        config and then load weights trained with different seq lengths / camera setup.
        Reloading the config from the checkpoint (as below) is what prevents that, so
        `--policy.path` is required rather than `--policy.type`.
        """
        # HACK: re-parse the CLI to recover `--policy.path` (draccus strips it out,
        # see __get_path_fields__). Same pattern as EvalPipelineConfig / RolloutConfig.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Keep any CLI overrides the user provided under --policy.*
            cli_overrides = parser.get_cli_overrides("policy") or []
            # Also preserve device if it was set
            device_before = getattr(self.policy, "device", None)

            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

            if device_before is not None:
                self.policy.device = device_before

        if self.policy is None:
            raise ValueError("--policy.path=<checkpoint dir or hub id> is required for inference.")

        if self.fps <= 0:
            raise ValueError(f"--fps must be > 0, got {self.fps}.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Enables loading the policy config from `--policy.path=local/dir`."""
        return ["policy"]


# ---------------------------------------------------------------------
# Inference loop (record_loop-like, but with optional dataset writing)
# ---------------------------------------------------------------------
def inference_loop(
    *,
    robot,
    events: dict,
    fps: int,
    robot_action_processor,
    robot_observation_processor,
    policy,
    preprocessor,
    postprocessor,
    features: dict[str, Any],
    dataset: LeRobotDataset | None,
    single_task: str,
    control_time_s: float | None,
    display_data: bool,
):
    device = get_safe_torch_device(policy.config.device)

    start_t = time.perf_counter()
    while True:
        loop_t = time.perf_counter()

        # Stop conditions
        if events.get("stop_recording", False):
            break
        if control_time_s is not None and (time.perf_counter() - start_t) >= control_time_s:
            break

        # 1) Observation
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        # 2) Dataset-shaped frame (same idea as record_loop)
        observation_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

        # 3) Policy inference
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )

        # 4) Convert into robot action (same as record_loop)
        act_processed_policy = make_robot_action(action_values, features)

        # 5) Robot action processing (clipping, formatting, etc.)
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # 6) Send to robot
        _sent = robot.send_action(robot_action_to_send)

        # 7) Optional: write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(features, act_processed_policy, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        # 8) Optional visualization
        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        # 9) Maintain FPS
        dt = time.perf_counter() - loop_t
        if dt > 1.0 / fps:
            logging.warning(f"Control loop overran its budget: {dt * 1e3:.1f}ms > {1e3 / fps:.1f}ms")
        precise_sleep(max(0.0, 1.0 / fps - dt))


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
@parser.wrap()
def run(cfg: InferenceConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="inference")

    # Load dataset metadata (features + stats)
    ds_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id, root=cfg.root)

    # Create robot
    robot = make_robot_from_config(cfg.robot)

    # Processors (same ones record uses)
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    listener = None
    policy_cfg = cfg.policy  # guaranteed non-None by __post_init__

    try:
        # Create policy (requires ds_meta)
        policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=cfg.rename_map)

        # Pre/post processors (use dataset stats from metadata)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            dataset_stats=rename_stats(ds_meta.stats, cfg.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": policy_cfg.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )

        # Connect robot
        robot.connect()

        # Keyboard listener for ESC
        listener, events = init_keyboard_listener()

        # Reset policy + pipelines
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        log_say("running policy inference (no dataset writes). Press ESC to stop.")
        inference_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            features=ds_meta.features,
            dataset=None,  # no writes
            single_task=cfg.single_task,
            control_time_s=cfg.control_time_s,  # None => infinite until ESC
            display_data=cfg.display_data,
        )

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        log_say("Exiting inference", blocking=False)
        if listener is not None:
            listener.stop()
        if robot is not None and getattr(robot, "is_connected", False):
            robot.disconnect()


def main():
    run()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 02:09:20 2026

@author: Aadi

Callable (non-CLI) policy inference. The robot lifecycle (create / connect /
disconnect) is owned by the caller; this module only builds the policy stack and
drives the control loop for a bounded duration.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Policy + processor builder (NO robot inside)
# ---------------------------------------------------------------------
def build_policy_pipeline(
    *,
    policy_path: str,
    dataset_repo_id: str,
    device: str | None = None,
    rename_map: dict[str, str] | None = None,
    dataset_root: str | None = None,
):
    """
    Builds policy + processors needed for inference.

    Args:
        policy_path: checkpoint directory or Hub id of the trained policy.
        dataset_repo_id: training dataset, used for its feature spec + stats.
        device: "cuda" / "cpu" / "mps". ``None`` auto-selects an available device.
        rename_map: optional observation key renaming.
        dataset_root: local directory of the dataset (skips the Hub lookup).

    Returns:
        policy
        preprocessor
        postprocessor
        robot_action_processor
        robot_observation_processor
        dataset_features

    Build this once and pass it to :func:`run_inference` via ``pipeline=`` when
    calling repeatedly — otherwise the policy weights are reloaded every call.
    """
    if rename_map is None:
        rename_map = {}

    # Dataset metadata (features + normalization stats)
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)

    # Same processors as record/inference scripts
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    # Load pretrained policy config correctly
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if device is not None:
        policy_cfg.device = device
    policy_cfg.pretrained_path = Path(policy_path)

    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            # `policy_cfg.device` is the resolved device (PreTrainedConfig falls
            # back to an available one when the requested device is missing).
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    return (
        policy,
        preprocessor,
        postprocessor,
        robot_action_processor,
        robot_observation_processor,
        ds_meta.features,
    )


# ---------------------------------------------------------------------
# Time-bounded inference loop
# ---------------------------------------------------------------------
def inference_loop(
    *,
    robot,
    policy,
    preprocessor,
    postprocessor,
    robot_action_processor,
    robot_observation_processor,
    features: dict[str, Any],
    single_task: str | None,
    fps: int,
    duration_s: float,
    display_data: bool,
    observation_hook: Callable[[dict], dict] | None = None,
):
    """Drive `robot` with `policy` for `duration_s` seconds at `fps`.

    Args:
        observation_hook: optional callable applied to the raw observation dict
            before the robot observation processor (used e.g. to draw markers).
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}.")

    device = get_safe_torch_device(policy.config.device)
    period = 1.0 / fps

    start_t = time.perf_counter()

    while (time.perf_counter() - start_t) < duration_s:
        loop_t = time.perf_counter()

        # 1) Get observation
        obs = robot.get_observation()
        if observation_hook is not None:
            obs = observation_hook(obs)
        obs_processed = robot_observation_processor(obs)

        # 2) Dataset-shaped observation frame
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

        # 4) Convert to robot action
        act_processed_policy = make_robot_action(action_values, features)

        # 5) Robot-side action processing
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # 6) Send to robot
        robot.send_action(robot_action_to_send)

        # 7) Optional visualization
        if display_data:
            log_rerun_data(observation=obs_processed, action=act_processed_policy)

        # 8) Maintain FPS
        dt = time.perf_counter() - loop_t
        if dt > period:
            logger.warning("Control loop overran its budget: %.1fms > %.1fms", dt * 1e3, period * 1e3)
        precise_sleep(max(0.0, period - dt))


# ---------------------------------------------------------------------
# Public API: callable inference (robot passed in)
# ---------------------------------------------------------------------
def run_inference(
    *,
    robot,
    policy_path: str | None = None,
    dataset_repo_id: str | None = None,
    duration_s: float = 60.0,
    fps: int = 30,
    device: str | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    rename_map: dict[str, str] | None = None,
    dataset_root: str | None = None,
    pipeline: tuple | None = None,
    observation_hook: Callable[[dict], dict] | None = None,
):
    """
    Runs policy inference on an already-initialized and connected robot.

    Robot lifecycle (create/connect/disconnect) MUST be handled externally.

    Pass ``pipeline`` (the return value of :func:`build_policy_pipeline`) to reuse
    an already-loaded policy instead of reloading it on every call; otherwise
    ``policy_path`` and ``dataset_repo_id`` are required.
    """

    init_logging()

    if display_data:
        init_rerun(session_name="inference")

    if pipeline is None:
        if policy_path is None or dataset_repo_id is None:
            raise ValueError("Provide either `pipeline=` or both `policy_path=` and `dataset_repo_id=`.")
        pipeline = build_policy_pipeline(
            policy_path=policy_path,
            dataset_repo_id=dataset_repo_id,
            device=device,
            rename_map=rename_map,
            dataset_root=dataset_root,
        )

    (
        policy,
        preprocessor,
        postprocessor,
        robot_action_processor,
        robot_observation_processor,
        features,
    ) = pipeline

    # Reset everything before running
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    logger.info("Running inference for %.1fs", duration_s)

    inference_loop(
        robot=robot,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        features=features,
        single_task=single_task,
        fps=fps,
        duration_s=duration_s,
        display_data=display_data,
        observation_hook=observation_hook,
    )

    logger.info("Inference finished")


def run_inference_with_markers(
    *,
    robot,
    prompt: str,
    policy_path: str | None = None,
    dataset_repo_id: str | None = None,
    duration_s: float = 60.0,
    fps: int = 30,
    device: str | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    rename_map: dict[str, str] | None = None,
    dataset_root: str | None = None,
    pipeline: tuple | None = None,
    camera_key: str = "phone",
    gemini_api_key: str | None = None,
):
    """
    Same as :func:`run_inference`, but the object described by `prompt` is located
    once (via Gemini) at startup and its marker is drawn onto every `camera_key`
    frame before the observation reaches the policy.

    Robot lifecycle (create/connect/disconnect) MUST be handled externally.
    """
    from lerobot.cameras.image_detection_tracking.gemini_utils import (
        draw_markers_on_image,
        get_object_coordinates,
    )

    obs = robot.get_observation()
    if camera_key not in obs:
        raise KeyError(
            f"Camera '{camera_key}' not in the robot observation. Available keys: {sorted(obs)}. "
            f"Pass camera_key=<name>."
        )

    detections = get_object_coordinates(obs[camera_key], prompt, api_key=gemini_api_key) or []
    logger.info("Marker detections for prompt %r: %s", prompt, detections)
    if not detections:
        logger.warning("No object matched prompt %r; running without markers.", prompt)

    def observation_hook(observation: dict) -> dict:
        if detections:
            observation[camera_key] = draw_markers_on_image(observation[camera_key], detections)
        return observation

    run_inference(
        robot=robot,
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        duration_s=duration_s,
        fps=fps,
        device=device,
        single_task=single_task,
        display_data=display_data,
        rename_map=rename_map,
        dataset_root=dataset_root,
        pipeline=pipeline,
        observation_hook=observation_hook,
    )

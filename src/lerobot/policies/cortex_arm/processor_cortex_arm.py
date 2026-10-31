# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

@author: Aadi
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import RobotObservation
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_cortex_arm import CortexArmConfig


@dataclass
@ProcessorStepRegistry.register(name="cortex_arm_depth_processor")
class CortexArmDepthProcessorStep(ObservationProcessorStep):
    """Give every depth map the form it has in the dataset, before it is normalized.

    Frames read from the dataset arrive as float32 millimetres, already dequantized by the dataset
    reader. Live frames come straight from the RealSense as uint16 millimetres. For each depth key:

    - float frames (from the dataset) are only made float32;
    - integer frames (raw camera) are cast to float32 and, for depth stored as video
      (``encodings[key]``, the dataset's depth encoder settings), put through LeRobot's own
      ``quantize_depth`` → ``dequantize_depth``. They then come out exactly as the same frame would
      after recording: clamped to [depth_min, depth_max] (holes at depth_min), on the 12-bit grid,
      rounded to whole millimetres. Depth stored as images is raw millimetres in the dataset too,
      so for it the cast is all.

    The cast matters on its own: the normalizer casts its stats to the input's dtype, so a uint16
    frame would be normalized in integer arithmetic.
    """

    depth_keys: list[str] = field(default_factory=list)
    # depth key → {"depth_min", "depth_max", "shift", "use_log"} (metres), from the dataset's info.json.
    encodings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def observation(self, observation: RobotObservation) -> RobotObservation:
        for key in self.depth_keys:
            if key not in observation:
                continue
            depth = torch.as_tensor(observation[key])
            if depth.is_floating_point():
                observation[key] = depth.to(torch.float32)
                continue
            device = depth.device
            # The round trip runs on the CPU (quantize_depth works on numpy), cast there as well.
            depth = depth.cpu().to(torch.float32)
            encoding = self.encodings.get(key)
            if encoding is not None:
                # Imported here: depth_utils needs PyAV, which importing the policy does not.
                from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth

                codes = quantize_depth(depth, **encoding, video_backend=None, input_unit="mm")
                depth = dequantize_depth(codes, **encoding, output_unit="mm")
            observation[key] = depth.to(device)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Depth keys keep their shape, only their dtype and values change."""
        return features

    def get_config(self) -> dict[str, Any]:
        return {"depth_keys": self.depth_keys, "encodings": self.encodings}


def make_cortex_arm_pre_post_processors(
    config: CortexArmConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """ACT's pipelines, with the depth step in front of device placement and normalization.

    Every depth input is prepared, also with ``use_depth=False``, so the normalizer never sees a
    uint16 frame.
    """

    depth_keys = list(config.depth_features)
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        CortexArmDepthProcessorStep(
            depth_keys=depth_keys,
            encodings={
                key: asdict(config.depth_encodings[key]) for key in depth_keys if key in config.depth_encodings
            },
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

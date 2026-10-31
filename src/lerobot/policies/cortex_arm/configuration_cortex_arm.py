# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

@author: Aadi
"""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature
from lerobot.configs.video import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
)
from lerobot.optim.optimizers import AdamWConfig


@dataclass
class DepthEncoding:
    """How a depth key is stored in its dataset: the quantization settings of LeRobot's depth
    encoder (``DepthEncoderConfig``), in metres. Defaults are the encoder's defaults."""

    depth_min: float = DEFAULT_DEPTH_MIN
    depth_max: float = DEFAULT_DEPTH_MAX
    shift: float = DEFAULT_DEPTH_SHIFT
    use_log: bool = DEFAULT_DEPTH_USE_LOG


@PreTrainedConfig.register_subclass("cortex_arm")
@dataclass
class CortexArmConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy, with a depth branch.

    Same inputs, defaults and training preset as ``act``. In addition, every visual input with one
    channel, i.e. a RealSense ``<cam>_depth`` key recorded with ``use_depth=True``, is encoded by
    its own ResNet instead of the colour one. Without depth keys in the dataset (or with
    ``use_depth=False``) the model is ACT, layer for layer.

    Depth is in millimetres, as the dataset returns it (default ``depth_output_unit``) and as the
    RealSense camera delivers it live, and it is normalized by the stock normalizer with the depth
    key's own dataset mean/std, like any other input. Before that, the preprocessor gives live
    frames the form recorded frames have in the dataset (see ``processor_cortex_arm.py``), using
    ``depth_encodings``.

    Added (depth):
        use_depth: Encode the depth keys. False ignores them, for an ablation on the same dataset
            (the preprocessor still prepares them, so the checkpoint runs with the same cameras).
        depth_encodings: Filled by ``make_policy`` from the dataset's ``info.json`` (not set by
            hand): for each depth key stored as video, the encoder settings it was quantized with.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone (colour and depth: same ResNet variant, separate weights).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # Depth.
    use_depth: bool = True
    depth_encodings: dict[str, DepthEncoding] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    @property
    def rgb_features(self) -> dict[str, PolicyFeature]:
        """Colour cameras: every visual input that is not a one-channel depth map."""
        return {key: ft for key, ft in self.image_features.items() if ft.shape[0] != 1}

    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        """Depth cameras among the inputs: visual inputs of shape ``(1, H, W)``, millimetres."""
        return {key: ft for key, ft in self.image_features.items() if ft.shape[0] == 1}

    @property
    def model_depth_features(self) -> dict[str, PolicyFeature]:
        """Depth cameras the model encodes: ``depth_features``, or none with ``use_depth=False``."""
        return self.depth_features if self.use_depth else {}

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        """Record how each depth key is stored in the dataset. Called by ``make_policy``.

        A depth video holds the 12-bit codes of LeRobot's depth encoder, whose settings
        ``info.json`` keeps as ``video.depth_min`` / ``video.depth_max`` / ``video.shift`` /
        ``video.use_log``, the same lookup the dataset reader uses to dequantize. Depth stored as
        images is raw millimetres and gets no entry.
        """
        self.depth_encodings = {}
        for key, ft in features.items():
            info = {**(ft.get("video_info") or {}), **(ft.get("info") or {})}
            is_depth = info.get("is_depth_map") or info.get("video.is_depth_map")
            if ft.get("dtype") != "video" or not is_depth:
                continue
            self.depth_encodings[key] = DepthEncoding(
                depth_min=float(info.get("video.depth_min", DEFAULT_DEPTH_MIN)),
                depth_max=float(info.get("video.depth_max", DEFAULT_DEPTH_MAX)),
                shift=float(info.get("video.shift", DEFAULT_DEPTH_SHIFT)),
                use_log=bool(info.get("video.use_log", DEFAULT_DEPTH_USE_LOG)),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.rgb_features and not self.model_depth_features and not self.env_state_feature:
            raise ValueError(
                "You must provide at least one image, depth map or the environment state among the inputs."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

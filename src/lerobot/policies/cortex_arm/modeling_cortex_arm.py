# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

@author: Aadi
"""
"""
cortex_arm: ACT with its own ResNet for depth.

Colour cameras go through ACT's ImageNet ResNet, unchanged. Depth cameras (one-channel visual
inputs, normalized millimetres) go through a second ResNet of the same variant whose first
convolution takes one channel, initialised from the colour filters summed over RGB, i.e. the
response the pretrained network gives a grey image. Each camera's feature map becomes tokens for
the same transformer encoder, colour and depth each through their own 1x1 projection. Separate
tokens per camera also mean depth does not have to be aligned to colour (LeRobot does not align).

Without depth inputs, or with ``use_depth=False``, this is ACT layer for layer, with ACT's
parameter names. The transformer, its layers and temporal ensembling are imported from
``act/modeling_act.py`` unchanged.
"""

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.cortex_arm.configuration_cortex_arm import CortexArmConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# Batch key the policy fills with the depth tensors, next to ACT's OBS_IMAGES for colour.
OBS_DEPTH_IMAGES = "observation.depth_images"


class CortexArmPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act),
    with depth inputs encoded by their own ResNet.
    """

    config_class = CortexArmConfig
    name = "cortex_arm"

    def __init__(
        self,
        config: CortexArmConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = CortexArm(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # Both ResNets train at the backbone learning rate, as ACT's single one does.
        backbones = ("model.backbone", "model.depth_backbone")
        return [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith(backbones) and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n.startswith(backbones) and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        batch = self._gather_cameras(batch)
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self._gather_cameras(batch)

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict

    def _gather_cameras(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Colour cameras under OBS_IMAGES (as ACT), depth cameras under OBS_DEPTH_IMAGES."""
        batch = dict(batch)  # shallow copy so that adding keys doesn't modify the original
        if self.config.rgb_features:
            batch[OBS_IMAGES] = [batch[key] for key in self.config.rgb_features]
        if self.config.model_depth_features:
            batch[OBS_DEPTH_IMAGES] = [batch[key] for key in self.config.model_depth_features]
        return batch


class CortexArm(nn.Module):
    """ACT's network (see ``ACT`` in ``act/modeling_act.py``) with a second ResNet for depth cameras.

    Transformer encoder tokens: [latent, (robot_state), (env_state), (colour feature map pixels),
    (depth feature map pixels)].
    """

    def __init__(self, config: CortexArmConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbones for image feature extraction. Note: The assumption here is that we are using a ResNet
        # model (and hence layer4 is the final feature map). The forward method of IntermediateLayerGetter
        # returns a dict: {"feature_map": output}.
        cam_feature_dim = None
        # Colour: ACT's backbone.
        if self.config.rgb_features:
            backbone_model = _make_resnet(config)
            cam_feature_dim = backbone_model.fc.in_features
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        # Depth: the same ResNet variant with a one-channel first convolution, its own weights.
        if self.config.model_depth_features:
            depth_model = _make_resnet(config)
            depth_model.conv1 = _one_channel_conv(depth_model.conv1)
            cam_feature_dim = depth_model.fc.in_features
            self.depth_backbone = IntermediateLayerGetter(depth_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (colour feature map pixels), (depth feature map pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.rgb_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(cam_feature_dim, config.dim_model, kernel_size=1)
        if self.config.model_depth_features:
            self.encoder_depth_feat_input_proj = nn.Conv2d(cam_feature_dim, config.dim_model, kernel_size=1)
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if cam_feature_dim is not None:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [rgb_features]: list of (B, 3, H, W) colour images, under OBS_IMAGES.
                AND/OR
            [model_depth_features]: list of (B, 1, H, W) normalized depth maps, under OBS_DEPTH_IMAGES.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        elif OBS_DEPTH_IMAGES in batch:
            batch_size = batch[OBS_DEPTH_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Camera tokens: colour through ACT's backbone, depth through its own, each with its own projection.
        cameras = []
        if self.config.rgb_features:
            cameras += [(img, self.backbone, self.encoder_img_feat_input_proj) for img in batch[OBS_IMAGES]]
        if self.config.model_depth_features:
            cameras += [
                (depth, self.depth_backbone, self.encoder_depth_feat_input_proj)
                for depth in batch[OBS_DEPTH_IMAGES]
            ]

        # For a list of images, the H and W may vary but H*W is constant.
        # NOTE: If modifying this section, verify on MPS devices that
        # gradients remain stable (no explosions or NaNs).
        for img, backbone, input_proj in cameras:
            cam_features = backbone(img)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = input_proj(cam_features)

            # Rearrange features to (sequence, batch, dim).
            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

            # Extend immediately instead of accumulating and concatenating
            # Convert to list to extend properly
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


def _make_resnet(config: CortexArmConfig) -> nn.Module:
    """ACT's image backbone: the torchvision ResNet variant, pretrained weights, frozen batch norm."""
    return getattr(torchvision.models, config.vision_backbone)(
        replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
        weights=config.pretrained_backbone_weights,
        norm_layer=FrozenBatchNorm2d,
    )


def _one_channel_conv(rgb_conv: nn.Conv2d) -> nn.Conv2d:
    """The ResNet's first convolution, for one-channel depth input.

    Starts from the colour filters summed over RGB: the response the pretrained network gives a grey
    image, so its edge detectors carry over to depth edges.
    """
    conv = nn.Conv2d(
        1,
        rgb_conv.out_channels,
        kernel_size=rgb_conv.kernel_size,
        stride=rgb_conv.stride,
        padding=rgb_conv.padding,
        bias=rgb_conv.bias is not None,
    )
    with torch.no_grad():
        conv.weight.copy_(rgb_conv.weight.sum(dim=1, keepdim=True))
        if rgb_conv.bias is not None:
            conv.bias.copy_(rgb_conv.bias)
    return conv

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from darkvggt.models.aggregator import Aggregator
from darkvggt.heads.camera_head import CameraHead
from darkvggt.heads.dpt_head import DPTHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True,
                 **kwargs):
        super().__init__()


        agg_kwargs = {
            k: kwargs[k] for k in
            ("enable_gstr", "gstr_rank_s", "gstr_rank_p", "gstr_mi_dim",
             "enable_phys", "zero_init_phys_proj",
             "phys_refl_identity_init_gain", "alpha_refl_logit_init",
             "alpha_gstr_logit_init",
             "gstr_gate_temperature", "gstr_router_init_std",
             "gstr_router_delta_znorm", "gstr_router_demean_lambda",
             "gstr_router_delta_clip", "gstr_router_use_bias",
             "gstr_router_bias_init",
             "alpha_gstr_min", "gstr_last_k_blocks")
            if k in kwargs
        }
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            **agg_kwargs,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None

        self.embed_dim = embed_dim
        self.rgb_only_lora_scale_train = float(
            kwargs.get("rgb_only_lora_scale_train", 1.0)
        )
        self.rgb_only_lora_scale_eval = float(
            kwargs.get("rgb_only_lora_scale_eval", 1.0)
        )
        for name, value in (
            ("rgb_only_lora_scale_train", self.rgb_only_lora_scale_train),
            ("rgb_only_lora_scale_eval", self.rgb_only_lora_scale_eval),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    def forward(self, images: torch.Tensor, thermal_images: torch.Tensor = None):
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if thermal_images is not None:
            if len(thermal_images.shape) == 4:
                thermal_images = thermal_images.unsqueeze(0)
            return self._forward_multimodal(images, thermal_images)


        return self._forward_standard(images)

    def _forward_standard(self, images):
        from darkvggt.layers.lora import lora_runtime_scale

        lora_scale = (
            self.rgb_only_lora_scale_train
            if self.training
            else self.rgb_only_lora_scale_eval
        )
        with lora_runtime_scale(self.aggregator, lora_scale):
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        return predictions

    def _forward_multimodal(self, rgb_images, thermal_images):
        B, S = rgb_images.shape[:2]
        H, W = rgb_images.shape[-2:]


        f_rgb = self.aggregator.encode(rgb_images)
        f_thermal = self.aggregator.encode(thermal_images)


        tokens_list, _, psi, uncertainty_info =\
            self.aggregator.forward_aa_multimodal(
                f_rgb, f_thermal, B, S, H, W, rgb_images.device,
            )

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    tokens_list, images=rgb_images, patch_start_idx=psi
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.camera_head is not None:
                pose_list = self.camera_head(tokens_list)
                predictions["pose_enc"] = pose_list[-1]
                predictions["pose_enc_list"] = pose_list


            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    tokens_list, images=rgb_images, patch_start_idx=psi
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        predictions["uncertainty_info"] = uncertainty_info
        predictions["is_multimodal"] = True

        return predictions

    def init_multimodal(self, lora_rank: int = 64, lora_alpha: float = 128.0):
        from darkvggt.layers.lora import apply_lora_to_module_list

        with torch.no_grad():
            self.aggregator.thermal_camera_token.data.copy_(
                self.aggregator.camera_token.data
            )


        apply_lora_to_module_list(
            self.aggregator.frame_blocks, rank=lora_rank, alpha=lora_alpha)
        apply_lora_to_module_list(
            self.aggregator.global_blocks, rank=lora_rank, alpha=lora_alpha)

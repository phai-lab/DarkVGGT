# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List, Dict

from darkvggt.layers import PatchEmbed
from darkvggt.layers.block import Block
from darkvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from darkvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class PhysGSTRFusionGate(nn.Module):
    """
    Phys-aware thermal factorization gate for RGB-T fusion.
    Implements Eq. (1)-(3) of Section 3.2.
    """

    ALPHA_MIN = 0.05

    def __init__(
        self,
        dim: int,
        rank: int = 64,
        enable_gstr: bool = False,
        rank_s: int = 64,
        rank_p: int = 64,
        mi_dim: int = 128,
        enable_phys: bool = True,
        zero_init_phys_proj: bool = True,
        phys_refl_identity_init_gain: float = 0.0,
        alpha_refl_logit_init: float = -2.0,
        alpha_gstr_logit_init: float = 0.0,
        alpha_gstr_min: float = 0.0,

        gstr_gate_temperature: float = 1.0,
        gstr_router_init_std: float = 0.01,
        gstr_router_delta_znorm: bool = True,
        gstr_router_demean_lambda: float = 1.0,
        gstr_router_delta_clip: float = 0.0,
        gstr_router_use_bias: bool = False,
        gstr_router_bias_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.enable_gstr = enable_gstr
        self.rank_s = rank_s
        self.rank_p = rank_p
        self.alpha_gstr_min = float(alpha_gstr_min)

        self.gstr_gate_temperature = float(gstr_gate_temperature)
        self.gstr_router_init_std = float(gstr_router_init_std)
        self.gstr_router_delta_znorm = bool(gstr_router_delta_znorm)
        self.gstr_router_demean_lambda = float(gstr_router_demean_lambda)
        self.gstr_router_delta_clip = float(gstr_router_delta_clip)
        self.gstr_router_use_bias = bool(gstr_router_use_bias)
        self.gstr_router_bias_init = float(gstr_router_bias_init)
        if self.gstr_router_delta_clip < 0.0:
            raise ValueError(
                "gstr_router_delta_clip must be non-negative, got "
                f"{self.gstr_router_delta_clip}"
            )
        self.phys_refl_identity_init_gain = float(
            phys_refl_identity_init_gain
        )
        if not 0.0 <= self.phys_refl_identity_init_gain <= 1.0:
            raise ValueError(
                "phys_refl_identity_init_gain must be in [0, 1], got "
                f"{self.phys_refl_identity_init_gain}"
            )
        self.alpha_refl_logit_init = float(alpha_refl_logit_init)


        self.mi_dim = mi_dim
        self.enable_phys = bool(enable_phys)


        self.proj_rgb = nn.Linear(dim, rank, bias=False)
        self.proj_thr = nn.Linear(dim, rank, bias=False)


        self.trunk = nn.Sequential(
            nn.LayerNorm(rank * 2),
            nn.Linear(rank * 2, rank),
            nn.GELU(),
        )


        self.head_gate_emit = nn.Linear(rank, dim)
        self.head_gate_refl = nn.Linear(rank, dim)
        self.head_emissivity = nn.Linear(rank, 1)


        self.proj_emit = nn.Linear(dim, dim, bias=False)
        self.proj_refl = nn.Linear(dim, dim, bias=False)


        self.zero_init_phys_proj = zero_init_phys_proj
        if self.phys_refl_identity_init_gain > 0.0 and not zero_init_phys_proj:
            raise ValueError(
                "phys_refl_identity_init_gain requires "
                "zero_init_phys_proj=True"
            )
        if zero_init_phys_proj:
            nn.init.zeros_(self.proj_emit.weight)
            if self.phys_refl_identity_init_gain > 0.0:
                with torch.no_grad():
                    nn.init.eye_(self.proj_refl.weight)
                    self.proj_refl.weight.mul_(
                        self.phys_refl_identity_init_gain
                    )
            else:
                nn.init.zeros_(self.proj_refl.weight)


        self.head_logvar_rgb = nn.Linear(rank, 1)
        self.head_logvar_thr = nn.Linear(rank, 1)


        self.alpha_emit_logit = nn.Parameter(torch.tensor(0.0))
        self.alpha_refl_logit = nn.Parameter(
            torch.tensor(self.alpha_refl_logit_init)
        )


        nn.init.constant_(self.head_emissivity.bias, 2.0)


        if self.enable_gstr:
            gstr_dim = rank_s + rank_p

            self.W_thr_gstr = nn.Linear(dim, gstr_dim, bias=False)

            self.W_rgb_gstr_shared = nn.Linear(dim, rank_s, bias=False)


            self.proj_rgb_gstr = nn.Linear(rank_s, mi_dim, bias=False)
            self.proj_thr_gstr = nn.Linear(rank_s, mi_dim, bias=False)


            self.gstr_norm_rgb = nn.LayerNorm(rank_s)
            self.gstr_norm_thr = nn.LayerNorm(rank_s)


            self.recon_gstr = nn.Linear(rank_p, dim, bias=False)
            nn.init.zeros_(self.recon_gstr.weight)


            self.gstr_up = nn.Linear(rank_s, dim, bias=False)
            nn.init.zeros_(self.gstr_up.weight)


            self.alpha_gstr_logit = nn.Parameter(torch.tensor(float(alpha_gstr_logit_init)))


            _router_init_std = self.gstr_router_init_std
            self.head_logvar_gstr_rgb = nn.Linear(rank_s, 1, bias=False)
            if _router_init_std > 0:
                nn.init.normal_(self.head_logvar_gstr_rgb.weight, mean=0.0, std=_router_init_std)
            else:
                nn.init.zeros_(self.head_logvar_gstr_rgb.weight)


            self.gstr_norm_thr_private = nn.LayerNorm(rank_p)
            self.head_logvar_gstr_private = nn.Linear(rank_p, 1, bias=False)
            if _router_init_std > 0:
                nn.init.normal_(self.head_logvar_gstr_private.weight, mean=0.0, std=_router_init_std)
            else:
                nn.init.zeros_(self.head_logvar_gstr_private.weight)
            self.w_gstr_gate = nn.Linear(
                2 * rank_s, 1, bias=self.gstr_router_use_bias
            )
            if _router_init_std > 0:
                nn.init.normal_(self.w_gstr_gate.weight, mean=0.0, std=_router_init_std)
            else:
                nn.init.zeros_(self.w_gstr_gate.weight)
            if self.w_gstr_gate.bias is not None:
                nn.init.constant_(
                    self.w_gstr_gate.bias, self.gstr_router_bias_init
                )


    @property
    def alpha_refl(self) -> torch.Tensor:
        return self.ALPHA_MIN + (1.0 - self.ALPHA_MIN) * torch.sigmoid(self.alpha_refl_logit)

    @property
    def alpha_emit(self) -> torch.Tensor:
        return self.ALPHA_MIN + (1.0 - self.ALPHA_MIN) * torch.sigmoid(self.alpha_emit_logit)

    @property
    def alpha_gstr(self) -> torch.Tensor:
        return self.alpha_gstr_min + (1.0 - self.alpha_gstr_min) * torch.sigmoid(self.alpha_gstr_logit)

    def _high_pass_tokens(
        self, x_thr: torch.Tensor, patch_grid_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """Extract high-frequency cues by removing local mean."""
        h, w = patch_grid_hw
        bs, p, c = x_thr.shape
        if p != h * w:
            raise ValueError(f"Patch count mismatch: expected {h*w}, got {p}")

        x_map = x_thr.transpose(1, 2).reshape(bs, c, h, w)
        blur = F.avg_pool2d(x_map, kernel_size=3, stride=1, padding=1)
        high = x_map - blur
        return high.reshape(bs, c, p).transpose(1, 2)

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_thr: torch.Tensor,
        patch_grid_hw: Tuple[int, int],
        enable_gstr_block: bool = True,
    ):
        """Phys-aware thermal factorization and fusion."""
        h_rgb = self.proj_rgb(x_rgb)
        h_thr = self.proj_thr(x_thr)
        h = self.trunk(torch.cat([h_rgb, h_thr], dim=-1))


        log_var_rgb = self.head_logvar_rgb(h)
        log_var_thr = self.head_logvar_thr(h)


        if not self.enable_phys:
            eps_hat = torch.ones_like(log_var_rgb)
            rho_hat = 1.0 - eps_hat
        else:
            eps_hat = torch.sigmoid(self.head_emissivity(h))
            rho_hat = 1.0 - eps_hat


        unc_boost = log_var_rgb - log_var_thr
        gate_emit = torch.sigmoid(self.head_gate_emit(h) + unc_boost)
        gate_refl = torch.sigmoid(self.head_gate_refl(h))
        gate = torch.clamp(gate_emit + gate_refl, min=0.0, max=1.0)


        if self.enable_phys:
            z_emit = eps_hat * self.proj_emit(x_thr)
            z_refl = rho_hat * self.proj_refl(
                self._high_pass_tokens(x_thr, patch_grid_hw)
            )
        else:
            z_emit = torch.zeros_like(x_thr)
            z_refl = torch.zeros_like(z_emit)


        inject = torch.zeros_like(z_emit)
        w_gstr = torch.ones_like(eps_hat)
        gstr_tuple = None
        if self.enable_gstr and enable_gstr_block:

            gstr_raw = self.W_thr_gstr(x_thr)
            s_rgb = self.W_rgb_gstr_shared(x_rgb)


            s_rgb_norm = self.gstr_norm_rgb(s_rgb)

            h_rgb_mi = F.normalize(self.proj_rgb_gstr(s_rgb_norm), dim=-1)
            gstr = gstr_raw

            s_thr = gstr[..., :self.rank_s]
            p_thr = gstr[..., self.rank_s:]
            s_thr_norm = self.gstr_norm_thr(s_thr)


            with torch.no_grad():
                h_thr_mi = F.normalize(self.proj_thr_gstr(s_thr_norm), dim=-1)


            x_thr_hat = self.recon_gstr(p_thr)


            inject_source = s_thr_norm.detach() - s_rgb_norm


            delta_u_gstr = torch.zeros_like(eps_hat)
            log_var_gstr_rgb = self.head_logvar_gstr_rgb(s_rgb_norm)
            p_thr_norm = self.gstr_norm_thr_private(p_thr)
            log_var_gstr_thr = self.head_logvar_gstr_private(p_thr_norm)
            h_gstr = torch.cat([s_rgb_norm, s_thr_norm], dim=-1)
            delta_u_gstr = log_var_gstr_rgb - log_var_gstr_thr
            delta_u_gate = delta_u_gstr
            if self.gstr_router_delta_znorm:
                lam = self.gstr_router_demean_lambda
                delta_u_centered = (
                    delta_u_gstr
                    - lam * delta_u_gstr.mean(dim=1, keepdim=True)
                )
                delta_u_std = delta_u_centered.std(
                    dim=1, keepdim=True, unbiased=False
                ).clamp_min(1e-6)
                delta_u_gate = delta_u_centered / delta_u_std
            if self.gstr_router_delta_clip > 0:
                delta_u_gate = delta_u_gate.clamp(
                    -self.gstr_router_delta_clip,
                    self.gstr_router_delta_clip,
                )
            gate_logit = self.w_gstr_gate(h_gstr) + delta_u_gate
            w_gstr = torch.sigmoid(
                gate_logit / max(self.gstr_gate_temperature, 1e-6)
            )
            inject = w_gstr * self.gstr_up(inject_source)


            gstr_tuple = (
                s_thr, p_thr, s_rgb,
                h_rgb_mi, h_thr_mi,
                x_thr_hat, inject,
                w_gstr, log_var_gstr_rgb, log_var_gstr_thr,
                s_thr_norm, s_rgb_norm, delta_u_gstr,
            )


        r_phys = (
            self.alpha_emit * (gate_emit * z_emit)
            + self.alpha_refl * (gate_refl * z_refl)
        )
        effective_inject = (
            self.alpha_gstr * inject
            if self.enable_gstr
            else torch.zeros_like(inject)
        )
        if gstr_tuple is not None:
            gstr_tuple = gstr_tuple + (r_phys.detach(), effective_inject.detach())
        fused_rgb = x_rgb + r_phys + effective_inject

        base = (fused_rgb,
                log_var_rgb, log_var_thr, gate,
                eps_hat, gate_emit, gate_refl,
                z_emit, z_refl)
        if gstr_tuple is not None:
            return base + gstr_tuple
        return base


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        **kwargs,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)


        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size


        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size


        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))


        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens


        self.thermal_camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))


        self.thermal_rope_offset = nn.Parameter(torch.zeros(2))


        self.enable_gstr = kwargs.get("enable_gstr", False)
        self.gstr_rank_s = kwargs.get("gstr_rank_s", 64)
        self.gstr_rank_p = kwargs.get("gstr_rank_p", 64)
        self.gstr_mi_dim = kwargs.get("gstr_mi_dim", 128)
        self.enable_phys = kwargs.get("enable_phys", True)
        self.zero_init_phys_proj = kwargs.get("zero_init_phys_proj", True)
        self.phys_refl_identity_init_gain = kwargs.get(
            "phys_refl_identity_init_gain", 0.0
        )
        self.alpha_refl_logit_init = kwargs.get("alpha_refl_logit_init", -2.0)
        self.alpha_gstr_logit_init = kwargs.get("alpha_gstr_logit_init", 0.0)
        self.alpha_gstr_min = kwargs.get("alpha_gstr_min", 0.0)
        self.gstr_gate_temperature = kwargs.get("gstr_gate_temperature", 1.0)
        self.gstr_router_init_std = kwargs.get("gstr_router_init_std", 0.01)
        self.gstr_router_delta_znorm = kwargs.get("gstr_router_delta_znorm", True)
        self.gstr_router_demean_lambda = kwargs.get("gstr_router_demean_lambda", 1.0)
        self.gstr_router_delta_clip = kwargs.get("gstr_router_delta_clip", 0.0)
        self.gstr_router_use_bias = kwargs.get("gstr_router_use_bias", False)
        self.gstr_router_bias_init = kwargs.get("gstr_router_bias_init", 0.0)
        self.gstr_last_k_blocks = int(kwargs.get("gstr_last_k_blocks", 0))
        self.fusion_gates = nn.ModuleList([
            PhysGSTRFusionGate(
                dim=embed_dim,
                enable_gstr=self.enable_gstr,
                rank_s=self.gstr_rank_s,
                rank_p=self.gstr_rank_p,
                mi_dim=self.gstr_mi_dim,
                enable_phys=self.enable_phys,
                zero_init_phys_proj=self.zero_init_phys_proj,
                phys_refl_identity_init_gain=(
                    self.phys_refl_identity_init_gain
                ),
                alpha_refl_logit_init=self.alpha_refl_logit_init,
                alpha_gstr_logit_init=self.alpha_gstr_logit_init,
                alpha_gstr_min=self.alpha_gstr_min,
                gstr_gate_temperature=self.gstr_gate_temperature,
                gstr_router_init_std=self.gstr_router_init_std,
                gstr_router_delta_znorm=self.gstr_router_delta_znorm,
                gstr_router_demean_lambda=self.gstr_router_demean_lambda,
                gstr_router_delta_clip=self.gstr_router_delta_clip,
                gstr_router_use_bias=self.gstr_router_use_bias,
                gstr_router_bias_init=self.gstr_router_bias_init,
            )
            for _ in range(self.aa_block_num)
        ])


        if self.enable_gstr:
            missing = []
            for i, gate in enumerate(self.fusion_gates):
                if not hasattr(gate, "head_logvar_gstr_private"):
                    missing.append(f"fusion_gates.{i}.head_logvar_gstr_private")
                if not hasattr(gate, "gstr_norm_thr_private"):
                    missing.append(f"fusion_gates.{i}.gstr_norm_thr_private")
            if missing:
                raise RuntimeError(
                    "GSTR private-router contract is not instantiated; "
                    "missing: " + ", ".join(missing)
                )

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        nn.init.normal_(self.thermal_camera_token, std=1e-6)


        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False  # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )


            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")


        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std


        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape


        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)


        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:


            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)


        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):

                    # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Input images [B, S, 3, H, W], range [0, 1].

        Returns:
            patch_tokens (torch.Tensor): Encoded features [B*S, P, C].
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        return patch_tokens

    def forward_aa_multimodal(
        self,
        rgb_tokens: torch.Tensor,
        thermal_tokens: torch.Tensor,
        B: int,
        S: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], int, Dict[str, List[torch.Tensor]]]:
        """
        Cross-modal alternating attention (Section 3.1).
        """
        P_single = rgb_tokens.shape[1]
        C = rgb_tokens.shape[2]
        psi = self.patch_start_idx
        P5 = P_single + psi


        cam_rgb = slice_expand_and_flatten(self.camera_token, B, S)
        cam_thr = slice_expand_and_flatten(self.thermal_camera_token, B, S)
        reg = slice_expand_and_flatten(self.register_token, B, S)


        rgb_tokens = torch.cat([cam_rgb, reg, rgb_tokens], dim=1)
        thr_tokens = torch.cat([cam_thr, reg, thermal_tokens], dim=1)

        grid_h = H // self.patch_size
        grid_w = W // self.patch_size


        pos_rgb = None
        pos_thr = None
        if self.rope is not None:
            pos_patches = self.position_getter(B * S, grid_h, grid_w, device=device)
            pos_patches = pos_patches + 1

            pos_special = torch.zeros(B * S, psi, 2, device=device, dtype=pos_patches.dtype)
            pos_rgb = torch.cat([pos_special, pos_patches], dim=1)


            pos_thr = pos_rgb + self.thermal_rope_offset

        frame_idx = 0
        global_idx = 0
        output_list: List[torch.Tensor] = []
        thr_output_list: List[torch.Tensor] = []
        all_log_var_rgb: List[torch.Tensor] = []
        all_log_var_thr: List[torch.Tensor] = []
        all_w_thr: List[torch.Tensor] = []
        all_eps_hat: List[torch.Tensor] = []
        all_gate_emit: List[torch.Tensor] = []
        all_gate_refl: List[torch.Tensor] = []
        all_z_emit: List[torch.Tensor] = []
        all_z_refl: List[torch.Tensor] = []
        all_raw_rgb: List[torch.Tensor] = []
        all_raw_thr: List[torch.Tensor] = []
        all_gstr_info: List[Dict[str, torch.Tensor]] = []

        for block_i in range(self.aa_block_num):

            if rgb_tokens.shape[0] != B * S:
                rgb_tokens = rgb_tokens.view(B * S, P5, C)
            if thr_tokens.shape[0] != B * S:
                thr_tokens = thr_tokens.view(B * S, P5, C)

            rgb_patches = rgb_tokens[:, psi:, :]
            thr_patches = thr_tokens[:, psi:, :]
            enable_gstr_block = True
            if self.enable_gstr and self.gstr_last_k_blocks > 0:
                enable_gstr_block = block_i >= (self.aa_block_num - self.gstr_last_k_blocks)
            fusion_gate = self.fusion_gates[block_i]
            patch_hw = (grid_h, grid_w)

            if self.training:
                result = checkpoint(
                    lambda rgb, thr, gate=fusion_gate, hw=patch_hw, gstr_on=enable_gstr_block: gate(
                        rgb,
                        thr,
                        hw,
                        enable_gstr_block=gstr_on,
                    ),
                    rgb_patches, thr_patches,
                    use_reentrant=self.use_reentrant)
            else:
                result = fusion_gate(
                    rgb_patches,
                    thr_patches,
                    patch_hw,
                    enable_gstr_block=enable_gstr_block,
                )


            (fused_rgb, lv_rgb, lv_thr, gate,
             eps_hat, gate_emit, gate_refl, z_emit, z_refl) = result[:9]
            gstr_info = None
            if len(result) > 9:
                (s_thr, p_thr, s_rgb,
                 h_rgb_mi, h_thr_mi,
                 x_thr_hat, inject,
                 w_gstr, log_var_gstr_rgb, log_var_gstr_thr,
                 s_thr_norm, s_rgb_norm, delta_u_gstr) = result[9:22]

                is_last_fusion_block = (block_i == self.aa_block_num - 1)
                r_phys_t = (
                    result[22]
                    if (is_last_fusion_block and len(result) > 22)
                    else None
                )
                eff_inject_t = (
                    result[23]
                    if (is_last_fusion_block and len(result) > 23)
                    else inject
                )

                log_var_gstr_rgb_cal = None
                log_var_gstr_thr_cal = None
                if self.training:
                    log_var_gstr_rgb_cal = (
                        fusion_gate.head_logvar_gstr_rgb(
                            s_rgb_norm.detach()
                        )
                    )
                    with torch.no_grad():
                        p_thr_norm_cal = (
                            fusion_gate.gstr_norm_thr_private(
                                p_thr.detach()
                            )
                        )
                    log_var_gstr_thr_cal = (
                        fusion_gate.head_logvar_gstr_private(
                            p_thr_norm_cal
                        )
                    )
                gstr_info = {


                    "block_index": block_i,
                    "s_thr":          s_thr,
                    "p_thr":          p_thr,
                    "s_rgb":          s_rgb,
                    "h_rgb_mi":       h_rgb_mi,
                    "h_thr_mi":       h_thr_mi,
                    "x_thr_hat":      x_thr_hat,
                    "x_thr":          thr_patches.detach(),
                    "inject":         inject,
                    "w_gstr":           w_gstr,
                    "log_var_gstr_rgb":  log_var_gstr_rgb,
                    "log_var_gstr_thr":  log_var_gstr_thr,
                    "log_var_gstr_rgb_cal": log_var_gstr_rgb_cal,
                    "log_var_gstr_thr_cal": log_var_gstr_thr_cal,
                    "s_thr_norm":     s_thr_norm,
                    "s_rgb_norm":     s_rgb_norm,
                    "delta_u_gstr":     delta_u_gstr,
                    "r_phys":         r_phys_t,
                    "effective_inject": eff_inject_t,
                }

            rgb_tokens = torch.cat([rgb_tokens[:, :psi, :], fused_rgb], dim=1)


            all_log_var_rgb.append(lv_rgb)
            all_log_var_thr.append(lv_thr)
            all_w_thr.append(gate)
            all_eps_hat.append(eps_hat)
            all_gate_emit.append(gate_emit)
            all_gate_refl.append(gate_refl)
            all_z_emit.append(z_emit)
            all_z_refl.append(z_refl)
            all_raw_rgb.append(rgb_patches.detach())
            all_raw_thr.append(thr_patches.detach())
            if gstr_info is not None:
                all_gstr_info.append(gstr_info)


            for attn_type in self.aa_order:
                if attn_type == "frame":

                    if rgb_tokens.shape[0] != B * S:
                        rgb_tokens = rgb_tokens.view(B * S, P5, C)
                    if thr_tokens.shape[0] != B * S:
                        thr_tokens = thr_tokens.view(B * S, P5, C)

                    pos_frame_rgb = pos_rgb
                    pos_frame_thr = pos_thr
                    if pos_rgb is not None and pos_rgb.shape[0] != B * S:
                        pos_frame_rgb = pos_rgb.view(B * S, P5, 2)
                        pos_frame_thr = pos_thr.view(B * S, P5, 2)

                    for bi in range(self.aa_block_size):
                        idx = frame_idx + bi
                        block = self.frame_blocks[idx]
                        if self.training:
                            rgb_tokens = checkpoint(
                                block, rgb_tokens, pos_frame_rgb,
                                use_reentrant=self.use_reentrant)
                        else:
                            rgb_tokens = block(rgb_tokens, pos=pos_frame_rgb)

                        if self.training:
                            thr_tokens = checkpoint(
                                block, thr_tokens, pos_frame_thr,
                                use_reentrant=self.use_reentrant)
                        else:
                            thr_tokens = block(thr_tokens, pos=pos_frame_thr)

                    frame_idx += self.aa_block_size
                    frame_inter = [rgb_tokens.view(B, S, P5, C)]
                    thr_frame_inter = [thr_tokens.view(B, S, P5, C)]

                elif attn_type == "global":

                    rgb_tokens, global_idx, global_inter =\
                        self._process_global_attention(
                            rgb_tokens, B, S, P5, C, global_idx, pos=pos_rgb)

                    thr_tokens, _, thr_global_inter =\
                        self._process_global_attention(
                            thr_tokens, B, S, P5, C, global_idx - self.aa_block_size,
                            pos=pos_thr)

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")


            for i in range(len(frame_inter)):
                output_list.append(
                    torch.cat([frame_inter[i], global_inter[i]], dim=-1))
                thr_output_list.append(
                    torch.cat([thr_frame_inter[i], thr_global_inter[i]], dim=-1))

        uncertainty_info = {
            "log_var_rgb": all_log_var_rgb,
            "log_var_thr": all_log_var_thr,
            "w_thr": all_w_thr,
            "eps_hat": all_eps_hat,
            "gate_emit": all_gate_emit,
            "gate_refl": all_gate_refl,
            "z_emit": all_z_emit,
            "z_refl": all_z_refl,
            "raw_rgb": all_raw_rgb,
            "raw_thr": all_raw_thr,
            "patch_grid_hw": (grid_h, grid_w),
            "gstr_info": all_gstr_info,
        }
        return output_list, thr_output_list, self.patch_start_idx, uncertainty_info

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """

        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []


        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []


        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """


    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])

    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])

    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)


    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

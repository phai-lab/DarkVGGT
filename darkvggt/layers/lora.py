


import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Frozen original weight W, trainable low-rank A and B:
        y = (W + (alpha/r) * B @ A) x + bias

    Args:
        original: The original nn.Linear to wrap (frozen).
        rank: LoRA rank r.
        alpha: LoRA scaling factor.
    """

    def __init__(self, original: nn.Linear, rank: int = 64, alpha: float = 128.0):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.runtime_scale = 1.0


        self.weight = original.weight
        self.bias = original.bias


        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))


        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = F.linear(x, self.weight, self.bias)

        effective_scale = self.scaling * self.runtime_scale
        if effective_scale != 0.0:
            out = out + F.linear(
                F.linear(x, self.lora_A), self.lora_B
            ) * effective_scale
        return out


@contextmanager
def lora_runtime_scale(module: nn.Module, scale: float):
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"LoRA runtime scale must be finite and non-negative, got {scale}")

    adapters = [submodule for submodule in module.modules()
                if isinstance(submodule, LoRALinear)]
    previous_scales = [adapter.runtime_scale for adapter in adapters]
    for adapter in adapters:
        adapter.runtime_scale = scale
    try:
        yield
    finally:
        for adapter, previous_scale in zip(adapters, previous_scales):
            adapter.runtime_scale = previous_scale


def apply_lora_to_block(block: nn.Module, rank: int = 64, alpha: float = 128.0):
    if rank <= 0:
        return

    if hasattr(block, 'attn'):
        attn = block.attn
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
        if hasattr(attn, 'proj') and isinstance(attn.proj, nn.Linear):
            attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha)


    if hasattr(block, 'mlp'):
        mlp = block.mlp
        if hasattr(mlp, 'fc1') and isinstance(mlp.fc1, nn.Linear):
            mlp.fc1 = LoRALinear(mlp.fc1, rank=rank, alpha=alpha)
        if hasattr(mlp, 'fc2') and isinstance(mlp.fc2, nn.Linear):
            mlp.fc2 = LoRALinear(mlp.fc2, rank=rank, alpha=alpha)


def apply_lora_to_module_list(module_list: nn.ModuleList, rank: int = 64, alpha: float = 128.0):
    if rank <= 0:
        return
    for block in module_list:
        apply_lora_to_block(block, rank=rank, alpha=alpha)

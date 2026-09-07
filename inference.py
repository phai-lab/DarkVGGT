#!/usr/bin/env python3
"""Shared helpers for loading DarkVGGT and its RGB-T inputs."""

import glob
import os
import tempfile

import torch

from darkvggt.config import load_model_kwargs
from darkvggt.models.vggt import VGGT
from darkvggt.utils.load_fn import load_and_preprocess_images
from darkvggt.utils.thermal import preprocess_thermal

EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")


def list_images(folder):
    paths = sorted(p for pattern in EXTENSIONS for p in glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise SystemExit(f"no images under {folder}")
    return paths


def load_thermal_images(thermal_dir, rgb_paths):
    """Load a folder of raw thermal frames the way the model expects to see them."""
    with tempfile.TemporaryDirectory(prefix="darkvggt-thermal-") as staging:
        paths = preprocess_thermal(list_images(thermal_dir), staging, rgb_paths)
        return load_and_preprocess_images(paths)


def build_model(vggt_checkpoint, darkvggt_checkpoint, device, config=None, enable_point=True):
    """Load DarkVGGT, or the plain VGGT-1B baseline when darkvggt_checkpoint is None."""
    kwargs = load_model_kwargs(config)
    kwargs["enable_point"] = enable_point
    model = VGGT(**kwargs)
    base = torch.load(vggt_checkpoint, map_location="cpu", weights_only=True)
    if isinstance(base, dict) and "model" in base:
        base = base["model"]
    filled = set(model.state_dict()) - set(model.load_state_dict(base, strict=False)[0])

    if darkvggt_checkpoint is None:
        return model.to(device).eval()

    model.init_multimodal(lora_rank=64, lora_alpha=128.0)
    adapters = torch.load(darkvggt_checkpoint, map_location="cpu", weights_only=False)
    if isinstance(adapters, dict) and "model" in adapters:
        adapters = adapters["model"]
    missing, unexpected = model.load_state_dict(adapters, strict=False)

    if unexpected:
        raise SystemExit(
            f"{darkvggt_checkpoint} has {len(unexpected)} tensors this model cannot hold, so "
            f"the config does not match the checkpoint: {unexpected[:5]}"
        )
    never_filled = [k for k in missing if k not in filled]
    if never_filled:
        raise SystemExit(
            f"{len(never_filled)} parameters stayed at their random init because neither "
            f"checkpoint supplies them: {never_filled[:5]}"
        )
    return model.to(device).eval()

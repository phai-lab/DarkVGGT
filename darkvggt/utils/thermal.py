import os

import cv2
import numpy as np


def preprocess_thermal(thermal_paths, output_dir, rgb_paths=None):
    """Infrared image processing"""
    os.makedirs(output_dir, exist_ok=True)

    target_hw = None
    for path in rgb_paths or []:
        reference = cv2.imread(path)
        if reference is not None:
            target_hw = reference.shape[:2]
            break

    written = []
    for path in thermal_paths:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise SystemExit(f"cannot read thermal frame {path}")
        if raw.ndim == 3:
            raw = raw[:, :, 0]

        frame = raw.astype(np.float32)
        low, high = np.percentile(frame, (1, 99))
        if high - low < 1e-3:
            low, high = frame.min(), frame.max()
        if high > low:
            frame = (np.clip(frame, low, high) - low) / (high - low) * 255.0
        else:
            frame = np.zeros_like(frame)

        equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            frame.astype(np.uint8))
        coloured = cv2.applyColorMap(equalised, cv2.COLORMAP_INFERNO)
        if target_hw is not None and coloured.shape[:2] != target_hw:
            coloured = cv2.resize(coloured, (target_hw[1], target_hw[0]))

        destination = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(destination, coloured)
        written.append(destination)
    return written

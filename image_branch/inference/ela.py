"""
Error Level Analysis (ELA) Module for FaceForensics++ C23 Image Branch.

# =========================================================================================
# TECHNICAL NOTE ON EXPLAINABILITY PAIRING:
# Error Level Analysis (ELA) is most meaningful for images that have gone through prior
# editing/pasting (splicing, boundary manipulation, face-swapping, multi-compression steps).
# It is a weaker signal for fully AI-generated (single-pass) synthetic identities, which is
# why ELA is paired with Grad-CAM (deep neural feature attention) rather than used alone.
# =========================================================================================
"""

import io
from pathlib import Path
from typing import Dict, Union, Tuple, Optional, Any

import numpy as np
from PIL import Image

import sys
from pathlib import Path

# Support relative & absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.preprocessing.preprocess import load_image


import cv2


def generate_ela(
    image_input: Union[str, Path, np.ndarray],
    quality: int = 90,
    scale: float = 15.0
) -> Dict[str, Any]:
    """
    Generates Error Level Analysis (ELA) visualization for an input image.

    Args:
        image_input (Union[str, Path, np.ndarray]): Filepath to image or RGB uint8 numpy array.
        quality (int): JPEG re-compression quality (1-100, default: 90).
        scale (float): Amplification scale multiplier for differences (default: 15.0).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'ela_image': RGB uint8 numpy array of ELA heatmap (H, W, 3)
            - 'mean_diff': Float mean pixel-wise difference across channels
            - 'max_diff': Float max pixel-wise difference
            - 'quality': Int compression quality used
            - 'scale': Float scale factor used
    """
    # 1. Load original RGB image array
    if isinstance(image_input, (str, Path)):
        orig_arr = load_image(str(image_input))
    elif isinstance(image_input, np.ndarray):
        orig_arr = image_input.copy()
        if orig_arr.ndim == 3 and orig_arr.shape[2] == 4:
            orig_arr = cv2.cvtColor(orig_arr, cv2.COLOR_RGBA2RGB)
        elif orig_arr.ndim == 2:
            orig_arr = cv2.cvtColor(orig_arr, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    orig_pil = Image.fromarray(orig_arr)

    # 2. Re-compress image in-memory using target JPEG quality
    buffer = io.BytesIO()
    orig_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed_pil = Image.open(buffer).convert("RGB")
    recompressed_arr = np.array(recompressed_pil, dtype=np.uint8)

    # 3. Calculate absolute difference
    diff_arr = np.abs(orig_arr.astype(np.float32) - recompressed_arr.astype(np.float32))

    mean_diff = float(np.mean(diff_arr))
    max_diff = float(np.max(diff_arr))

    # 4. Amplify differences for visualization (adapt scale if max_diff is subtle)
    effective_scale = scale
    if 0.0 < max_diff and (max_diff * scale) < 40.0:
        effective_scale = min(255.0 / max_diff, 45.0)

    ela_arr = np.clip(diff_arr * effective_scale, 0.0, 255.0).astype(np.uint8)

    return {
        "ela_image": ela_arr,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "quality": quality,
        "scale": effective_scale
    }


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    test_csv = PROJECT_ROOT / "data" / "splits" / "image_test.csv"
    if test_csv.is_file():
        import pandas as pd
        df = pd.read_csv(test_csv)
        sample_path = df["filepath"].iloc[0]
        print(f"[Smoke Test] Running ELA on sample: {sample_path}")
        res = generate_ela(sample_path)
        print(f"  ELA output shape: {res['ela_image'].shape} | Mean diff: {res['mean_diff']:.2f} | Max diff: {res['max_diff']:.2f}")

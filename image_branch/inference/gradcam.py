"""
Grad-CAM Explainability Module for FaceForensics++ C23 Image Classification.

Target Layer: Last convolutional layer of EfficientNet-B0 backbone (model.conv_head).
Outputs: Grad-CAM heatmap overlay, predicted class, confidence, and binary verdict.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
if hasattr(torch, "compiler"):
    torch.compiler.disable = lambda fn=None, recursive=True: (lambda f: f) if fn is None else fn
from PIL import Image
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.preprocessing.preprocess import load_image, get_eval_transforms

import cv2

# Label mapping constant
LABEL_MAPPING = {
    0: "Original",
    1: "FaceSwap",
    2: "Deepfakes",
    3: "Face2Face",
    4: "NeuralTextures",
    5: "FaceShifter"
}

_MODEL_CACHE = {}

# Active model checkpoint configuration
# Switched from efficientnet_b0_best.pth to efficientnet_b0_rebalanced.pth on 2026-09-25
# to address real-image false-positive bias (see reports/metrics/step_14_rebalancing_report.md
# for the full comparison). Original checkpoint retained for rollback if needed.
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_rebalanced.pth"


def load_trained_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, torch.device]:
    """
    Loads and caches the trained EfficientNet-B0 model checkpoint.
    """
    if checkpoint_path is None:
        checkpoint_path = ACTIVE_MODEL_PATH
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (str(checkpoint_path), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], device

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model, device


def generate_gradcam(
    image_input: Union[str, Path, np.ndarray],
    target_class: Optional[int] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Generates Grad-CAM heatmap overlay and model inference for a single input image.

    Args:
        image_input (Union[str, Path, np.ndarray]): Filepath to image or RGB uint8 numpy array.
        target_class (Optional[int]): Target class ID for Grad-CAM. If None, uses predicted class.
        checkpoint_path (Optional[Union[str, Path]]): Path to model checkpoint.
        model (Optional[nn.Module]): Pre-loaded model instance.
        device (Optional[torch.device]): Target torch device.

    Returns:
        Dict[str, Any]: Dictionary containing overlay image, heatmap, class ID, verdict, etc.
    """
    if model is None:
        model, device = load_trained_model(checkpoint_path, device)
    elif device is None:
        device = next(model.parameters()).device

    # 1. Load original RGB image array with format safety check
    if isinstance(image_input, (str, Path)):
        rgb_img_arr = load_image(str(image_input))
    elif isinstance(image_input, np.ndarray):
        rgb_img_arr = image_input.copy()
        if rgb_img_arr.ndim == 3 and rgb_img_arr.shape[2] == 4:
            rgb_img_arr = cv2.cvtColor(rgb_img_arr, cv2.COLOR_RGBA2RGB)
        elif rgb_img_arr.ndim == 2:
            rgb_img_arr = cv2.cvtColor(rgb_img_arr, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # 2. Apply deterministic eval transforms (224x224, Normalized tensor)
    eval_transform = get_eval_transforms(target_size=224)
    transformed = eval_transform(image=rgb_img_arr)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # 3. Model Forward Pass
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            logits = model(input_tensor)
        probs_tensor = torch.softmax(logits, dim=1).squeeze(0)

    probs_np = probs_tensor.cpu().numpy()
    pred_class_id = int(np.argmax(probs_np))
    confidence = float(probs_np[pred_class_id])
    pred_class_name = LABEL_MAPPING.get(pred_class_id, f"Class_{pred_class_id}")
    binary_verdict = "AUTHENTIC" if pred_class_id == 0 else "MANIPULATED"

    probs_dict = {LABEL_MAPPING[i]: float(probs_np[i]) for i in range(len(LABEL_MAPPING))}

    # Determine Grad-CAM target class
    cam_target_class = pred_class_id if target_class is None else target_class

    # 4. Generate Grad-CAM Heatmap using last conv layer (model.conv_head)
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(cam_target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Try releasing hooks cleanly
    try:
        cam.release()
    except Exception:
        pass

    # 5. Overlay Heatmap on resized RGB Image (0..1 float format for show_cam_on_image)
    pil_resized = Image.fromarray(rgb_img_arr).resize((224, 224), Image.Resampling.BILINEAR)
    resized_rgb_arr = np.array(pil_resized, dtype=np.uint8)
    rgb_float_01 = resized_rgb_arr.astype(np.float32) / 255.0

    # show_cam_on_image returns uint8 image array directly in range [0, 255]
    cam_overlay_uint8 = show_cam_on_image(rgb_float_01, grayscale_cam, use_rgb=True)

    return {
        "overlay_image": cam_overlay_uint8,
        "heatmap_raw": grayscale_cam,
        "pred_class_id": pred_class_id,
        "pred_class_name": pred_class_name,
        "confidence": confidence,
        "probs": probs_dict,
        "binary_verdict": binary_verdict,
        "original_image": resized_rgb_arr
    }


if __name__ == "__main__":
    test_csv = PROJECT_ROOT / "data" / "splits" / "image_test.csv"
    if test_csv.is_file():
        import pandas as pd
        df = pd.read_csv(test_csv)
        sample_path = df["filepath"].iloc[0]
        print(f"[Smoke Test] Running Grad-CAM on sample: {sample_path}")
        res = generate_gradcam(sample_path)
        print(f"  Predicted: {res['pred_class_name']} ({res['confidence']*100:.1f}%) | Verdict: {res['binary_verdict']}")
        print(f"  Overlay shape: {res['overlay_image'].shape} | Heatmap shape: {res['heatmap_raw'].shape}")

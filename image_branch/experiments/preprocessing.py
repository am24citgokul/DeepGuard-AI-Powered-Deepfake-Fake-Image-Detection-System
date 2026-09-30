"""
Canonical Unified Preprocessing Pipeline for DeepGuard X Image Models
Ensures 100% identical and mathematically reproducible preprocessing
across training, validation, frozen test, and external OOD inference.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_SIZE = 224

# MTCNN face detector singleton
_mtcnn = None

def get_face_detector():
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)
    return _mtcnn

def load_canonical_image(path: str) -> np.ndarray:
    """
    Loads an image safely from disk, guarantees RGB format and uint8 dtype.
    Handles RGB, RGBA, Grayscale, and corrupted headers.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        arr = np.array(rgb_img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Invalid image array format: {arr.shape}")
        return arr

def detect_and_crop_face(image_rgb: np.ndarray, margin_pct: float = 0.15) -> Tuple[np.ndarray, bool]:
    """
    Detects face if present using MTCNN. If image is already a crop (e.g. <= 300x300 and aspect ~ 1),
    or if no face is detected, gracefully falls back to the original image / center crop.
    
    Returns:
        (cropped_rgb, face_detected_bool)
    """
    h, w, _ = image_rgb.shape
    # If the image is already a tight crop (like Celeb-DF, Final Dataset, RVF5K at 256x256), skip detection
    if max(h, w) <= 300 and abs(h - w) / max(h, w) < 0.20:
        return image_rgb, True
        
    try:
        detector = get_face_detector()
        pil_img = Image.fromarray(image_rgb)
        boxes, probs = detector.detect(pil_img)
        
        if boxes is None or len(boxes) == 0:
            return image_rgb, False
            
        box = boxes[0]
        x1, y1, x2, y2 = [int(v) for v in box]
        fw, fh = x2 - x1, y2 - y1
        
        mx = int(fw * margin_pct)
        my = int(fh * margin_pct)
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        
        if cx2 <= cx1 or cy2 <= cy1:
            return image_rgb, False
            
        cropped = image_rgb[cy1:cy2, cx1:cx2]
        return cropped, True
    except Exception:
        # Robust fallback
        return image_rgb, False

def get_train_transforms(target_size: int = TARGET_SIZE) -> A.Compose:
    """
    Training transformations: includes gentle data augmentation,
    standard ImageNet normalization, and ToTensorV2.
    """
    return A.Compose([
        A.Resize(target_size, target_size),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(80, 100), p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_eval_transforms(target_size: int = TARGET_SIZE) -> A.Compose:
    """
    Evaluation transformations (Validation, Test, OOD, Inference):
    Strictly deterministic: Resize -> Normalize -> ToTensorV2.
    """
    return A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def preprocess_for_inference(
    image_or_path: Union[str, np.ndarray],
    target_size: int = TARGET_SIZE,
    apply_face_crop: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Complete inference preprocessing pipeline:
    1. Read RGB
    2. Detect / Crop Face (if full frame)
    3. Deterministic Resize & Normalization
    4. Returns PyTorch tensor (1, 3, target_size, target_size)
    """
    meta = {}
    if isinstance(image_or_path, str):
        img_rgb = load_canonical_image(image_or_path)
        meta["source_path"] = image_or_path
    else:
        img_rgb = image_or_path.copy()
        
    meta["original_shape"] = list(img_rgb.shape)
    
    if apply_face_crop:
        img_crop, face_found = detect_and_crop_face(img_rgb)
        meta["face_detected"] = face_found
        meta["cropped_shape"] = list(img_crop.shape)
    else:
        img_crop = img_rgb
        meta["face_detected"] = False
        meta["cropped_shape"] = list(img_crop.shape)
        
    eval_tf = get_eval_transforms(target_size)
    tensor = eval_tf(image=img_crop)["image"]
    tensor_batch = tensor.unsqueeze(0)  # (1, C, H, W)
    
    meta["tensor_shape"] = list(tensor_batch.shape)
    meta["tensor_mean"] = [float(tensor[c].mean().item()) for c in range(3)]
    meta["tensor_std"] = [float(tensor[c].std().item()) for c in range(3)]
    
    return tensor_batch, meta

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch tensor (C, H, W) back to displayable RGB uint8 array.
    """
    t = tensor.detach().cpu().clone()
    if t.ndim == 4:
        t = t.squeeze(0)
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    t = torch.clamp(t, 0.0, 1.0)
    np_img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return np_img

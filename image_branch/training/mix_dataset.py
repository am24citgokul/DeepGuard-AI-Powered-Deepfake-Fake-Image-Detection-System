"""
Dataset class for Mixed Image Dataset (mix_v2).
Supports Partial-Label Loss training with binary_label and subtype_label,
safe image loading across color modes, and advanced Albumentations data augmentation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_mix_train_transforms(target_size: int = 224) -> A.Compose:
    """
    Training augmentations:
    - RandomResizedCrop(224, scale 0.6-1.0, ratio 0.8-1.25)
    - HorizontalFlip (p=0.5)
    - JPEG compression (quality 40-95, p=0.6)
    - Downscale then upscale (scale 0.4-0.9, p=0.3) to simulate social media / phone compression
    - GaussianBlur (p=0.3)
    - RandomBrightnessContrast (p=0.5)
    - GaussNoise (p=0.3)
    - Normalize (ImageNet) + ToTensorV2
    """
    return A.Compose([
        A.RandomResizedCrop(
            size=(target_size, target_size),
            scale=(0.6, 1.0),
            ratio=(0.8, 1.25)
        ),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(40, 95), p=0.6),
        A.Downscale(scale_range=(0.4, 0.9), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_mix_eval_transforms(target_size: int = 224) -> A.Compose:
    """
    Deterministic evaluation transforms:
    - Resize(224, 224)
    - Normalize (ImageNet)
    - ToTensorV2
    """
    return A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def safe_load_rgb_image(filepath: str) -> Optional[np.ndarray]:
    """
    Safely loads an image from disk and converts it to an RGB numpy array uint8.
    Handles Grayscale, RGBA, CMYK, Palette modes.
    Returns None if unreadable or corrupted.
    """
    try:
        with Image.open(filepath) as img:
            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)
            if arr.size == 0 or len(arr.shape) != 3 or arr.shape[2] != 3:
                return None
            return arr
    except Exception as exc:
        logger.warning(f"Failed to load image at {filepath}: {exc}")
        return None


class MixedImageDataset(Dataset):
    """
    PyTorch Dataset for mix_v2 manifests.
    Yields:
      - image: torch.Tensor of shape (3, H, W)
      - binary_label: torch.Tensor int64 (0=Real, 1=Fake)
      - subtype_label: torch.Tensor int64 (0..5 for FF++, -1 for non-subtype fake)
      - filepath: str
    """
    def __init__(
        self,
        manifest_path: str,
        is_train: bool = False,
        transform: Optional[A.Compose] = None,
        target_size: int = 224
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.is_train = is_train
        self.df = pd.read_csv(manifest_path)
        self.failed_files = set()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_mix_train_transforms(target_size) if is_train else get_mix_eval_transforms(target_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        row = self.df.iloc[idx]
        fp = str(row["filepath"])
        bin_label = int(row["binary_label"])
        sub_label = int(row["subtype_label"]) if "subtype_label" in row and not pd.isna(row["subtype_label"]) else -1

        img_arr = safe_load_rgb_image(fp)
        if img_arr is None:
            self.failed_files.add(fp)
            # Fallback to neutral gray image to avoid crashing training
            img_arr = np.full((224, 224, 3), 128, dtype=np.uint8)

        augmented = self.transform(image=img_arr)
        img_tensor = augmented["image"]

        return img_tensor, bin_label, sub_label, fp

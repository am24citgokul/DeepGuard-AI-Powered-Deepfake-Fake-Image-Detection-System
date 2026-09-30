"""
Preprocessing utilities for FaceForensics++ C23 image classification.
Supports EfficientNet-B0 input sizing (224x224), ImageNet normalization,
and Albumentations-based data augmentation for training splits.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet standard normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(filepath: str) -> np.ndarray:
    """
    Loads and validates a single image from disk.

    Args:
        filepath (str): Absolute or relative path to the image file.

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be read or is corrupted.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image not found at path: {filepath}")

    try:
        with Image.open(filepath) as img:
            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img, dtype=np.uint8)
            if arr.size == 0 or len(arr.shape) != 3 or arr.shape[2] != 3:
                raise ValueError(f"Invalid image array format for {filepath}: shape {arr.shape}")
            return arr
    except Exception as exc:
        raise ValueError(f"Failed to load or parse image at {filepath}: {exc}") from exc


def resize_and_normalize(
    image: np.ndarray,
    target_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> np.ndarray:
    """
    Resizes an RGB image to target_size x target_size and normalizes pixel values
    using standard dataset mean and standard deviation.

    Args:
        image (np.ndarray): Input RGB image array of shape (H, W, 3), dtype uint8.
        target_size (int): Desired output spatial dimension (default: 224 for EfficientNet-B0).
        mean (Tuple[float, float, float]): Per-channel mean for normalization.
        std (Tuple[float, float, float]): Per-channel standard deviation for normalization.

    Returns:
        np.ndarray: Normalized float32 image array of shape (target_size, target_size, 3).
    """
    transform = A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=mean, std=std),
    ])
    augmented = transform(image=image)
    return augmented["image"]


def get_train_augmentations(
    target_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """
    Returns the Albumentations augmentation and preprocessing pipeline for training.
    Includes horizontal flip, mild JPEG compression, mild blur, mild brightness/contrast jitter,
    mild noise, and tensor normalization.

    Args:
        target_size (int): Output spatial dimension (default 224).
        mean (Tuple[float, float, float]): Mean for normalization.
        std (Tuple[float, float, float]): Std for normalization.

    Returns:
        A.Compose: Composed augmentation pipeline returning PyTorch Tensor (C, H, W).
    """
    return A.Compose([
        A.Resize(target_size, target_size),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_eval_transforms(
    target_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> A.Compose:
    """
    Returns the deterministic preprocessing pipeline for validation and testing.
    Applies only spatial resizing, normalization, and tensor conversion with no randomness.

    Args:
        target_size (int): Output spatial dimension (default 224).
        mean (Tuple[float, float, float]): Mean for normalization.
        std (Tuple[float, float, float]): Std for normalization.

    Returns:
        A.Compose: Deterministic pipeline returning PyTorch Tensor (C, H, W).
    """
    return A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

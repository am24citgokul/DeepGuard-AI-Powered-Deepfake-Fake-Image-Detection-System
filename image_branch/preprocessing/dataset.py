"""
PyTorch Dataset implementation for FaceForensics++ C23 image classification.
Supports train/val/test split manifests and applies appropriate Albumentations pipelines.
"""

import os
import time
from typing import Tuple, Optional, Union
import pandas as pd
import torch
from torch.utils.data import Dataset

# Support relative and absolute imports
try:
    from .preprocess import (
        load_image,
        get_train_augmentations,
        get_eval_transforms,
    )
except ImportError:
    from preprocess import (
        load_image,
        get_train_augmentations,
        get_eval_transforms,
    )


class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ Frame Dataset for image classification.
    Reads split CSVs (train/val/test) containing columns:
    [filename, filepath, label, label_id, group_id, split]
    """

    def __init__(
        self,
        split_input: Union[str, pd.DataFrame],
        split: Optional[str] = None,
        target_size: int = 224,
        transform=None
    ):
        """
        Args:
            split_input (Union[str, pd.DataFrame]): Path to split CSV file or loaded DataFrame.
            split (Optional[str]): Explicit split type ('train', 'val', or 'test').
                                  If None, inferred from the 'split' column of the CSV.
            target_size (int): Spatial resolution for the model input (default: 224).
            transform: Optional custom transform pipeline. If None, auto-selected based on split.
        """
        if isinstance(split_input, str):
            if not os.path.isfile(split_input):
                raise FileNotFoundError(f"Split CSV file not found: {split_input}")
            self.df = pd.read_csv(split_input)
        elif isinstance(split_input, pd.DataFrame):
            self.df = split_input.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Unsupported split_input type: {type(split_input)}")

        required_cols = {"filepath", "label_id"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV missing required columns: {required_cols - set(self.df.columns)}")

        # Determine split name
        if split is not None:
            self.split = split.lower().strip()
        elif "split" in self.df.columns and len(self.df) > 0:
            self.split = str(self.df["split"].iloc[0]).lower().strip()
        else:
            self.split = "train"

        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split name '{self.split}'. Must be 'train', 'val', or 'test'.")

        self.target_size = target_size

        # Assign Albumentations transform pipeline
        if transform is not None:
            self.transform = transform
        else:
            if self.split == "train":
                self.transform = get_train_augmentations(target_size=target_size)
            else:
                self.transform = get_eval_transforms(target_size=target_size)

        # Cache filepaths and labels for fast index access
        self.filepaths = self.df["filepath"].tolist()
        self.label_ids = self.df["label_id"].tolist()
        self.labels = self.df["label"].tolist() if "label" in self.df.columns else None
        self.filenames = self.df["filename"].tolist() if "filename" in self.df.columns else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a preprocessed sample.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, int]: (image_tensor of shape [3, target_size, target_size], label_id)
        """
        filepath = self.filepaths[idx]
        label_id = int(self.label_ids[idx])

        # Load RGB image
        image_arr = load_image(filepath)

        # Apply Albumentations transform pipeline
        transformed = self.transform(image=image_arr)
        image_tensor = transformed["image"]

        return image_tensor, label_id


if __name__ == "__main__":
    print("=" * 65)
    print("SMOKE TEST: FaceForensicsDataset (5 Samples from image_train.csv)")
    print("=" * 65)

    train_csv_path = r"E:\ADL_PROJECT\data\splits\image_train.csv"
    if not os.path.isfile(train_csv_path):
        train_csv_path = os.path.join("data", "splits", "image_train.csv")

    dataset = FaceForensicsDataset(split_input=train_csv_path, split="train")
    print(f"Dataset successfully initialized. Total samples: {len(dataset):,}")
    print(f"Split: '{dataset.split}' | Target Input Size: {dataset.target_size}x{dataset.target_size}")
    print("-" * 65)

    num_samples = 5
    latencies = []

    for i in range(num_samples):
        t0 = time.perf_counter()
        img_tensor, label_id = dataset[i]
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        filename = dataset.filenames[i] if dataset.filenames else f"sample_{i}"
        label_name = dataset.labels[i] if dataset.labels else f"class_{label_id}"

        print(f"Sample {i+1}:")
        print(f"  Filename:      {filename}")
        print(f"  Class Label:   {label_name} (label_id: {label_id})")
        print(f"  Tensor Shape:  {tuple(img_tensor.shape)} [C, H, W]")
        print(f"  Tensor Dtype:  {img_tensor.dtype}")
        print(f"  Min/Max Range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"  Process Time:  {dt_ms:.2f} ms")
        print("-" * 65)

    avg_time_ms = sum(latencies) / len(latencies)
    print(f"Smoke test completed successfully with 0 errors!")
    print(f"Average preprocessing latency: {avg_time_ms:.2f} ms per image ({1000/avg_time_ms:.1f} images/sec per worker)")
    print("=" * 65)

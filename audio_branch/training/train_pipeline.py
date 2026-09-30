"""
IdentityGuard AI — Audio Deepfake Detection Training, Validation, and Evaluation Pipeline.
Trains Baseline CNN and Primary AASIST on the verified In-The-Wild Audio dataset.
"""

import os
import sys
import json
import time
import random
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_branch.models.baseline_cnn import BaselineAudioCNN
from audio_branch.models.aasist import AASISTModel

# Set deterministic random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SAMPLES = 64000  # 4 seconds @ 16 kHz

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "audio"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = REPORTS_DIR / "audio_feature_examples"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


class AudioDataset(Dataset):
    def __init__(self, manifest_csv, split="train", augment=False):
        self.records = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)
        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        fp = row["filepath"]
        label = float(row["binary_label"])

        try:
            audio, sr = sf.read(fp)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono
        except Exception:
            # Fallback zero waveform in rare corrupted read
            audio = np.zeros(TARGET_SAMPLES, dtype=np.float32)

        audio_len = len(audio)

        # 4-Second Standardization Chunking
        if audio_len < TARGET_SAMPLES:
            # Symmetrical / right zero pad
            pad_amount = TARGET_SAMPLES - audio_len
            audio = np.pad(audio, (0, pad_amount), mode="constant")
        elif audio_len > TARGET_SAMPLES:
            if self.split == "train":
                max_start = audio_len - TARGET_SAMPLES
                start = random.randint(0, max_start)
                audio = audio[start : start + TARGET_SAMPLES]
            else:
                # Deterministic Center Crop for Validation and Test
                start = (audio_len - TARGET_SAMPLES) // 2
                audio = audio[start : start + TARGET_SAMPLES]
        else:
            audio = audio[:TARGET_SAMPLES]

        # Training Augmentation
        if self.augment and self.split == "train":
            # 1. Random Gain Variation (+/- 2dB)
            gain = random.uniform(0.85, 1.15)
            audio = audio * gain

            # 2. Low-level Background Noise (SNR ~25-30dB)
            if random.random() < 0.4:
                noise = np.random.normal(0, 0.001, size=audio.shape)
                audio = audio + noise

            # 3. Small Time Masking (1000 samples)
            if random.random() < 0.3:
                mask_start = random.randint(0, TARGET_SAMPLES - 1000)
                audio[mask_start : mask_start + 1000] = 0.0

        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        return audio_tensor, torch.tensor(label, dtype=torch.float32), row


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    fake_recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # Specificity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = 0.5

    return {
        "accuracy": acc,
        "precision": prec,
        "fake_recall": fake_recall,
        "real_recall": real_recall,
        "specificity": real_recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist()
    }


def compute_brier_and_ece(y_true, y_probs, n_bins=10):
    brier = float(np.mean((y_probs - y_true) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_probs >= bins[i]) & (y_probs < bins[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_probs[bin_mask])
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
    return brier, float(ece)

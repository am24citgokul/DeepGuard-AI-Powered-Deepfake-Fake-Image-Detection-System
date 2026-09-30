"""
Controlled Training & Validation Pipeline for DeepGuard X Image Branch
Supports:
- MODEL 1: Xception (legacy_xception)
- MODEL 2: EfficientNet (efficientnet_b0)
- MODEL 3: ViT-B/16 (torchvision vit_b_16)

Guarantees:
- Binary classification (Real: 0, Fake: 1)
- Identical frozen dataset splits across all models (frozen_train.csv, frozen_val.csv)
- Checkpoint selection ONLY using validation metrics (ROC-AUC / F1)
- Optimal threshold selected ONLY using validation data (never test or OOD data)
- Full metrics logged: Accuracy, Precision, Real Recall, Fake Recall, Specificity, FPR, FNR, F1, ROC-AUC, PR-AUC, Brier, ECE, Confusion Matrix
"""

import os
import sys
import time
import json
import argparse
import functools
from pathlib import Path
from typing import Dict, Any, Tuple, List

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path("E:/ADL_PROJECT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe patch for Windows App Control policy blocking optree._C in torch._dynamo
import types
mock_pytree = types.ModuleType('torch._dynamo.polyfills.pytree')
mock_pytree.__all__ = []
sys.modules['torch._dynamo.polyfills.pytree'] = mock_pytree

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.models as tvm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_SIZE = 224

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def get_train_transforms(target_size: int = TARGET_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(target_size, target_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.ImageCompression(quality_range=(80, 100), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_eval_transforms(target_size: int = TARGET_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

class FastImageDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = False, target_size: int = TARGET_SIZE):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        self.transform = get_train_transforms(target_size) if is_train else get_eval_transforms(target_size)
        self.filepaths = self.df["filepath"].values
        self.labels = self.df["binary_label"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        fp = self.filepaths[idx]
        lbl = int(self.labels[idx])
        try:
            bgr = cv2.imread(fp)
            if bgr is None:
                with Image.open(fp) as img:
                    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = np.full((TARGET_SIZE, TARGET_SIZE, 3), 128, dtype=np.uint8)

        tensor = self.transform(image=rgb)["image"]
        return tensor, lbl, fp

def create_model(arch: str) -> nn.Module:
    if arch == "xception":
        print("[Model] Initializing Xception (legacy_xception) with ImageNet pretrained weights...")
        model = timm.create_model("legacy_xception", pretrained=True, num_classes=2)
    elif arch == "efficientnet":
        print("[Model] Initializing EfficientNet-B0 with ImageNet pretrained weights...")
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    elif arch == "vit":
        print("[Model] Initializing ViT-B/16 (vit_b_16) with ImageNet pretrained weights...")
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets = []
    all_fake_probs = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            probs = torch.softmax(outputs, dim=-1)
            fake_p = probs[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()

            all_targets.extend(targets.cpu().numpy())
            all_fake_probs.extend(fake_p)
            all_preds.extend(preds)
            all_paths.extend(paths)

    avg_loss = total_loss / len(all_targets)
    return avg_loss, np.array(all_targets), np.array(all_fake_probs), np.array(all_preds), all_paths

def compute_detailed_metrics(y_true: np.ndarray, fake_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (fake_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    fake_rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    real_rec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, fake_probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, fake_probs))
    except Exception:
        pr_auc = 0.5

    brier = float(brier_score_loss(y_true, fake_probs))
    ece = float(compute_ece(fake_probs, y_true))

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "real_recall": real_rec,
        "fake_recall": fake_rec,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw": cm.tolist()
        }
    }

def find_best_threshold(y_true: np.ndarray, fake_probs: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    best_th = 0.5
    best_f1 = -1.0
    best_metrics = None
    for th in np.arange(0.10, 0.91, 0.02):
        m = compute_detailed_metrics(y_true, fake_probs, threshold=float(th))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = float(th)
            best_metrics = m
    return best_th, best_metrics

def main():
    parser = argparse.ArgumentParser(description="DeepGuard X Controlled Model Training")
    parser.add_argument("--model", type=str, required=True, choices=["xception", "efficientnet", "vit"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    arch = args.model
    exp_dir = PROJECT_ROOT / "image_branch" / "experiments" / arch
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_csv = PROJECT_ROOT / "image_branch" / "experiments" / "frozen_manifests" / "frozen_train.csv"
    val_csv = PROJECT_ROOT / "image_branch" / "experiments" / "frozen_manifests" / "frozen_val.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"DEEPGUARD X — PHASE 4: TRAINING MODEL [{arch.upper()}] ON {device}")
    print(f"  Train Manifest: {train_csv}")
    print(f"  Val Manifest:   {val_csv}")
    print(f"  Batch Size:     {args.batch_size} | Initial LR: {args.lr} | Epochs: {args.epochs}")
    print(f"================================================================================")

    train_ds = FastImageDataset(str(train_csv), is_train=True)
    val_ds = FastImageDataset(str(val_csv), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset loaded: {len(train_ds):,} train samples, {len(val_ds):,} val samples.")

    model = create_model(arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    history = []
    best_val_auc = -1.0
    best_ckpt_path = exp_dir / "best_model.pth"
    best_val_metrics = None
    best_threshold = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        for b_idx, (imgs, lbls, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item() * len(lbls)
            if b_idx % 100 == 0 or b_idx == len(train_loader):
                elapsed = time.time() - t0
                cur_loss = running_loss / (b_idx * args.batch_size)
                print(f"  Batch [{b_idx:03d}/{len(train_loader):03d}] | Train Loss: {cur_loss:.4f} | Elapsed: {elapsed:.1f}s")

        epoch_train_loss = running_loss / len(train_ds)
        scheduler.step()

        # Validation phase
        print(f"  Running Validation Evaluation on {len(val_ds):,} samples...")
        val_t0 = time.time()
        val_loss, y_true, fake_probs, y_pred_default, _ = evaluate(model, val_loader, device)
        val_dt = time.time() - val_t0

        # Compute default threshold metrics
        default_metrics = compute_detailed_metrics(y_true, fake_probs, threshold=0.5)
        # Select optimal threshold ONLY using validation data
        opt_th, opt_metrics = find_best_threshold(y_true, fake_probs)

        print(f"  [Epoch {epoch} Results]")
        print(f"    Train Loss:      {epoch_train_loss:.4f}")
        print(f"    Val Loss:        {val_loss:.4f} ({val_dt:.1f}s)")
        print(f"    Val Acc (th=0.5): {default_metrics['accuracy']*100:.2f}% | Val F1: {default_metrics['f1']:.4f}")
        print(f"    Val ROC-AUC:     {default_metrics['roc_auc']:.4f} | PR-AUC: {default_metrics['pr_auc']:.4f}")
        print(f"    Optimal Val Th:  {opt_th:.2f} (Acc: {opt_metrics['accuracy']*100:.2f}%, F1: {opt_metrics['f1']:.4f})")

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": val_loss,
            "val_metrics_default_th": default_metrics,
            "optimal_val_threshold": opt_th,
            "val_metrics_optimal_th": opt_metrics
        }
        history.append(epoch_record)

        # Checkpoint selection strictly based on validation ROC-AUC
        if default_metrics["roc_auc"] > best_val_auc:
            best_val_auc = default_metrics["roc_auc"]
            best_val_metrics = opt_metrics
            best_threshold = opt_th
            print(f"  >>> New Best Validation Model! ROC-AUC: {best_val_auc:.4f}. Saving checkpoint to {best_ckpt_path}...")
            torch.save({
                "epoch": epoch,
                "model_architecture": arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": default_metrics,
                "optimal_val_metrics": opt_metrics,
                "optimal_threshold": opt_th,
                "label_mapping": {0: "Real", 1: "Fake"}
            }, best_ckpt_path)

    # Save training log and final validation report
    log_path = exp_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
        
    val_report_path = exp_dir / "val_metrics.json"
    with open(val_report_path, "w") as f:
        json.dump({
            "model": arch,
            "checkpoint": str(best_ckpt_path),
            "best_val_auc": best_val_auc,
            "selected_threshold_from_val": best_threshold,
            "metrics_at_selected_threshold": best_val_metrics
        }, f, indent=2)

    print(f"\n[Completed] Model {arch.upper()} training complete.")
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Selected Validation Threshold: {best_threshold:.2f}")
    print(f"  Validation Report: {val_report_path}")

if __name__ == "__main__":
    main()

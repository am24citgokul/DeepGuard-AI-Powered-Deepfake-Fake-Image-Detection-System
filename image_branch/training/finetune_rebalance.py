"""
Class-Weighted Fine-Tuning Pipeline for EfficientNet-B0 Real-Image Bias Rebalancing.

Freezes the EfficientNet-B0 backbone (all convolutional feature extractors) and fine-tunes
ONLY the final linear classification head using a class-weighted CrossEntropyLoss that
upweights the Original (Real) class.

Target checkpoint saved to: models/efficientnet_b0_rebalanced.pth
Preserves: models/efficientnet_b0_best.pth (untouched)
"""

import os
import sys
import time
import argparse
import functools
from pathlib import Path
from typing import Dict, Tuple, List, Any

# Ensure stdout flushes immediately
print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from image_branch.preprocessing.dataset import FaceForensicsDataset

LABEL_MAPPING = {
    0: "Original",
    1: "FaceSwap",
    2: "Deepfakes",
    3: "Face2Face",
    4: "NeuralTextures",
    5: "FaceShifter"
}


def compute_metrics(targets: List[int], preds: List[int]) -> Dict[str, float]:
    targets_arr = np.array(targets)
    preds_arr = np.array(preds)

    # 6-class metrics
    acc_6class = accuracy_score(targets_arr, preds_arr)
    f1_macro_6class = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)

    # Binary metrics: Real = 0, Fake = 1..5
    bin_targets = (targets_arr != 0).astype(int)
    bin_preds = (preds_arr != 0).astype(int)
    bin_acc = accuracy_score(bin_targets, bin_preds)

    # Original class (Real) metrics where Real is positive label (1)
    real_targets = (targets_arr == 0).astype(int)
    real_preds = (preds_arr == 0).astype(int)
    real_rec = recall_score(real_targets, real_preds, pos_label=1, zero_division=0)
    real_prec = precision_score(real_targets, real_preds, pos_label=1, zero_division=0)
    real_f1 = f1_score(real_targets, real_preds, pos_label=1, zero_division=0)

    return {
        "acc_6class": float(acc_6class),
        "f1_macro_6class": float(f1_macro_6class),
        "binary_acc": float(bin_acc),
        "real_recall": float(real_rec),
        "real_precision": float(real_prec),
        "real_f1": float(real_f1)
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float], np.ndarray]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            preds = outputs.argmax(dim=1)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / total
    metrics = compute_metrics(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    return val_loss, metrics, cm


def run_finetuning(
    epochs: int = 5,
    patience: int = 2,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    orig_weight: float = 3.0,
    batch_size: int = 32,
    num_workers: int = 4
):
    print("=" * 80)
    print("  ADL PROJECT — STEP 14: CLASS-WEIGHTED HEAD FINE-TUNING")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_checkpoint_path = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
    rebalanced_checkpoint_path = PROJECT_ROOT / "models" / "efficientnet_b0_rebalanced.pth"

    if not base_checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found at: {base_checkpoint_path}")

    # 1. Load Data
    train_csv = PROJECT_ROOT / "data" / "splits" / "image_train.csv"
    val_csv = PROJECT_ROOT / "data" / "splits" / "image_val.csv"

    print(f"\n[Data Setup] Loading datasets...")
    train_dataset = FaceForensicsDataset(split_input=str(train_csv), split="train")
    val_dataset = FaceForensicsDataset(split_input=str(val_csv), split="val")

    print(f"  Train: {len(train_dataset):,} samples across 6 classes")
    print(f"  Val:   {len(val_dataset):,} samples across 6 classes")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Class Weights Formulation
    # Class 0 (Original) is upweighted to orig_weight (e.g. 3.0), Fake classes (1..5) are 1.0
    weights_list = [orig_weight, 1.0, 1.0, 1.0, 1.0, 1.0]
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)

    print(f"\n[Class Weights Setup]")
    print(f"  Class Weights Vector: {weights_list}")
    print(f"  Reasoning: Original (Class 0) upweighted by {orig_weight}x to counter 5:1 binary fake collapse.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model Setup: Load Base Model and Freeze Backbone
    print(f"\n[Model Setup] Loading base checkpoint from: {base_checkpoint_path}")
    base_ckpt = torch.load(base_checkpoint_path, map_location=device)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)
    model.load_state_dict(base_ckpt["model_state_dict"])
    model = model.to(device)

    # Freeze ALL backbone parameters
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "classifier" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (classifier head only: {trainable_params/total_params*100:.3f}%)")
    print(f"  Backbone frozen:      YES (requires_grad=False for all conv blocks)")

    # 4. Baseline Evaluation (Before Fine-Tuning)
    print(f"\n[Baseline Evaluation] Measuring existing model on validation set with class-weighted loss...")
    val_loss_base, metrics_base, cm_base = evaluate(model, val_loader, criterion, device)
    print(f"  Base Val Loss (weighted): {val_loss_base:.4f}")
    print(f"  Base 6-Class Acc:         {metrics_base['acc_6class']*100:.2f}%")
    print(f"  Base Binary Acc:          {metrics_base['binary_acc']*100:.2f}%")
    print(f"  Base Original Recall:     {metrics_base['real_recall']*100:.2f}%")
    print(f"  Base Original Precision:  {metrics_base['real_precision']*100:.2f}%")
    print(f"  Base Original F1:         {metrics_base['real_f1']:.4f}")

    # 5. Optimizer for Head Only
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # 6. Fine-Tuning Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_metrics = None

    print(f"\n[Training] Beginning head fine-tuning for max {epochs} epochs (patience={patience}, lr={lr})...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_sz = targets.size(0)
            running_train_loss += loss.item() * batch_sz
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            train_total += batch_sz

        train_loss = running_train_loss / train_total
        train_acc = train_correct / train_total

        # Validate
        val_loss, val_metrics, val_cm = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val 6-Class Acc: {val_metrics['acc_6class']*100:.2f}%")
        print(f"  Val Binary: Acc: {val_metrics['binary_acc']*100:.2f}% | "
              f"Real Recall: {val_metrics['real_recall']*100:.2f}% | "
              f"Real Prec: {val_metrics['real_precision']*100:.2f}% | "
              f"Real F1: {val_metrics['real_f1']:.4f}")

        # Check for improvement on validation loss
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0

            # Save to models/efficientnet_b0_rebalanced.pth
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_metrics["acc_6class"],
                "val_f1_macro": val_metrics["f1_macro_6class"],
                "val_binary_metrics": {
                    "binary_acc": val_metrics["binary_acc"],
                    "real_recall": val_metrics["real_recall"],
                    "real_precision": val_metrics["real_precision"],
                    "real_f1": val_metrics["real_f1"]
                },
                "label_mapping": LABEL_MAPPING,
                "model_architecture": "efficientnet_b0",
                "num_classes": 6,
                "class_weights": weights_list,
                "orig_weight": orig_weight,
                "finetune_type": "head_only_class_weighted"
            }
            torch.save(checkpoint, rebalanced_checkpoint_path)
            print(f"  >>> [Saved Rebalanced Model] Val loss improved to {val_loss:.4f} -> {rebalanced_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  [Early Stopping Counter] No improvement in val loss. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at epoch {epoch}. Stopping fine-tuning.")
                break

    print("\n" + "=" * 80)
    print("  FINE-TUNING COMPLETE")
    print(f"  Best Epoch:           {best_epoch}")
    print(f"  Best Val Loss:        {best_val_loss:.4f}")
    print(f"  Saved Rebalanced Ckpt: {rebalanced_checkpoint_path}")
    print(f"  Original Ckpt Intact: {base_checkpoint_path}")
    print("=" * 80)

    return metrics_base, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--orig-weight", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_finetuning(
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        orig_weight=args.orig_weight,
        batch_size=args.batch_size
    )

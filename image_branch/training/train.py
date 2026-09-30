"""
EfficientNet-B0 Baseline Training Pipeline for FaceForensics++ C23 Image Classification.

Tasks Handled:
1. Model: EfficientNet-B0 via timm (pretrained=True on ImageNet), adapted for 6 classes.
2. Dataset & DataLoader: FaceForensicsDataset with batch_size=32, num_workers=4.
3. Optimization: CrossEntropyLoss, AdamW (lr=1e-4, weight_decay=1e-5), ReduceLROnPlateau.
4. Early Stopping: patience=3 on validation loss (maximum 15 epochs).
5. Metrics: 6-class loss, accuracy, macro-F1 + Binary (Real vs Fake) accuracy, precision, recall, F1.
6. Artifacts:
   - models/efficientnet_b0_best.pth (saved ONLY when validation loss improves)
   - reports/training_curves/training_log.csv
   - reports/training_curves/training_vs_validation_loss.png
   - reports/training_curves/training_vs_validation_accuracy.png
"""

import os
import sys
import time
import argparse
import functools
from pathlib import Path
from typing import Dict, Tuple, List, Any

# Ensure all prints are flushed immediately for real-time progress logging
print = functools.partial(print, flush=True)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from image_branch.preprocessing.dataset import FaceForensicsDataset


# Label mapping constant
LABEL_MAPPING = {
    0: "Original",
    1: "FaceSwap",
    2: "Deepfakes",
    3: "Face2Face",
    4: "NeuralTextures",
    5: "FaceShifter"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 baseline for FaceForensics++ C23")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay (default: 1e-5)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on val loss (default: 3)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    parser.add_argument("--train-csv", type=str, default=str(PROJECT_ROOT / "data" / "splits" / "image_train.csv"))
    parser.add_argument("--val-csv", type=str, default=str(PROJECT_ROOT / "data" / "splits" / "image_val.csv"))
    parser.add_argument("--checkpoint-dir", type=str, default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "reports" / "training_curves"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def select_device_and_batch_size(requested_batch_size: int = 32) -> Tuple[torch.device, int]:
    """
    Selects GPU if available, reports GPU memory, and adjusts batch size if necessary.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device Selection] Using GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        
        # RTX 4050 has ~6GB VRAM, which comfortably supports batch_size=32
        if vram_gb < 3.5:
            batch_size = min(requested_batch_size, 16)
            print(f"[Memory Check] VRAM < 3.5 GB detected. Reduced batch size to: {batch_size}")
        else:
            batch_size = requested_batch_size
            print(f"[Memory Check] VRAM adequate for requested batch size: {batch_size}")
    else:
        device = torch.device("cpu")
        batch_size = min(requested_batch_size, 16)
        print(f"[Device Selection] CUDA not available. Running on CPU with batch size: {batch_size}")
    return device, batch_size


def compute_metrics(
    targets: List[int],
    preds: List[int]
) -> Dict[str, float]:
    """
    Computes both 6-class and binary (Real vs Fake) classification metrics.
    Real = class 0 (Original)
    Fake = classes 1..5 (FaceSwap, Deepfakes, Face2Face, NeuralTextures, FaceShifter)
    """
    targets_arr = np.array(targets)
    preds_arr = np.array(preds)

    # 6-class metrics
    acc_6class = accuracy_score(targets_arr, preds_arr)
    f1_macro_6class = f1_score(targets_arr, preds_arr, average="macro", zero_division=0)

    # Binary metrics: Real = 0, Fake = 1
    binary_targets = (targets_arr != 0).astype(int)
    binary_preds = (preds_arr != 0).astype(int)

    bin_acc = accuracy_score(binary_targets, binary_preds)
    bin_prec = precision_score(binary_targets, binary_preds, pos_label=1, zero_division=0)
    bin_rec = recall_score(binary_targets, binary_preds, pos_label=1, zero_division=0)
    bin_f1 = f1_score(binary_targets, binary_preds, pos_label=1, zero_division=0)

    return {
        "acc_6class": float(acc_6class),
        "f1_macro_6class": float(f1_macro_6class),
        "binary_acc": float(bin_acc),
        "binary_prec": float(bin_prec),
        "binary_rec": float(bin_rec),
        "binary_f1": float(bin_f1)
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """
    Runs one full training epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()
    num_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0 or batch_idx == num_batches:
            elapsed = time.time() - t0
            cur_loss = running_loss / total
            cur_acc = 100.0 * correct / total
            print(
                f"  Epoch [{epoch:02d}/{total_epochs:02d}] "
                f"Step [{batch_idx:03d}/{num_batches:03d}] "
                f"Train Loss: {cur_loss:.4f} | Train Acc: {cur_acc:.2f}% "
                f"({elapsed:.1f}s)"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates model on validation split.
    """
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

            running_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            _, preds = outputs.max(1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    val_loss = running_loss / total
    metrics = compute_metrics(all_targets, all_preds)
    return val_loss, metrics


def plot_diagnostic_curves(
    history: pd.DataFrame,
    output_dir: Path,
    best_epoch: int
):
    """
    Generates Task 2 diagnostic plots:
    1. training_vs_validation_loss.png
    2. training_vs_validation_accuracy.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"].values

    # Plot 1: Loss Curve
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2.2, color="#1f77b4", label="Training Loss")
    plt.plot(epochs, history["val_loss"], marker="s", linewidth=2.2, color="#d62728", label="Validation Loss")
    
    # Mark best checkpoint epoch
    best_row = history[history["epoch"] == best_epoch].iloc[0]
    plt.axvline(x=best_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Best Checkpoint (Epoch {best_epoch})")
    plt.scatter([best_epoch], [best_row["val_loss"]], color="#2ca02c", s=120, zorder=5, edgecolors="black")

    plt.title("FaceForensics++ C23: Training vs. Validation Loss (EfficientNet-B0)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
    plt.tight_layout()
    loss_fig_path = output_dir / "training_vs_validation_loss.png"
    plt.savefig(loss_fig_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Diagnostic loss curve saved to: {loss_fig_path}")

    # Plot 2: Accuracy Curve
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(epochs, history["train_acc"] * 100.0, marker="o", linewidth=2.2, color="#1f77b4", label="Training Accuracy (6-Class)")
    plt.plot(epochs, history["val_acc"] * 100.0, marker="s", linewidth=2.2, color="#d62728", label="Validation Accuracy (6-Class)")
    plt.axvline(x=best_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Best Checkpoint (Epoch {best_epoch})")
    plt.scatter([best_epoch], [best_row["val_acc"] * 100.0], color="#2ca02c", s=120, zorder=5, edgecolors="black")

    plt.title("FaceForensics++ C23: Training vs. Validation Accuracy (EfficientNet-B0)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11, loc="lower right")
    plt.tight_layout()
    acc_fig_path = output_dir / "training_vs_validation_accuracy.png"
    plt.savefig(acc_fig_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Diagnostic accuracy curve saved to: {acc_fig_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 75)
    print("  ADL PROJECT — STEP 4: EFFICIENTNET-B0 BASELINE TRAINING")
    print("=" * 75)

    # 1. Device and Batch Size
    device, batch_size = select_device_and_batch_size(args.batch_size)

    # 2. Datasets and DataLoaders
    print(f"\n[Data Setup] Initializing FaceForensicsDataset from CSV manifests:")
    print(f"  Train CSV: {args.train_csv}")
    print(f"  Val CSV:   {args.val_csv}")

    train_dataset = FaceForensicsDataset(split_input=args.train_csv, split="train")
    val_dataset = FaceForensicsDataset(split_input=args.val_csv, split="val")

    print(f"  Train samples: {len(train_dataset):,} across 6 classes")
    print(f"  Val samples:   {len(val_dataset):,} across 6 classes")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 3. Model Architecture: EfficientNet-B0 pretrained with 6-class classification head
    print(f"\n[Model Setup] Instantiating EfficientNet-B0 via timm (pretrained=True, num_classes=6)...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=6)
    model = model.to(device)

    # 4. Criterion, Optimizer, Scheduler, Scaler
    # Classes are balanced (3,500/class in train, 750/class in val), so unweighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # 5. Checkpoint paths and output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "efficientnet_b0_best.pth"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = output_dir / "training_log.csv"

    print(f"[Training Setup] Configuration:")
    print(f"  Optimizer:       AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Loss Function:   CrossEntropyLoss (Unweighted, balanced classes confirmed)")
    print(f"  Scheduler:       ReduceLROnPlateau (mode=min, factor=0.5, patience=1, min_lr=1e-6)")
    print(f"  Max Epochs:      {args.epochs}")
    print(f"  Early Stopping:  patience={args.patience} on validation loss")
    print(f"  Best Checkpoint: {best_checkpoint_path}")
    print(f"  Training Log:    {log_csv_path}")

    # 6. Training Loop
    history_records = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    print("\n" + "=" * 75)
    print("  BEGINNING MODEL TRAINING")
    print("=" * 75)

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n>>> Epoch [{epoch:02d}/{args.epochs:02d}] (Current LR: {current_lr:.2e})")

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs
        )

        # Validate
        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        # Update scheduler
        scheduler.step(val_loss)

        epoch_duration = time.time() - epoch_start_time

        # Print metrics
        print(f"--- Epoch [{epoch:02d}/{args.epochs:02d}] Summary ({epoch_duration:.1f}s) ---")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['acc_6class'] * 100:.2f}% | Val Macro F1: {val_metrics['f1_macro_6class']:.4f}")
        print(
            f"  [Binary Real vs Fake] Acc: {val_metrics['binary_acc'] * 100:.2f}% | "
            f"Prec: {val_metrics['binary_prec']:.4f} | "
            f"Rec: {val_metrics['binary_rec']:.4f} | "
            f"F1: {val_metrics['binary_f1']:.4f}"
        )

        # Record metrics
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_metrics["acc_6class"],
            "val_f1_macro": val_metrics["f1_macro_6class"],
            "val_binary_acc": val_metrics["binary_acc"],
            "val_binary_prec": val_metrics["binary_prec"],
            "val_binary_rec": val_metrics["binary_rec"],
            "val_binary_f1": val_metrics["binary_f1"],
            "lr": current_lr
        }
        history_records.append(record)

        # Check for improvement
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint ONLY when validation loss improves
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_metrics["acc_6class"],
                "val_f1_macro": val_metrics["f1_macro_6class"],
                "val_binary_metrics": {
                    "binary_acc": val_metrics["binary_acc"],
                    "binary_prec": val_metrics["binary_prec"],
                    "binary_rec": val_metrics["binary_rec"],
                    "binary_f1": val_metrics["binary_f1"]
                },
                "label_mapping": LABEL_MAPPING,
                "model_architecture": "efficientnet_b0",
                "num_classes": 6
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  >>> [Checkpoint Saved] Val loss improved by {improvement:.4f} -> {best_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  [Early Stopping Counter] No improvement in val loss. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping Triggered] Validation loss failed to improve for {args.patience} consecutive epochs.")
                early_stopped = True
                break

    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 75)
    print(f"  TRAINING COMPLETE ({total_training_time / 60:.2f} minutes total)")
    print(f"  Final Epoch Reached: {len(history_records)} | Early Stopped: {early_stopped}")
    print(f"  Best Epoch: {best_epoch} with Validation Loss: {best_val_loss:.4f}")
    print("=" * 75)

    # 7. Save Training Log CSV
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(log_csv_path, index=False)
    print(f"\n[Log Saved] Training log CSV successfully saved to: {log_csv_path}")

    # 8. Generate Diagnostic Plots (Task 2)
    plot_diagnostic_curves(history_df, output_dir, best_epoch)

    # 9. Print Best Epoch Summary
    best_record = history_df[history_df["epoch"] == best_epoch].iloc[0]
    print("\n" + "=" * 75)
    print(f"  BEST MODEL CHECKPOINT SUMMARY (Epoch {best_epoch})")
    print("=" * 75)
    print(f"  Checkpoint File:         {best_checkpoint_path} (Exists: {best_checkpoint_path.is_file()})")
    print(f"  Validation Loss:         {best_record['val_loss']:.4f}")
    print(f"  Validation Accuracy:     {best_record['val_acc'] * 100:.2f}% (6-class)")
    print(f"  Validation Macro F1:     {best_record['val_f1_macro']:.4f}")
    print(f"  Binary (Real vs Fake):")
    print(f"    Binary Accuracy:       {best_record['val_binary_acc'] * 100:.2f}%")
    print(f"    Binary Precision:      {best_record['val_binary_prec']:.4f}")
    print(f"    Binary Recall:         {best_record['val_binary_rec']:.4f}")
    print(f"    Binary F1 Score:       {best_record['val_binary_f1']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

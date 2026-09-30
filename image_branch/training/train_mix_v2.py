"""
Fine-Tuning Pipeline for EfficientNet-B0 on the Mixed Dataset (mix_v2).
Initializes from models/efficientnet_b0_best.pth (original checkpoint).
Trains the entire network using Partial-Label Loss, AdamW with differential learning rates,
and AMP on NVIDIA RTX 4050 GPU.
Saves the new checkpoint to models/efficientnet_b0_mix_v2.pth.
"""

import os
import sys
import time
import json
import logging
import functools
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure immediate terminal flushing
print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe patch for Windows App Control policy blocking optree._C in torch._dynamo
import sys, types
mock_pytree = types.ModuleType('torch._dynamo.polyfills.pytree')
mock_pytree.__all__ = []
sys.modules['torch._dynamo.polyfills.pytree'] = mock_pytree

import torch.compiler
torch.compiler.disable = lambda fn=None, recursive=True: (lambda f: f) if fn is None else fn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from image_branch.training.mix_dataset import MixedImageDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAPPING = {
    0: "Original",
    1: "FaceSwap",
    2: "Deepfakes",
    3: "Face2Face",
    4: "NeuralTextures",
    5: "FaceShifter"
}


os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

class PartialLabelLoss(nn.Module):
    """
    Numerically stable Partial-Label Loss:
    - Real (binary_label == 0): -log p[0]
    - Fake with subtype_label >= 0 (FF++): standard cross-entropy on that subtype class
    - Fake with subtype_label == -1 (Celeb-DF, Final Dataset): -log( sum_{c=1..5} p[c] )
      computed stably as: logsumexp(logits) - logsumexp(logits[:, 1:])
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        binary_labels: torch.Tensor,
        subtype_labels: torch.Tensor
    ) -> torch.Tensor:
        # Cast to float32 for AMP loss stability
        logits = logits.float()
        log_p = torch.log_softmax(logits, dim=-1)
        total_lse = torch.logsumexp(logits, dim=-1)
        fake_lse = torch.logsumexp(logits[:, 1:], dim=-1)

        loss = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)

        # 1. Real: binary_label == 0 -> -log p[0]
        mask_real = (binary_labels == 0)
        if mask_real.any():
            loss[mask_real] = -log_p[mask_real, 0]

        # 2. Fake with subtype: binary_label == 1 and subtype_label >= 0
        mask_sub = (binary_labels == 1) & (subtype_labels >= 0)
        if mask_sub.any():
            sub_targets = subtype_labels[mask_sub]
            loss[mask_sub] = -log_p[mask_sub, sub_targets]

        # 3. Fake with subtype == -1: binary_label == 1 and subtype_label == -1
        mask_nosub = (binary_labels == 1) & (subtype_labels == -1)
        if mask_nosub.any():
            loss[mask_nosub] = total_lse[mask_nosub] - fake_lse[mask_nosub]

        return loss.mean()


def evaluate_val(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_bin_targets = []
    all_p_fake = []

    with torch.no_grad():
        for images, bin_labels, sub_labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            bin_labels = bin_labels.to(device, non_blocking=True)
            sub_labels = sub_labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, bin_labels, sub_labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            # Softmax probabilities
            probs = torch.softmax(logits, dim=-1)
            p_fake = (1.0 - probs[:, 0]).cpu().numpy()
            all_p_fake.extend(p_fake)
            all_bin_targets.extend(bin_labels.cpu().numpy())

    val_loss = running_loss / total
    y_true = np.array(all_bin_targets)
    p_fake_arr = np.array(all_p_fake)
    y_pred = (p_fake_arr >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    real_rec = recall_score(y_true == 0, y_pred == 0, pos_label=True, zero_division=0)
    fake_rec = recall_score(y_true == 1, y_pred == 1, pos_label=True, zero_division=0)
    bal_acc = (real_rec + fake_rec) / 2.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "val_loss": float(val_loss),
        "binary_acc": float(acc),
        "balanced_acc": float(bal_acc),
        "real_recall": float(real_rec),
        "fake_recall": float(fake_rec),
        "f1": float(f1)
    }
    return val_loss, metrics


def train_mix_v2():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  ADL PROJECT — STAGE 2: FINE-TUNING ON MIXED DATASET (mix_v2)")
    print("=" * 80)
    print(f"[Device] Using: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_checkpoint_path = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
    output_checkpoint_path = PROJECT_ROOT / "models" / "efficientnet_b0_mix_v2.pth"
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = metrics_dir / "mix_v2_training_log.csv"
    config_json_path = metrics_dir / "mix_v2_config.json"

    # Hyperparameters
    batch_size = 32
    num_workers = 4
    max_epochs = 6
    patience = 2
    backbone_lr = 2e-5
    head_lr = 1e-4
    weight_decay = 1e-5

    config = {
        "model_architecture": "efficientnet_b0",
        "init_checkpoint": str(base_checkpoint_path),
        "output_checkpoint": str(output_checkpoint_path),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=min, factor=0.5, patience=1)",
        "loss": "PartialLabelLoss",
        "seed": seed
    }
    with open(config_json_path, "w") as f:
        json.dump(config, f, indent=2)

    train_manifest = PROJECT_ROOT / "data" / "mix_v2" / "mix_train.csv"
    val_manifest = PROJECT_ROOT / "data" / "mix_v2" / "mix_val.csv"

    print(f"\n[Data Setup] Loading datasets...")
    train_dataset = MixedImageDataset(str(train_manifest), is_train=True)
    val_dataset = MixedImageDataset(str(val_manifest), is_train=False)

    print(f"  Train: {len(train_dataset):,} samples (1:1 Real:Fake)")
    print(f"  Val:   {len(val_dataset):,} samples")

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

    # Model Setup
    print(f"\n[Model Setup] Loading base checkpoint from: {base_checkpoint_path}")
    base_ckpt = torch.load(base_checkpoint_path, map_location=device)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)
    model.load_state_dict(base_ckpt["model_state_dict"])
    model = model.to(device)

    # Differential Learning Rates: backbone vs head
    head_params = list(model.classifier.parameters())
    head_ids = set(map(id, head_params))
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    criterion = PartialLabelLoss()

    print(f"[Training Setup] AdamW (backbone={backbone_lr}, head={head_lr}), ReduceLROnPlateau, AMP=True")
    print(f"[Training Setup] Early Stopping patience={patience} on mix_val loss")

    # Initial Validation check before training
    init_val_loss, init_val_metrics = evaluate_val(model, val_loader, criterion, device)
    print(f"[Init Model Eval] Val Loss: {init_val_loss:.4f} | Binary Acc: {init_val_metrics['binary_acc']*100:.2f}% | Bal Acc: {init_val_metrics['balanced_acc']*100:.2f}% | Real Rec: {init_val_metrics['real_recall']*100:.2f}% | Fake Rec: {init_val_metrics['fake_recall']*100:.2f}%")

    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []

    print("\n" + "=" * 80)
    print("STARTING TRAINING LOOP")
    print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        running_train_loss = 0.0
        train_total = 0

        for images, bin_labels, sub_labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            bin_labels = bin_labels.to(device, non_blocking=True)
            sub_labels = sub_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, bin_labels, sub_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_cur = images.size(0)
            running_train_loss += loss.item() * batch_size_cur
            train_total += batch_size_cur

        epoch_train_loss = running_train_loss / train_total

        # Validation
        val_loss, val_metrics = evaluate_val(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr_head = optimizer.param_groups[1]["lr"]
        epoch_time = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": float(epoch_train_loss),
            "val_loss": float(val_loss),
            "val_binary_acc": float(val_metrics["binary_acc"]),
            "val_balanced_acc": float(val_metrics["balanced_acc"]),
            "val_real_recall": float(val_metrics["real_recall"]),
            "val_fake_recall": float(val_metrics["fake_recall"]),
            "val_f1": float(val_metrics["f1"]),
            "lr_head": float(current_lr_head),
            "epoch_time_sec": float(epoch_time)
        }
        training_log.append(log_entry)
        pd.DataFrame(training_log).to_csv(training_log_path, index=False)

        # Single line epoch summary
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Bal Acc: {val_metrics['balanced_acc']*100:.2f}% | "
            f"Val Real Rec: {val_metrics['real_recall']*100:.2f}% | "
            f"Val Fake Rec: {val_metrics['fake_recall']*100:.2f}% | "
            f"Head LR: {current_lr_head:.2e}"
        )

        # Checkpoint saving & Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "label_mapping": LABEL_MAPPING,
                "model_architecture": "efficientnet_b0",
                "num_classes": 6,
                "config": config
            }, output_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {output_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  --> Val loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch}.")
                break

    print("=" * 80)
    print(f"Training completed! Best Val Loss: {best_val_loss:.4f}")
    print(f"New model saved to: {output_checkpoint_path}")
    print(f"Training log saved to: {training_log_path}")
    if len(train_dataset.failed_files) > 0:
        print(f"WARNING: {len(train_dataset.failed_files)} files failed to load during training.")
    else:
        print("All training files loaded cleanly with 0 load failures.")


if __name__ == "__main__":
    train_mix_v2()

"""
Master Comprehensive Evaluation Engine for DeepGuard X Image Branch
Executes:
- Phase 6: Frozen Test Evaluation (In-Distribution, 4,004 images)
- Phase 7: Frozen OOD Evaluation (RVF5K, 5,000 images)
- Phase 8: Systematic Error Analysis (False Negatives & False Positives Contact Sheets)
- Phase 9: Generalization Gap Analysis
- Phase 11: Multi-Model Comparison Table & Visualizations
- Phase 12: High-Resolution Publication-Ready Plots
"""

import os
import sys
import time
import json
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
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.models as tvm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)

from image_branch.experiments.train_controlled import (
    FastImageDataset,
    compute_detailed_metrics,
    compute_ece,
    create_model
)

FROZEN_DIR = PROJECT_ROOT / "image_branch" / "experiments" / "frozen_manifests"
EXP_DIR = PROJECT_ROOT / "image_branch" / "experiments"
REPORTS_DIR = EXP_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["xception", "efficientnet", "vit"]

def load_checkpoint_and_model(arch: str, device: torch.device):
    ckpt_path = EXP_DIR / arch / "best_model.pth"
    val_json_path = EXP_DIR / arch / "val_metrics.json"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device)
    model = create_model(arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    threshold = 0.5
    if val_json_path.exists():
        with open(val_json_path) as f:
            v_data = json.load(f)
            threshold = float(v_data.get("selected_threshold_from_val", 0.5))
    elif "optimal_threshold" in ckpt:
        threshold = float(ckpt["optimal_threshold"])
        
    print(f"Loaded {arch.upper()} from {ckpt_path} (Validation Threshold = {threshold:.2f})")
    return model, ckpt_path, threshold

def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    model_name: str,
    ckpt_path: Path
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    all_targets = []
    all_fake_probs = []
    all_real_probs = []
    all_preds = []
    all_paths = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets, paths) in enumerate(dataloader, 1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            fake_p = probs[:, 1].cpu().numpy()
            real_p = probs[:, 0].cpu().numpy()
            preds = (fake_p >= threshold).astype(int)

            all_targets.extend(targets.numpy().tolist())
            all_fake_probs.extend(fake_p.tolist())
            all_real_probs.extend(real_p.tolist())
            all_preds.extend(preds.tolist())
            all_paths.extend(paths)

    eval_time = time.time() - t0
    y_true = np.array(all_targets)
    fake_probs = np.array(all_fake_probs)
    real_probs = np.array(all_real_probs)
    y_pred = np.array(all_preds)

    metrics = compute_detailed_metrics(y_true, fake_probs, threshold=threshold)
    metrics["eval_time_sec"] = float(eval_time)
    metrics["throughput_img_sec"] = float(len(y_true) / eval_time)

    # Prediction dataframe
    df_preds = pd.DataFrame({
        "filepath": all_paths,
        "true_label": y_true,
        "predicted_label": y_pred,
        "real_probability": real_probs,
        "fake_probability": fake_probs,
        "correct": (y_true == y_pred),
        "model": model_name,
        "checkpoint": str(ckpt_path)
    })

    return metrics, df_preds

def generate_evaluation_plots(
    y_true: np.ndarray,
    fake_probs: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, Any],
    model_name: str,
    prefix: str,
    out_dir: Path
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    cm = np.array(metrics["confusion_matrix"]["raw"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    labels = ["Real (0)", "Fake (1)"]
    annotations = np.array([[f"{count}\n({pct:.1%})" for count, pct in zip(row_cnt, row_pct)]
                            for row_cnt, row_pct in zip(cm, cm_norm)])
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"{model_name.upper()} — {prefix.upper()} Confusion Matrix\nAccuracy: {metrics['accuracy']*100:.2f}% | Th: {metrics['threshold']:.2f}", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=200)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, fake_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=10)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=10)
    plt.title(f"{model_name.upper()} — {prefix.upper()} ROC Curve", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc_curve.png", dpi=200)
    plt.close()

    # 3. Precision-Recall Curve Plot
    precisions, recalls, _ = precision_recall_curve(y_true, fake_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, color="blue", lw=2, label=f"PR (AUC = {metrics['pr_auc']:.4f})")
    plt.xlabel("Recall", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.title(f"{model_name.upper()} — {prefix.upper()} Precision-Recall Curve", fontsize=11, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pr_curve.png", dpi=200)
    plt.close()

    # 4. Confidence Histogram
    plt.figure(figsize=(7, 4.5))
    plt.hist(fake_probs[y_true == 0], bins=30, alpha=0.6, color="green", label="True Real (0)", density=True)
    plt.hist(fake_probs[y_true == 1], bins=30, alpha=0.6, color="red", label="True Fake (1)", density=True)
    plt.axvline(metrics["threshold"], color="black", linestyle="--", lw=2, label=f"Threshold ({metrics['threshold']:.2f})")
    plt.xlabel("Predicted Fake Probability", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.title(f"{model_name.upper()} — {prefix.upper()} Confidence Distribution", fontsize=11, fontweight="bold")
    plt.legend(loc="upper center")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confidence_histogram.png", dpi=200)
    plt.close()

def build_contact_sheet(
    images_info: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    grid_cols: int = 10,
    grid_rows: int = 5,
    thumb_size: Tuple[int, int] = (120, 120)
):
    tw, th = thumb_size
    pad = 8
    header_h = 50
    sheet_w = grid_cols * (tw + pad) + pad
    sheet_h = header_h + grid_rows * (th + pad + 25) + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(25, 25, 30))
    draw = ImageDraw.Draw(sheet)

    # Title
    draw.text((pad + 10, 15), title, fill=(255, 255, 255))

    font_color_err = (255, 100, 100)

    for idx, item in enumerate(images_info[:grid_cols * grid_rows]):
        r = idx // grid_cols
        c = idx % grid_cols
        x = pad + c * (tw + pad)
        y = header_h + pad + r * (th + pad + 25)

        fp = item["filepath"]
        try:
            with Image.open(fp) as img:
                thumb = img.convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
                sheet.paste(thumb, (x, y))
        except Exception:
            err_box = Image.new("RGB", (tw, th), color=(80, 20, 20))
            sheet.paste(err_box, (x, y))

        p_str = f"P(Real): {item['real_probability']:.2f}" if "real_probability" in item else f"P(Fake): {item['fake_probability']:.2f}"
        lbl_str = f"T:{item['true_label']} | P:{item['predicted_label']}"
        draw.text((x, y + th + 2), lbl_str, fill=(200, 200, 200))
        draw.text((x, y + th + 14), p_str, fill=font_color_err)

    sheet.save(output_path, quality=90)
    print(f"Contact sheet saved: {output_path}")

def perform_error_analysis(df_preds: pd.DataFrame, model_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fn_dir = out_dir / "image_false_negatives"
    fp_dir = out_dir / "image_false_positives"
    fn_dir.mkdir(parents=True, exist_ok=True)
    fp_dir.mkdir(parents=True, exist_ok=True)

    # 4 Groups
    true_real = df_preds[(df_preds["true_label"] == 0) & (df_preds["predicted_label"] == 0)]
    true_fake = df_preds[(df_preds["true_label"] == 1) & (df_preds["predicted_label"] == 1)]
    false_positives = df_preds[(df_preds["true_label"] == 0) & (df_preds["predicted_label"] == 1)]
    false_negatives = df_preds[(df_preds["true_label"] == 1) & (df_preds["predicted_label"] == 0)]

    print(f"\n[{model_name.upper()} Error Breakdown on Test Set]")
    print(f"  True Real:       {len(true_real):,}")
    print(f"  True Fake:       {len(true_fake):,}")
    print(f"  False Positives: {len(false_positives):,} (Real predicted as Fake)")
    print(f"  False Negatives: {len(false_negatives):,} (Fake/AI predicted as Real)")

    # False Negatives sorted by highest REAL confidence (worst failures)
    fn_sorted = false_negatives.sort_values(by="real_probability", ascending=False)
    fn_sorted.to_csv(fn_dir / f"{model_name}_false_negatives.csv", index=False)

    top50_fn = fn_sorted.head(50).to_dict("records")
    build_contact_sheet(
        top50_fn,
        fn_dir / f"{model_name}_top50_false_negatives_sheet.png",
        f"DeepGuard X [{model_name.upper()}] — Top 50 False Negatives (Fake/AI Predicted as Real)"
    )

    # False Positives sorted by highest FAKE confidence
    fp_sorted = false_positives.sort_values(by="fake_probability", ascending=False)
    fp_sorted.to_csv(fp_dir / f"{model_name}_false_positives.csv", index=False)

    top50_fp = fp_sorted.head(50).to_dict("records")
    build_contact_sheet(
        top50_fp,
        fp_dir / f"{model_name}_top50_false_positives_sheet.png",
        f"DeepGuard X [{model_name.upper()}] — Top 50 False Positives (Real Predicted as Fake)"
    )

def plot_generalization_gap(comparison_records: List[Dict[str, Any]], output_path: Path):
    df = pd.DataFrame(comparison_records)
    
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5*width, df["accuracy_gap"] * 100, width, label="Accuracy Gap (%)", color="#3498db")
    plt.bar(x - 0.5*width, df["fake_recall_gap"] * 100, width, label="Fake Recall Gap (%)", color="#e74c3c")
    plt.bar(x + 0.5*width, df["f1_gap"] * 100, width, label="F1 Gap (%)", color="#9b59b6")
    plt.bar(x + 1.5*width, df["roc_auc_gap"] * 100, width, label="ROC-AUC Gap (%)", color="#f39c12")
    
    plt.axhline(0, color="gray", linestyle="--", lw=1)
    plt.xlabel("Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Generalization Drop: Test (In-Dist) - OOD (%)", fontsize=11, fontweight="bold")
    plt.title("DeepGuard X — Generalization Gap Analysis (In-Distribution vs External OOD)", fontsize=13, fontweight="bold")
    plt.xticks(x, [m.upper() for m in models], fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generalization gap chart saved: {output_path}")

def plot_model_comparison(comparison_records: List[Dict[str, Any]], output_path: Path):
    df = pd.DataFrame(comparison_records)
    
    metrics_to_plot = ["test_accuracy", "test_f1", "test_roc_auc", "ood_accuracy", "ood_fake_recall", "ood_roc_auc"]
    labels = ["Test Acc", "Test F1", "Test ROC", "OOD Acc", "OOD Fake Rec", "OOD ROC"]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    colors = ["#2ecc71", "#3498db", "#e67e22"]
    
    for idx, row in df.iterrows():
        vals = [row[m] * 100 for m in metrics_to_plot]
        offset = (idx - 1) * width
        plt.bar(x + offset, vals, width, label=row["model"].upper(), color=colors[idx % len(colors)], alpha=0.85)
        for vx, vy in zip(x + offset, vals):
            plt.text(vx, vy + 1.0, f"{vy:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
            
    plt.xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.ylim([0, 110])
    plt.title("DeepGuard X — Complete Model Comparison Across Test & OOD Benchmarks", fontsize=13, fontweight="bold")
    plt.xticks(x, labels, fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Model comparison chart saved: {output_path}")

def main():
    print("=" * 80)
    print("DEEPGUARD X — PHASES 6 - 12: RIGOROUS EVALUATION, OOD, ERROR & COMPARISON")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    test_csv = FROZEN_DIR / "frozen_test.csv"
    ood_csv = FROZEN_DIR / "frozen_ood.csv"

    test_ds = FastImageDataset(str(test_csv), is_train=False)
    ood_ds = FastImageDataset(str(ood_csv), is_train=False)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    ood_loader = DataLoader(ood_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Loaded Frozen In-Dist Test Set: {len(test_ds):,} samples ({test_csv})")
    print(f"Loaded Frozen External OOD Set:  {len(ood_ds):,} samples ({ood_csv})")

    comparison_records = []
    all_model_results = {}

    for arch in MODELS:
        print("\n" + "=" * 80)
        print(f"EVALUATING MODEL: {arch.upper()}")
        print("=" * 80)

        model, ckpt_path, threshold = load_checkpoint_and_model(arch, device)
        model_exp_dir = EXP_DIR / arch

        # PHASE 6: Frozen Test Evaluation
        print(f"\n--- Running In-Distribution Test Evaluation ({len(test_ds):,} samples) ---")
        test_metrics, df_test_preds = evaluate_dataset(model, test_loader, device, threshold, arch, ckpt_path)
        test_pred_path = model_exp_dir / "test_predictions.csv"
        df_test_preds.to_csv(test_pred_path, index=False)
        print(f"Saved test predictions to: {test_pred_path}")

        generate_evaluation_plots(
            df_test_preds["true_label"].values,
            df_test_preds["fake_probability"].values,
            df_test_preds["predicted_label"].values,
            test_metrics,
            arch,
            prefix="test",
            out_dir=model_exp_dir
        )

        print(f"  Test Accuracy:     {test_metrics['accuracy']*100:.2f}%")
        print(f"  Test Precision:    {test_metrics['precision']*100:.2f}%")
        print(f"  Test Real Recall:  {test_metrics['real_recall']*100:.2f}% (Specificity: {test_metrics['specificity']*100:.2f}%)")
        print(f"  Test Fake Recall:  {test_metrics['fake_recall']*100:.2f}%")
        print(f"  Test FPR:          {test_metrics['fpr']*100:.2f}% | FNR: {test_metrics['fnr']*100:.2f}%")
        print(f"  Test F1:           {test_metrics['f1']:.4f}")
        print(f"  Test ROC-AUC:      {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"  Test Brier Score:  {test_metrics['brier']:.4f} | ECE: {test_metrics['ece']:.4f}")

        # PHASE 7: Frozen OOD Evaluation
        print(f"\n--- Running External OOD Evaluation ({len(ood_ds):,} samples) ---")
        ood_metrics, df_ood_preds = evaluate_dataset(model, ood_loader, device, threshold, arch, ckpt_path)
        ood_pred_path = model_exp_dir / "ood_predictions.csv"
        df_ood_preds.to_csv(ood_pred_path, index=False)
        print(f"Saved OOD predictions to: {ood_pred_path}")

        generate_evaluation_plots(
            df_ood_preds["true_label"].values,
            df_ood_preds["fake_probability"].values,
            df_ood_preds["predicted_label"].values,
            ood_metrics,
            arch,
            prefix="ood",
            out_dir=model_exp_dir
        )

        print(f"  OOD Accuracy:      {ood_metrics['accuracy']*100:.2f}%")
        print(f"  OOD Precision:     {ood_metrics['precision']*100:.2f}%")
        print(f"  OOD Real Recall:   {ood_metrics['real_recall']*100:.2f}%")
        print(f"  OOD Fake Recall:   {ood_metrics['fake_recall']*100:.2f}%")
        print(f"  OOD Specificity:   {ood_metrics['specificity']*100:.2f}%")
        print(f"  OOD FPR:           {ood_metrics['fpr']*100:.2f}% | FNR: {ood_metrics['fnr']*100:.2f}%")
        print(f"  OOD F1:            {ood_metrics['f1']:.4f}")
        print(f"  OOD ROC-AUC:       {ood_metrics['roc_auc']:.4f} | PR-AUC: {ood_metrics['pr_auc']:.4f}")
        print(f"  OOD Brier Score:   {ood_metrics['brier']:.4f} | ECE: {ood_metrics['ece']:.4f}")

        # PHASE 8: Error Analysis (Test Set & OOD)
        print(f"\n--- Running Phase 8 Error Analysis for {arch.upper()} ---")
        perform_error_analysis(df_test_preds, arch, REPORTS_DIR)

        # Investigate AI-Generated predicted as Real on OOD
        ood_ai_as_real = df_ood_preds[(df_ood_preds["true_label"] == 1) & (df_ood_preds["predicted_label"] == 0)]
        print(f"  >>> OOD AI-Generated predicted as REAL: {len(ood_ai_as_real)} / {(df_ood_preds['true_label']==1).sum()} ({len(ood_ai_as_real)/(df_ood_preds['true_label']==1).sum()*100:.2f}%)")

        # PHASE 9: Generalization Gaps
        acc_gap = test_metrics["accuracy"] - ood_metrics["accuracy"]
        fake_rec_gap = test_metrics["fake_recall"] - ood_metrics["fake_recall"]
        f1_gap = test_metrics["f1"] - ood_metrics["f1"]
        roc_auc_gap = test_metrics["roc_auc"] - ood_metrics["roc_auc"]

        print(f"\n[Generalization Gap: Test vs OOD for {arch.upper()}]")
        print(f"  Accuracy Gap:    {acc_gap*100:+.2f}%")
        print(f"  Fake Recall Gap: {fake_rec_gap*100:+.2f}%")
        print(f"  F1 Gap:          {f1_gap:+.4f}")
        print(f"  ROC-AUC Gap:     {roc_auc_gap:+.4f}")

        row = {
            "model": arch,
            "checkpoint": str(ckpt_path),
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_real_recall": test_metrics["real_recall"],
            "test_fake_recall": test_metrics["fake_recall"],
            "test_specificity": test_metrics["specificity"],
            "test_fpr": test_metrics["fpr"],
            "test_fnr": test_metrics["fnr"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "ood_accuracy": ood_metrics["accuracy"],
            "ood_precision": ood_metrics["precision"],
            "ood_real_recall": ood_metrics["real_recall"],
            "ood_fake_recall": ood_metrics["fake_recall"],
            "ood_specificity": ood_metrics["specificity"],
            "ood_fpr": ood_metrics["fpr"],
            "ood_fnr": ood_metrics["fnr"],
            "ood_f1": ood_metrics["f1"],
            "ood_roc_auc": ood_metrics["roc_auc"],
            "ood_pr_auc": ood_metrics["pr_auc"],
            "brier": test_metrics["brier"],
            "ece": test_metrics["ece"],
            "accuracy_gap": acc_gap,
            "fake_recall_gap": fake_rec_gap,
            "f1_gap": f1_gap,
            "roc_auc_gap": roc_auc_gap
        }
        comparison_records.append(row)
        all_model_results[arch] = {
            "test_metrics": test_metrics,
            "ood_metrics": ood_metrics
        }

    # PHASE 11: Model Comparison Output
    print("\n" + "=" * 80)
    print("PHASE 11: COMPILING FINAL MODEL COMPARISON")
    print("=" * 80)
    df_comp = pd.DataFrame(comparison_records)
    comp_csv_path = REPORTS_DIR / "image_model_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"Saved model comparison table to: {comp_csv_path}")

    comp_png_path = REPORTS_DIR / "image_model_comparison.png"
    plot_model_comparison(comparison_records, comp_png_path)

    gen_gap_png_path = REPORTS_DIR / "generalization_gap_chart.png"
    plot_generalization_gap(comparison_records, gen_gap_png_path)

    # Save complete JSON summary
    summary_path = REPORTS_DIR / "complete_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "models": all_model_results,
            "comparison": comparison_records
        }, f, indent=2)
    print(f"Saved complete evaluation summary to: {summary_path}")
    print("\nPhases 6 through 12 Evaluation Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()

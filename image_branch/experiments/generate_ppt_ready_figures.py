"""
Phase 12 PPT-Ready Graphics Generator for DeepGuard X Image Branch
Programmatically generates publication/presentation-ready graphics
strictly from verified prediction files (no hardcoded numbers):
1. Binary Confusion Matrix (In-Distribution Test)
2. OOD Confusion Matrix (External Held-out RVF5K)
3. Multi-Model ROC Curves
4. Multi-Model Precision-Recall Curves
5. Comprehensive Model Comparison Chart
6. Generalization-Gap Chart
7. High-Resolution Error Analysis Contact Sheet
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score
)
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path("E:/ADL_PROJECT")
EXP_DIR = PROJECT_ROOT / "image_branch" / "experiments"
REPORTS_DIR = EXP_DIR / "reports"
PPT_DIR = REPORTS_DIR / "ppt_figures"
ROOT_REPORTS_DIR = PROJECT_ROOT / "reports"
PPT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling parameters
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

MODELS = ["xception", "efficientnet", "vit"]
MODEL_COLORS = {
    "xception": "#2563eb",     # Royal Blue
    "efficientnet": "#059669", # Emerald Green
    "vit": "#7c3aed"           # Purple
}
MODEL_DISPLAY = {
    "xception": "Xception (CNN)",
    "efficientnet": "EfficientNet-B0 (CNN)",
    "vit": "ViT-B/16 (Transformer)"
}

def load_predictions(model_name: str, split: str = "test") -> pd.DataFrame:
    p = EXP_DIR / model_name / f"{split}_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(f"Predictions missing: {p}")
    return pd.read_csv(p)

def generate_figure_1_2_confusion_matrices():
    # Model: Xception (highest overall performer)
    df_test = load_predictions("xception", "test")
    df_ood = load_predictions("xception", "ood")

    # 1. In-Distribution Test Confusion Matrix
    cm_test = confusion_matrix(df_test["true_label"], df_test["predicted_label"], labels=[0, 1])
    cm_test_pct = cm_test.astype(float) / cm_test.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    labels = ["Authentic / Real", "Deepfake / Fake"]
    annots = np.array([[f"{c:,}\n({p:.1%})" for c, p in zip(r_c, r_p)] for r_c, r_p in zip(cm_test, cm_test_pct)])
    sns.heatmap(cm_test, annot=annots, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, annot_kws={"size": 13, "weight": "bold"})
    acc = accuracy_score(df_test["true_label"], df_test["predicted_label"])
    plt.title(f"DeepGuard X [Xception] — In-Distribution Test Confusion Matrix\nTotal: {len(df_test):,} Samples | Accuracy: {acc*100:.2f}%", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p1 = PPT_DIR / "1_binary_confusion_matrix_test.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {p1}")

    # 2. OOD Confusion Matrix
    cm_ood = confusion_matrix(df_ood["true_label"], df_ood["predicted_label"], labels=[0, 1])
    cm_ood_pct = cm_ood.astype(float) / cm_ood.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    labels_ood = ["Authentic / Real", "AI-Generated / GAN"]
    annots_ood = np.array([[f"{c:,}\n({p:.1%})" for c, p in zip(r_c, r_p)] for r_c, r_p in zip(cm_ood, cm_ood_pct)])
    sns.heatmap(cm_ood, annot=annots_ood, fmt="", cmap="Purples", xticklabels=labels_ood, yticklabels=labels_ood, cbar=False, annot_kws={"size": 13, "weight": "bold"})
    acc_ood = accuracy_score(df_ood["true_label"], df_ood["predicted_label"])
    plt.title(f"DeepGuard X [Xception] — External OOD Benchmark (RVF5K)\nTotal: {len(df_ood):,} Samples | Accuracy: {acc_ood*100:.2f}%", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p2 = PPT_DIR / "2_ood_confusion_matrix.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Generated Figure 2: {p2}")

def generate_figure_3_multi_roc():
    plt.figure(figsize=(7.5, 6), dpi=300)
    
    for m in MODELS:
        df = load_predictions(m, "test")
        fpr, tpr, _ = roc_curve(df["true_label"], df["fake_probability"])
        auc_score = roc_auc_score(df["true_label"], df["fake_probability"])
        plt.plot(fpr, tpr, color=MODEL_COLORS[m], lw=2.2, label=f"{MODEL_DISPLAY[m]} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Fake Recall)", fontsize=11, fontweight="bold")
    plt.title("DeepGuard X — ROC Curves Comparison (Frozen In-Distribution Test)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p3 = PPT_DIR / "3_roc_curves_all_models.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Generated Figure 3: {p3}")

def generate_figure_4_multi_pr():
    plt.figure(figsize=(7.5, 6), dpi=300)
    
    for m in MODELS:
        df = load_predictions(m, "test")
        prec, rec, _ = precision_recall_curve(df["true_label"], df["fake_probability"])
        ap_score = average_precision_score(df["true_label"], df["fake_probability"])
        plt.plot(rec, prec, color=MODEL_COLORS[m], lw=2.2, label=f"{MODEL_DISPLAY[m]} (PR-AUC = {ap_score:.4f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("DeepGuard X — Precision-Recall Curves (Frozen In-Distribution Test)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    p4 = PPT_DIR / "4_precision_recall_curves_all_models.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Generated Figure 4: {p4}")

def generate_figure_5_and_6_comparison_and_gap():
    csv_path = REPORTS_DIR / "image_model_comparison.csv"
    df = pd.read_csv(csv_path)

    # Export clean specified CSV table to root reports and ppt dir
    cols_subset = [
        "model", "checkpoint", "test_accuracy", "test_precision", "test_real_recall",
        "test_fake_recall", "test_specificity", "test_fpr", "test_fnr", "test_f1",
        "test_roc_auc", "test_pr_auc", "ood_accuracy", "ood_fake_recall", "ood_f1",
        "ood_roc_auc", "brier", "ece"
    ]
    df_clean = df[cols_subset]
    df_clean.to_csv(ROOT_REPORTS_DIR / "image_model_comparison.csv", index=False)
    df_clean.to_csv(PPT_DIR / "image_model_comparison.csv", index=False)

    # 5. Model Comparison Chart
    plt.figure(figsize=(11, 5.5), dpi=300)
    metrics_keys = ["test_accuracy", "test_fake_recall", "test_f1", "test_roc_auc", "ood_accuracy", "ood_fake_recall", "ood_roc_auc"]
    metric_labels = ["Test Acc", "Test Fake Rec", "Test F1", "Test ROC", "OOD Acc", "OOD Fake Rec", "OOD ROC"]
    x = np.arange(len(metrics_keys))
    width = 0.25

    for idx, (_, row) in enumerate(df.iterrows()):
        m_name = row["model"]
        vals = [row[k] * 100 for k in metrics_keys]
        pos = x + (idx - 1) * width
        bars = plt.bar(pos, vals, width, label=MODEL_DISPLAY[m_name], color=MODEL_COLORS[m_name], alpha=0.88)
        for b, val in zip(bars, vals):
            plt.text(b.get_x() + b.get_width()/2, val + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.xlabel("Evaluation Metric", fontsize=11, fontweight="bold")
    plt.ylabel("Performance Score (%)", fontsize=11, fontweight="bold")
    plt.ylim([0, 115])
    plt.title("DeepGuard X — Multi-Architecture Controlled Performance Benchmark", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, metric_labels, fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9.5)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p5 = PPT_DIR / "5_model_comparison_chart.png"
    plt.savefig(p5, dpi=300)
    plt.savefig(ROOT_REPORTS_DIR / "image_model_comparison.png", dpi=300)
    plt.close()
    print(f"Generated Figure 5: {p5}")

    # 6. Generalization Gap Chart
    plt.figure(figsize=(9.5, 5.2), dpi=300)
    gap_metrics = ["accuracy_gap", "fake_recall_gap", "f1_gap", "roc_auc_gap"]
    gap_labels = ["Accuracy Shift", "Fake Recall Shift", "F1 Score Shift", "ROC-AUC Shift"]
    gx = np.arange(len(gap_metrics))
    gwidth = 0.25

    for idx, (_, row) in enumerate(df.iterrows()):
        m_name = row["model"]
        # Multiply by 100 to express in percentage points
        gvals = [row[k] * 100 for k in gap_metrics]
        pos = gx + (idx - 1) * gwidth
        bars = plt.bar(pos, gvals, gwidth, label=MODEL_DISPLAY[m_name], color=MODEL_COLORS[m_name], alpha=0.88)
        for b, val in zip(bars, gvals):
            va = "bottom" if val >= 0 else "top"
            offset = 1.0 if val >= 0 else -2.5
            plt.text(b.get_x() + b.get_width()/2, val + offset, f"{val:+.1f}%", ha="center", va=va, fontsize=8, fontweight="bold")

    plt.axhline(0, color="black", linestyle="-", lw=1.2)
    plt.xlabel("Evaluation Metric Shift", fontsize=11, fontweight="bold")
    plt.ylabel("Generalization Shift (Test In-Dist - External OOD) [%]", fontsize=11, fontweight="bold")
    plt.title("DeepGuard X — Generalization Shift Analysis (In-Distribution vs External OOD RVF5K)\nNegative Shift indicates Strong Generalization to Unseen GAN Faces", fontsize=11, fontweight="bold", pad=12)
    plt.xticks(gx, gap_labels, fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9.5)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    p6 = PPT_DIR / "6_generalization_gap_chart.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"Generated Figure 6: {p6}")

def generate_figure_7_contact_sheet():
    src_sheet = REPORTS_DIR / "image_false_negatives" / "xception_top50_false_negatives_sheet.png"
    dst_sheet = PPT_DIR / "7_error_analysis_contact_sheet.png"
    if src_sheet.exists():
        im = Image.open(src_sheet)
        im.save(dst_sheet)
        print(f"Generated Figure 7: {dst_sheet}")

def main():
    print("=" * 80)
    print("DEEPGUARD X — PHASE 12: GENERATING PPT-READY VISUAL ARTIFACTS")
    print("=" * 80)

    generate_figure_1_2_confusion_matrices()
    generate_figure_3_multi_roc()
    generate_figure_4_multi_pr()
    generate_figure_5_and_6_comparison_and_gap()
    generate_figure_7_contact_sheet()

    print("\nPhase 12 PPT-Ready Visual Generation Complete!")
    print(f"All figures available at: {PPT_DIR}")

if __name__ == "__main__":
    main()

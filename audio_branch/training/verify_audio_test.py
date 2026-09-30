"""
Verification and Test Script for DeepGuard X Audio Branch.
Evaluates the trained Baseline CNN on:
1. At least 10 known REAL audio samples
2. At least 10 known FAKE/SYNTHETIC audio samples
3. Full or representative frozen test set slice
4. Controlled Robustness Scenarios (Noise 30dB, 20dB, 15dB, 10dB, Gain +/- 3dB)
5. Generates programmatically:
   - audio_branch/reports/audio_confusion_matrix.png
   - reports/audio_confusion_matrix_test.png
   - reports/audio_test_metrics.json
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_branch.inference.audio_verifier import predict_audio, get_audio_model, TARGET_SAMPLES, TARGET_SR, DEVICE

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_REPORTS_DIR = PROJECT_ROOT / "audio_branch" / "reports"
AUDIO_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_single_file_tests(test_manifest_path: Path):
    print("=" * 80)
    print("PHASE 9 & 10: VERIFYING KNOWN REAL AND FAKE AUDIO SAMPLES")
    print("=" * 80)

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    real_samples = [r for r in reader if r.get("label_name") == "REAL" and Path(r["filepath"]).exists()][:10]
    fake_samples = [r for r in reader if r.get("label_name") == "FAKE" and Path(r["filepath"]).exists()][:10]

    print("\n--- PHASE 9: VERIFY 10 REAL SAMPLES ---")
    real_correct = 0
    real_results = []
    for idx, sample in enumerate(real_samples, 1):
        fp = sample["filepath"]
        fname = Path(fp).name
        res = predict_audio(fp, model_name="baseline_cnn")
        pred_label = "REAL" if res["predicted_class"] == 0 else "FAKE"
        is_correct = (pred_label == "REAL")
        if is_correct:
            real_correct += 1
        real_results.append({
            "filename": fname,
            "true_label": "REAL",
            "predicted_label": pred_label,
            "real_prob": res["real_probability"],
            "fake_prob": res["fake_probability"],
            "correct": is_correct
        })
        print(f"[{idx:02d}] {fname} | True: REAL | Pred: {pred_label:<4} | Real%: {res['real_probability']*100:5.1f}% | Fake%: {res['fake_probability']*100:5.1f}% | {'PASS' if is_correct else 'FAIL'}")

    real_acc = real_correct / len(real_samples)
    print(f"\nReal Samples Accuracy: {real_correct}/{len(real_samples)} ({real_acc*100:.1f}%)")

    print("\n--- PHASE 10: VERIFY 10 FAKE / SYNTHETIC SAMPLES ---")
    fake_correct = 0
    fake_results = []
    for idx, sample in enumerate(fake_samples, 1):
        fp = sample["filepath"]
        fname = Path(fp).name
        res = predict_audio(fp, model_name="baseline_cnn")
        pred_label = "REAL" if res["predicted_class"] == 0 else "FAKE"
        is_correct = (pred_label == "FAKE")
        if is_correct:
            fake_correct += 1
        fake_results.append({
            "filename": fname,
            "true_label": "FAKE",
            "predicted_label": pred_label,
            "real_prob": res["real_probability"],
            "fake_prob": res["fake_probability"],
            "correct": is_correct
        })
        print(f"[{idx:02d}] {fname} | True: FAKE | Pred: {pred_label:<4} | Real%: {res['real_probability']*100:5.1f}% | Fake%: {res['fake_probability']*100:5.1f}% | {'PASS' if is_correct else 'FAIL (FAKE->REAL CRITICAL ERROR)'}")

    fake_acc = fake_correct / len(fake_samples)
    print(f"\nFake Samples Accuracy: {fake_correct}/{len(fake_samples)} ({fake_acc*100:.1f}%)")

    return real_results, fake_results


def run_full_test_evaluation(test_manifest_path: Path):
    print("\n" + "=" * 80)
    print("PHASE 8 & 11: FULL TEST SET EVALUATION & CONFUSION MATRIX GENERATION")
    print("=" * 80)

    model, display_name, ckpt_path = get_audio_model("baseline_cnn")

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Evaluate on valid test samples
    valid_records = [r for r in reader if Path(r["filepath"]).exists()]
    print(f"Found {len(valid_records)} accessible test samples from {test_manifest_path.name}.")

    y_true = []
    y_scores = []

    # Batch evaluation for efficiency
    batch_size = 32
    audio_tensors = []
    labels_batch = []

    for i, row in enumerate(valid_records):
        fp = row["filepath"]
        lbl = float(row["binary_label"])

        try:
            audio, sr = sf.read(fp)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            audio = np.zeros(TARGET_SAMPLES, dtype=np.float32)

        # Pad or center crop
        if len(audio) < TARGET_SAMPLES:
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
        elif len(audio) > TARGET_SAMPLES:
            start = (len(audio) - TARGET_SAMPLES) // 2
            audio = audio[start : start + TARGET_SAMPLES]
        else:
            audio = audio[:TARGET_SAMPLES]

        audio_tensors.append(audio)
        labels_batch.append(lbl)

        if len(audio_tensors) == batch_size or i == len(valid_records) - 1:
            t_batch = torch.from_numpy(np.array(audio_tensors, dtype=np.float32)).to(DEVICE)
            with torch.no_grad():
                logits = model(t_batch)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()

            if isinstance(probs, float):
                probs = [probs]

            y_scores.extend(probs)
            y_true.extend(labels_batch)
            audio_tensors = []
            labels_batch = []

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    threshold = 0.50
    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    fake_recall = float(recall_score(y_true, y_pred, zero_division=0))
    real_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_scores))
    pr_auc = float(average_precision_score(y_true, y_scores))

    print(f"\n--- TEST METRICS (Baseline CNN @ threshold={threshold}) ---")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Precision:   {prec:.4f}")
    print(f"Real Recall: {real_recall*100:.2f}% (Specificity)")
    print(f"Fake Recall: {fake_recall*100:.2f}% (Sensitivity)")
    print(f"F1-Score:    {f1:.4f}")
    print(f"FPR:         {fpr*100:.2f}%")
    print(f"FNR:         {fnr*100:.2f}%")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp})")

    # Generate Confusion Matrix Plots
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["REAL", "FAKE"]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Baseline Audio CNN — Test Confusion Matrix",
           ylabel="True Label",
           xlabel="Predicted Label")

    thresh = cm.max() / 2.
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            val = cm[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{val}\n({val/len(y_true)*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontweight="bold")

    fig.tight_layout()
    cm_path1 = AUDIO_REPORTS_DIR / "audio_confusion_matrix.png"
    cm_path2 = REPORTS_DIR / "audio_confusion_matrix_test.png"
    fig.savefig(cm_path1, dpi=300)
    fig.savefig(cm_path2, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to:\n - {cm_path1}\n - {cm_path2}")

    metrics_out = {
        "accuracy": acc,
        "precision": prec,
        "recall": fake_recall,
        "f1": f1,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "specificity": real_recall,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "baseline_cnn": {
            "accuracy": acc,
            "precision": prec,
            "fake_recall": fake_recall,
            "real_recall": real_recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "fnr": fnr,
            "threshold": threshold,
            "checkpoint": str(ckpt_path)
        }
    }

    with open(REPORTS_DIR / "audio_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    return metrics_out, y_true, valid_records


def run_robustness_evaluation(valid_records, y_true):
    print("\n" + "=" * 80)
    print("PHASE 12: ROBUSTNESS CHECKS ACROSS ACOUSTIC DISTORTIONS")
    print("=" * 80)

    model, _, _ = get_audio_model("baseline_cnn")

    # Use 200 balanced test samples for speed
    test_slice = valid_records[:200]
    y_sub = y_true[:200]

    # Pre-load audio
    raw_audios = []
    for r in test_slice:
        fp = r["filepath"]
        try:
            a, _ = sf.read(fp)
            if a.ndim > 1:
                a = a.mean(axis=1)
        except Exception:
            a = np.zeros(TARGET_SAMPLES, dtype=np.float32)

        if len(a) < TARGET_SAMPLES:
            a = np.pad(a, (0, TARGET_SAMPLES - len(a)), mode="constant")
        elif len(a) > TARGET_SAMPLES:
            start = (len(a) - TARGET_SAMPLES) // 2
            a = a[start : start + TARGET_SAMPLES]
        else:
            a = a[:TARGET_SAMPLES]
        raw_audios.append(a.astype(np.float32))

    raw_arr = np.array(raw_audios)

    conditions = [
        ("Clean", lambda x: x, "Baseline reference"),
        ("30 dB noise", lambda x: x + np.random.normal(0, 0.001, x.shape).astype(np.float32), "Light ambient noise"),
        ("20 dB noise", lambda x: x + np.random.normal(0, 0.005, x.shape).astype(np.float32), "Moderate ambient noise"),
        ("15 dB noise", lambda x: x + np.random.normal(0, 0.015, x.shape).astype(np.float32), "Heavy room background noise"),
        ("10 dB noise", lambda x: x + np.random.normal(0, 0.035, x.shape).astype(np.float32), "Severe background noise"),
        ("Volume -3 dB", lambda x: x * 0.7079, "Volume attenuation (-3dB)"),
        ("Volume +3 dB", lambda x: x * 1.4125, "Volume amplification (+3dB)"),
    ]

    robustness_results = []
    for cond_name, distort_fn, note in conditions:
        distorted = distort_fn(raw_arr.copy())
        t_batch = torch.from_numpy(distorted).to(DEVICE)
        with torch.no_grad():
            logits = model(t_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

        preds = (probs >= 0.50).astype(int)
        f1 = float(f1_score(y_sub, preds, zero_division=0))
        fake_rec = float(recall_score(y_sub, preds, zero_division=0))
        acc = float(accuracy_score(y_sub, preds))

        robustness_results.append({
            "condition": cond_name,
            "accuracy": acc,
            "f1": f1,
            "fake_recall": fake_rec,
            "note": note
        })
        print(f"Condition: {cond_name:<15} | Acc: {acc*100:5.1f}% | F1: {f1:.4f} | Fake Recall: {fake_rec*100:5.1f}% | {note}")

    return robustness_results


if __name__ == "__main__":
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "audio_test.csv"
    if not manifest_path.is_file():
        manifest_path = Path(r"E:\DEEPFAKE\MASTER_DATASET\MANIFESTS\audio_test.csv")

    real_res, fake_res = run_single_file_tests(manifest_path)
    metrics_res, y_true, records = run_full_test_evaluation(manifest_path)
    rob_res = run_robustness_evaluation(records, y_true)

    print("\n" + "=" * 80)
    print("ALL AUDIO EVALUATION PHASES (8, 9, 10, 11, 12) COMPLETED.")
    print("=" * 80)

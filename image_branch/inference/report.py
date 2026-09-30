"""
Combined Explainability Inference Report Module for FaceForensics++ C23 Image Branch.

Integrates:
1. EfficientNet-B0 6-Class & Binary Classifier
2. Grad-CAM Neural Feature Attention Map
3. Error Level Analysis (ELA) Compression Artifact Map

Outputs: Combined 3-panel visualization report (Original Image | Grad-CAM Overlay | ELA Map).
"""

import os
import sys
import json
import functools
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ensure stdout prints flush immediately
print = functools.partial(print, flush=True)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.preprocessing.preprocess import load_image
from image_branch.inference.gradcam import generate_gradcam, LABEL_MAPPING
from image_branch.inference.ela import generate_ela


# Active model checkpoint configuration
# Switched from efficientnet_b0_best.pth to efficientnet_b0_rebalanced.pth on 2026-09-25
# to address real-image false-positive bias (see reports/metrics/step_14_rebalancing_report.md
# for the full comparison). Original checkpoint retained for rollback if needed.
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_rebalanced.pth"

CONFIDENCE_THRESHOLD = 0.60


def generate_report(
    image_input: Union[str, Path, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    save_fig: bool = True,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generates a unified explainability report for a single input image or frame array, combining
    model prediction, Grad-CAM attention heatmap, and ELA compression artifact map.

    Args:
        image_input (Union[str, Path, np.ndarray]): Filepath to image or RGB uint8 numpy array.
        output_path (Optional[Union[str, Path]]): Filepath to save combined PNG report.
        save_fig (bool): Whether to render and save matplotlib 3-panel figure.
        checkpoint_path (Optional[Union[str, Path]]): Path to model checkpoint. If None, uses ACTIVE_MODEL_PATH.

    Returns:
        Dict[str, Any]: Unified result dictionary containing prediction details,
                        confidence, binary verdict, image arrays, and metadata.
    """
    if checkpoint_path is None:
        checkpoint_path = ACTIVE_MODEL_PATH
    if isinstance(image_input, (str, Path)):
        img_path_obj = Path(image_input)
        if not img_path_obj.is_file():
            raise FileNotFoundError(f"Input image file not found at: {img_path_obj}")
        image_name_str = img_path_obj.name
        image_path_str = str(img_path_obj)
        full_orig_rgb = load_image(str(img_path_obj))
    elif isinstance(image_input, np.ndarray):
        image_name_str = "Webcam Live Frame"
        image_path_str = "in-memory frame array"
        full_orig_rgb = image_input.copy()
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # 1. Run Grad-CAM module (includes model forward pass & heatmap generation)
    gradcam_res = generate_gradcam(image_input, checkpoint_path=checkpoint_path)

    # 2. Run ELA module
    ela_res = generate_ela(image_input, quality=90, scale=15.0)

    confidence = float(gradcam_res["confidence"])
    raw_verdict = gradcam_res["binary_verdict"]

    # 3. Task 2: Apply Confidence Threshold (UNCERTAIN for confidence < 60%)
    if confidence < CONFIDENCE_THRESHOLD:
        binary_verdict = "UNCERTAIN"
        confidence_note = "Low model confidence — human review recommended"
    else:
        binary_verdict = raw_verdict
        confidence_note = ""

    result = {
        "prediction": gradcam_res["pred_class_name"],
        "pred_class_id": gradcam_res["pred_class_id"],
        "confidence": confidence,
        "binary_verdict": binary_verdict,
        "raw_binary_verdict": raw_verdict,
        "confidence_note": confidence_note,
        "probs": gradcam_res["probs"],
        "image_path": image_path_str,
        "gradcam_image": gradcam_res["overlay_image"],
        "heatmap_raw": gradcam_res["heatmap_raw"],
        "ela_image": ela_res["ela_image"],
        "ela_mean_diff": ela_res["mean_diff"],
        "original_image": full_orig_rgb
    }

    # 4. Render and Save Combined 3-Panel Visualization Figure
    if save_fig:
        h, w = full_orig_rgb.shape[:2]

        # Resize Grad-CAM overlay and ELA map to match full original image dimensions for consistent sizing
        gradcam_panel = cv2.resize(result["gradcam_image"], (w, h), interpolation=cv2.INTER_LINEAR)
        ela_panel = cv2.resize(result["ela_image"], (w, h), interpolation=cv2.INTER_LINEAR)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)

        # Title / Header Styling with larger fonts & UNCERTAIN amber badge
        if result["binary_verdict"] == "MANIPULATED":
            verdict_color = "#d9534f"  # Red
        elif result["binary_verdict"] == "AUTHENTIC":
            verdict_color = "#5cb85c"  # Green
        else:
            verdict_color = "#f0ad4e"  # Amber for UNCERTAIN

        sub_note = f"\nNote: {result['confidence_note']}" if result["confidence_note"] else ""
        header_title = (
            f"DEEPFAKE EXPLAINABILITY REPORT — {image_name_str}\n"
            f"Verdict: {result['binary_verdict']}  |  "
            f"Predicted Class: {result['prediction']} ({result['confidence']*100:.1f}% Confidence){sub_note}"
        )
        fig.suptitle(header_title, fontsize=14, fontweight="bold", y=0.98, color=verdict_color)

        # Panel 1: Original Input Image
        axes[0].imshow(full_orig_rgb)
        axes[0].set_title("1. Input Frame (RGB)", fontsize=13, fontweight="bold", pad=10)

        # Panel 2: Grad-CAM Neural Attention Overlay
        axes[1].imshow(gradcam_panel)
        axes[1].set_title(
            f"2. Grad-CAM Model Attention\nTarget Class: {result['prediction']}",
            fontsize=13, fontweight="bold", pad=10
        )

        # Panel 3: ELA Compression Artifact Map
        axes[2].imshow(ela_panel)
        axes[2].set_title(
            f"3. ELA Artifact Map (Q=90)\nMean Pixel Err: {result['ela_mean_diff']:.2f}",
            fontsize=13, fontweight="bold", pad=10
        )

        # Add light border frame around each of the 3 panels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#cccccc")
                spine.set_linewidth(2.0)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"[Report Saved] Combined 3-panel report saved to: {output_path}")

        plt.close()

    return result


def run_3_sample_test():
    """
    Executes Task 3 verification on 3 sample images pulled from image_test.csv:
    1. Original (Real class)
    2. Deepfakes (Well-detected fake class, F1 = 0.6851)
    3. NeuralTextures (Poorly-detected fake class, F1 = 0.5044)
    """
    print("=" * 80)
    print("  ADL PROJECT — STEP 6: GRAD-CAM + ELA 3-SAMPLE EXPLAINABILITY TEST")
    print("=" * 80)

    test_csv = PROJECT_ROOT / "data" / "splits" / "image_test.csv"
    sample_dir = PROJECT_ROOT / "reports" / "dataset_audit" / "gradcam_ela_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV manifest missing at: {test_csv}")

    df = pd.read_csv(test_csv)

    # Select 1 sample from each targeted category
    orig_samples = df[df["label"] == "Original"]
    deepfakes_samples = df[df["label"] == "Deepfakes"]
    neural_samples = df[df["label"] == "NeuralTextures"]

    if orig_samples.empty or deepfakes_samples.empty or neural_samples.empty:
        raise ValueError("Could not find all required sample classes in image_test.csv")

    sample_targets = [
        ("sample_1_original", orig_samples.iloc[0]),
        ("sample_2_deepfakes", deepfakes_samples.iloc[0]),
        ("sample_3_neuraltextures", neural_samples.iloc[0]),
    ]

    summary_records = []

    for sample_name, row in sample_targets:
        img_path = Path(row["filepath"])
        ground_truth_class = row["label"]
        output_png = sample_dir / f"{sample_name}.png"

        print(f"\n[Processing {sample_name}] Ground Truth: '{ground_truth_class}' | File: {img_path.name}")
        report_res = generate_report(img_path, output_path=output_png, save_fig=True)

        correct = (report_res["prediction"] == ground_truth_class)
        print(f"  Prediction:       {report_res['prediction']} (Confidence: {report_res['confidence']*100:.1f}%)")
        print(f"  Binary Verdict:   {report_res['binary_verdict']}")
        print(f"  Ground Truth:     {ground_truth_class} (Correct: {correct})")
        print(f"  ELA Mean Diff:    {report_res['ela_mean_diff']:.2f}")

        summary_records.append({
            "sample_name": sample_name,
            "filename": img_path.name,
            "filepath": str(img_path),
            "ground_truth": ground_truth_class,
            "prediction": report_res["prediction"],
            "confidence": report_res["confidence"],
            "binary_verdict": report_res["binary_verdict"],
            "is_correct": correct,
            "ela_mean_diff": report_res["ela_mean_diff"],
            "report_image_path": str(output_png)
        })

    print("\n" + "=" * 80)
    print("  3-SAMPLE EXPLAINABILITY TEST COMPLETE")
    print("=" * 80)
    print(f"  Sample reports saved to: {sample_dir}")
    return summary_records


if __name__ == "__main__":
    run_3_sample_test()

"""
Video Deepfake Detection Stub Module (Fallback Provider).

# =========================================================================================
# ARCHITECTURE NOTE (VIDEO BRANCH STATUS):
# Primary: Trained CNN-GRU temporal model (InceptionV3 features + GRU classifier),
# integrated via isolated environment + subprocess bridge (video_branch/inference/video_cnn_gru.py).
# Fallback: Multi-frame aggregation stub re-using the trained EfficientNet-B0 image model across
# 5-8 sampled frames when the CNN-GRU subprocess is unavailable.
# =========================================================================================
"""

import os
import sys
import functools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

# Ensure stdout prints flush immediately
print = functools.partial(print, flush=True)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.inference.gradcam import generate_gradcam, load_trained_model

_LABEL_MAPPING = {
    0: "Original",
    1: "FaceSwap",
    2: "Deepfakes",
    3: "Face2Face",
    4: "NeuralTextures",
    5: "FaceShifter"
}


# Active fallback model checkpoint (historical DeepGuard reference checkpoint)
FALLBACK_CHECKPOINT = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"


def analyze_video(
    video_path: str,
    num_frames: int = 6,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes a video file by sampling N evenly-spaced frames and running each through
    the trained EfficientNet-B0 image model.

    Args:
        video_path (str): Filepath to target video (.mp4, .avi, etc.).
        num_frames (int): Number of frames to sample (default: 6).
        checkpoint_path (Optional[str]): Optional custom model checkpoint path.

    Returns:
        Dict[str, Any]: Dictionary containing video-level verdict, average confidence,
                        dominant class prediction, and detailed per-frame results.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    if checkpoint_path is None:
        if FALLBACK_CHECKPOINT.is_file():
            checkpoint_path = str(FALLBACK_CHECKPOINT)

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        total_frames = 30  # Fallback estimate

    # Calculate frame sampling indices
    step = max(1, total_frames // num_frames)
    frame_indices = [min(i * step, total_frames - 1) for i in range(num_frames)]

    model, device = load_trained_model(checkpoint_path)

    frame_results = []
    frame_scores = []
    fake_count = 0
    real_count = 0
    class_counts = {}
    confidences = []

    for idx, f_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()

        if not ret or frame_bgr is None:
            continue

        # Convert BGR frame to RGB numpy array
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run trained image classifier + Grad-CAM heatmap generator
        res = generate_gradcam(frame_rgb, model=model, device=device)

        pred_class = res["pred_class_name"]
        conf = float(res["confidence"])
        verdict = res["binary_verdict"]

        p_orig = float(res["probs"].get("Original", 1.0 - conf if verdict == "MANIPULATED" else conf))
        p_fake = float(1.0 - p_orig)

        if verdict == "MANIPULATED":
            fake_count += 1
        else:
            real_count += 1

        class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        confidences.append(conf)
        frame_scores.append(round(p_fake, 4))

        frame_results.append({
            "sample_index": idx + 1,
            "frame_number": f_idx,
            "pred_class": pred_class,
            "confidence": conf,
            "fake_probability": round(p_fake, 4),
            "real_probability": round(p_orig, 4),
            "binary_verdict": verdict
        })

    cap.release()

    if not frame_results:
        raise ValueError(f"Could not extract any valid frames from video: {video_path}")

    # Aggregate frame results into video-level verdict
    video_verdict = "MANIPULATED" if fake_count >= real_count else "AUTHENTIC"
    avg_confidence = float(np.mean(confidences))
    fake_prob = float(np.mean([r["fake_probability"] for r in frame_results]))
    real_prob = float(1.0 - fake_prob)
    eps = 1e-7
    clipped_p = np.clip(fake_prob, eps, 1.0 - eps)
    raw_logit = float(np.log(clipped_p / (1.0 - clipped_p)))
    dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]

    return {
        "video_verdict": video_verdict,
        "verdict": video_verdict,
        "binary_verdict": video_verdict,
        "prediction": f"LIKELY {video_verdict}",
        "confidence": round(avg_confidence, 4),
        "avg_confidence": round(avg_confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "raw_logit": round(raw_logit, 4),
        "threshold": 0.50,
        "pred_class": dominant_class,
        "frames_analyzed": len(frame_results),
        "num_frames_analyzed": len(frame_results),
        "total_video_frames": total_frames,
        "sequence_shape": [1, len(frame_results), 2048],
        "fake_frame_count": fake_count,
        "real_frame_count": real_count,
        "class_counts": class_counts,
        "frame_results": frame_results,
        "frame_scores": frame_scores,
        "model": "Prototype Frame-Aggregation Fallback (EfficientNet-B0)",
        "maturity": "TRAINED_FALLBACK",
        "status": "fallback_active",
        "checkpoint": str(checkpoint_path),
        "fallback": True,
        "source": "Frame-Aggregation Fallback (EfficientNet-B0)",
        "video_path": str(video_path_obj),
        "stub_note": "Prototype Frame-Aggregation Fallback (EfficientNet-B0) across sampled video frames."
    }


if __name__ == "__main__":
    sample_video = PROJECT_ROOT / "raw_archives" / "FAKEAVCELEB" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "RealVideo-RealAudio" / "African" / "men" / "id00076" / "00001.mp4"
    if not sample_video.is_file():
        # Fallback to any mp4 in raw_archives
        vids = list((PROJECT_ROOT / "raw_archives").glob("**/*.mp4"))
        if vids:
            sample_video = vids[0]

    if sample_video.is_file():
        print(f"[Video Stub Test] Running analyze_video on: {sample_video.name}")
        out = analyze_video(str(sample_video), num_frames=6)
        print(f"  Video Verdict:    {out['video_verdict']} (Avg Conf: {out['avg_confidence']*100:.1f}%)")
        print(f"  Dominant Class:   {out['pred_class']}")
        print(f"  Frames Analyzed:  {out['num_frames_analyzed']} (Fake: {out['fake_frame_count']}, Real: {out['real_frame_count']})")

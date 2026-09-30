"""
Video Deepfake Detection Module using External Trained CNN-GRU Model.

Calls run_inference.py in the isolated venv as a Python subprocess.
Returns standard format compatible with DeepGuard X:
{
    "video_verdict": "AUTHENTIC" / "MANIPULATED",
    "verdict": "AUTHENTIC" / "MANIPULATED",
    "avg_confidence": float,
    "confidence": float,
    "fake_probability": float,
    "real_probability": float,
    "raw_logit": float,
    "frames_analyzed": int,
    "sequence_shape": list,
    "frame_scores": list,
    "model": "CNN-GRU Temporal Model",
    "checkpoint": str,
    "maturity": "TRAINED",
    "status": "trained_model",
    "fallback": False
}
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_MODEL_DIR = PROJECT_ROOT / "video_branch" / "external_model"
VENV_PYTHON = EXTERNAL_MODEL_DIR / "venv_tf" / "Scripts" / "python.exe"
INFERENCE_SCRIPT = EXTERNAL_MODEL_DIR / "run_inference.py"
WEIGHTS_FILE = EXTERNAL_MODEL_DIR / "deepfake_video_model.h5"


class VideoModelUnavailableError(Exception):
    """Raised when the trained CNN-GRU model checkpoint or isolated venv is unavailable."""
    pass


def analyze_video_cnn_gru(video_path: str) -> Dict[str, Any]:
    """
    Analyzes a video using the isolated CNN-GRU model via subprocess.

    Args:
        video_path (str): Filepath to target video.

    Returns:
        Dict[str, Any]: Standardized video analysis dictionary.

    Raises:
        FileNotFoundError: If video file does not exist.
        VideoModelUnavailableError: If checkpoint, venv, or inference script cannot be loaded.
    """
    video_path_obj = Path(video_path).resolve()
    if not video_path_obj.is_file():
        raise FileNotFoundError(f"[CNN-GRU Video] File not found: {video_path}")

    if not WEIGHTS_FILE.is_file():
        raise VideoModelUnavailableError(f"Video model checkpoint not found at: {WEIGHTS_FILE}")

    if not VENV_PYTHON.is_file():
        raise VideoModelUnavailableError(f"Isolated venv python not found at: {VENV_PYTHON}")

    if not INFERENCE_SCRIPT.is_file():
        raise VideoModelUnavailableError(f"Inference script not found at: {INFERENCE_SCRIPT}")

    cmd = [
        str(VENV_PYTHON),
        str(INFERENCE_SCRIPT),
        str(video_path_obj)
    ]

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(EXTERNAL_MODEL_DIR)
        )

        stdout_lines = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
        if not stdout_lines:
            raise VideoModelUnavailableError("[CNN-GRU Video] Subprocess returned empty output.")

        data = json.loads(stdout_lines[-1])
        if "error" in data:
            raise VideoModelUnavailableError(f"[CNN-GRU Video] Model error: {data['error']}")

        verdict_raw = data.get("verdict", "REAL").upper()
        video_verdict = data.get("video_verdict", "AUTHENTIC" if verdict_raw == "REAL" else "MANIPULATED")
        confidence = float(data.get("confidence", 0.0))
        fake_prob = float(data.get("fake_probability", confidence if video_verdict == "MANIPULATED" else 1.0 - confidence))
        real_prob = float(data.get("real_probability", 1.0 - fake_prob))
        raw_logit = float(data.get("raw_logit", 0.0))
        frames_analyzed = int(data.get("frames_analyzed", 20))
        frame_scores = data.get("frame_scores", [])
        seq_shape = data.get("sequence_shape", [1, 20, 2048])
        model_name = data.get("model", "CNN-GRU Temporal Model")
        checkpoint_path = data.get("checkpoint", str(WEIGHTS_FILE))

        result = {
            "video_verdict": video_verdict,
            "verdict": video_verdict,
            "binary_verdict": video_verdict,
            "prediction": data.get("prediction", f"LIKELY {video_verdict}"),
            "confidence": confidence,
            "avg_confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "raw_logit": raw_logit,
            "threshold": data.get("threshold", 0.50),
            "frames_analyzed": frames_analyzed,
            "sequence_shape": seq_shape,
            "frame_scores": frame_scores,
            "model": model_name,
            "maturity": "TRAINED",
            "status": "trained_model",
            "checkpoint": checkpoint_path,
            "source": "CNN-GRU trained model (external, isolated env)",
            "fallback": False
        }

        return result

    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else f"Exit code {e.returncode}"
        raise VideoModelUnavailableError(f"CNN-GRU video subprocess failed: {stderr_msg}") from e
    except json.JSONDecodeError as e:
        raise VideoModelUnavailableError(f"Failed parsing CNN-GRU output JSON: {e}") from e


if __name__ == "__main__":
    sample_video = PROJECT_ROOT / "test_samples" / "sample_video.mp4"
    if not sample_video.is_file():
        vids = list((PROJECT_ROOT / "archive" / "raw_archives").glob("**/*.mp4"))
        if vids:
            sample_video = vids[0]

    if sample_video.is_file():
        print(f"[CNN-GRU Bridge Test] Running analyze_video_cnn_gru on: {sample_video.name}")
        out = analyze_video_cnn_gru(str(sample_video))
        print("  Output:", json.dumps(out, indent=2))

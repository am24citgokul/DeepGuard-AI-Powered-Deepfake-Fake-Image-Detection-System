"""
Audio Deepfake Detection Stub Module.

# =========================================================================================
# ARCHITECTURE NOTE (AUDIO STUB):
# Audio branch is a prototype-level stub using acoustic feature heuristics (spectral flatness,
# zero-crossing rate, MFCC variance). Full audio branch (Mel-spectrogram CNN / Wav2Vec2)
# is future work.
# =========================================================================================
"""

import os
import sys
import functools
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import librosa

# Ensure stdout prints flush immediately
print = functools.partial(print, flush=True)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    Analyzes an audio file using acoustic feature heuristics (spectral flatness,
    zero-crossing rate, high-order MFCC variance) to produce a prototype deepfake verdict.

    Args:
        audio_path (str): Filepath to target audio (.wav, .mp3, etc.).

    Returns:
        Dict[str, Any]: Dictionary containing audio verdict, heuristic confidence,
                        extracted feature statistics, and prototype stub metadata.
    """
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.is_file():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # Load audio at 16 kHz sampling rate
    try:
        y, sr = librosa.load(str(audio_path_obj), sr=16000)
    except Exception as exc:
        raise ValueError(f"Failed to load audio from {audio_path}: {exc}") from exc

    if len(y) == 0:
        raise ValueError(f"Audio file {audio_path} contains 0 samples")

    # 1. Compute Spectral Flatness (Higher values indicate noise/vocoder reconstruction artifacts)
    flatness_arr = librosa.feature.spectral_flatness(y=y)
    mean_flatness = float(np.mean(flatness_arr))

    # 2. Compute Zero-Crossing Rate (ZCR)
    zcr_arr = librosa.feature.zero_crossing_rate(y=y)
    mean_zcr = float(np.mean(zcr_arr))

    # 3. Compute High-Order MFCC Variance (Neural vocoders often lack high-freq variance)
    mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    high_mfcc_var = float(np.mean(np.var(mfcc_matrix[5:], axis=1)))

    # 4. Apply Dynamic Heuristic Feature Rules
    # Synthetic / vocoded speech exhibits higher spectral flatness (>0.020), lower MFCC variance (<150.0), or ZCR anomalies
    flatness_score = max(0.0, (mean_flatness - 0.020) / 0.030)
    zcr_anomaly = max(0.0, (mean_zcr - 0.120) / 0.10)
    mfcc_norm = max(0.0, (150.0 - high_mfcc_var) / 150.0) if high_mfcc_var < 150.0 else 0.0

    anomaly_score = float(np.clip(0.50 * flatness_score + 0.30 * mfcc_norm + 0.20 * zcr_anomaly, 0.0, 1.0))
    is_synthetic = anomaly_score > 0.30

    audio_verdict = "MANIPULATED" if is_synthetic else "AUTHENTIC"

    # Dynamic continuous confidence score computed per-file based on acoustic deviation
    margin = abs(anomaly_score - 0.30)
    conf = float(np.clip(0.55 + margin * 0.75, 0.50, 0.85))

    return {
        "audio_verdict": audio_verdict,
        "confidence": conf,
        "features": {
            "spectral_flatness": mean_flatness,
            "zero_crossing_rate": mean_zcr,
            "high_mfcc_variance": high_mfcc_var
        },
        "audio_duration_sec": float(len(y) / sr),
        "sampling_rate": sr,
        "audio_path": str(audio_path_obj),
        "stub_note": "PROTOTYPE STUB, LOW CONF (Acoustic Feature Heuristics)"
    }


def test_audio_stub():
    """
    Tests analyze_audio on 2-3 sample audio files from raw_archives.
    """
    print("=" * 80)
    print("  AUDIO STUB INFERENCE TEST (Acoustic Heuristics)")
    print("=" * 80)

    raw_dir = PROJECT_ROOT / "raw_archives"
    audio_files = list(raw_dir.glob("**/*.wav"))[:3]

    if not audio_files:
        print("[Warning] No sample .wav files found in raw_archives to test.")
        return

    for idx, fpath in enumerate(audio_files, 1):
        print(f"\n[Sample {idx}] File: {fpath.name}")
        res = analyze_audio(str(fpath))
        print(f"  Audio Verdict:   {res['audio_verdict']} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"  Duration:        {res['audio_duration_sec']:.2f}s | SR: {res['sampling_rate']} Hz")
        print(f"  Flatness:        {res['features']['spectral_flatness']:.4f}")
        print(f"  ZCR:             {res['features']['zero_crossing_rate']:.4f}")
        print(f"  MFCC Variance:   {res['features']['high_mfcc_variance']:.2f}")


if __name__ == "__main__":
    test_audio_stub()

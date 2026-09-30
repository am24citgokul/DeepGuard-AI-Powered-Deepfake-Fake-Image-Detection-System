"""
Simple Multimodal Fusion Module (Rule-Based Evidence Aggregator).

Fuses image, video, and audio analysis results to produce a unified cross-modal verdict:
- TRUSTED (Unanimous Authentic)
- HIGH RISK (Unanimous Manipulated)
- REVIEW REQUIRED (Mixed Cross-Modal Flags)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def fuse_evidence(
    image_result: Optional[Dict[str, Any]] = None,
    video_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Combines evidence from image, video, and audio inspection modules.

    Args:
        image_result (Optional[Dict[str, Any]]): Result dictionary from image branch.
        video_result (Optional[Dict[str, Any]]): Result dictionary from video stub branch.
        audio_result (Optional[Dict[str, Any]]): Result dictionary from audio stub branch.

    Returns:
        Dict[str, Any]: Unified multi-modal assessment including final verdict,
                        consistency label, modality breakdown, and human-readable reasoning.
    """
    modality_breakdown = {}
    verdicts = {}
    confidences = {}

    # 1. Parse Image Evidence
    if image_result is not None:
        bin_v = image_result.get("binary_verdict", "MANIPULATED" if image_result.get("prediction") != "Original" else "AUTHENTIC")
        conf = float(image_result.get("confidence", 0.0))
        verdicts["image"] = bin_v
        confidences["image"] = conf
        modality_breakdown["image"] = {
            "verdict": bin_v,
            "confidence": conf,
            "pred_class": image_result.get("prediction", "Unknown"),
            "model_type": "FULLY TRAINED MODEL"
        }
    else:
        modality_breakdown["image"] = None

    # 2. Parse Video Evidence
    if video_result is not None:
        bin_v = video_result.get("video_verdict", "MANIPULATED")
        conf = float(video_result.get("avg_confidence", 0.0))
        verdicts["video"] = bin_v
        confidences["video"] = conf

        if video_result.get("model_type"):
            model_type = video_result["model_type"]
        elif "CNN-GRU" in str(video_result.get("source", "")):
            model_type = "TRAINED CNN-GRU MODEL"
        elif video_result.get("is_fallback"):
            model_type = "PROTOTYPE STUB FALLBACK"
        else:
            model_type = "PROTOTYPE STUB"

        modality_breakdown["video"] = {
            "verdict": bin_v,
            "confidence": conf,
            "pred_class": video_result.get("pred_class", "N/A"),
            "model_type": model_type
        }
    else:
        modality_breakdown["video"] = None

    # 3. Parse Audio Evidence
    if audio_result is not None:
        bin_v = audio_result.get("audio_verdict", "AUTHENTIC")
        conf = float(audio_result.get("confidence", 0.0))
        verdicts["audio"] = bin_v
        confidences["audio"] = conf
        modality_breakdown["audio"] = {
            "verdict": bin_v,
            "confidence": conf,
            "model_type": "PROTOTYPE STUB, LOW CONF"
        }
    else:
        modality_breakdown["audio"] = None

    active_modalities = list(verdicts.keys())
    num_active = len(active_modalities)

    if num_active == 0:
        return {
            "final_verdict": "NO DATA",
            "cross_modal_consistency": "NONE",
            "num_modalities_analyzed": 0,
            "modality_breakdown": modality_breakdown,
            "reasoning": "No modality inputs were provided for fusion."
        }

    # Separate decisive verdicts (AUTHENTIC/MANIPULATED) from UNCERTAIN states
    decisive_verdicts = {m: v for m, v in verdicts.items() if v in ("AUTHENTIC", "MANIPULATED")}
    uncertain_modalities = [m for m, v in verdicts.items() if v not in ("AUTHENTIC", "MANIPULATED")]
    num_decisive = len(decisive_verdicts)

    if num_decisive == 0:
        final_v = "REVIEW REQUIRED"
        consistency = "UNCERTAIN_ONLY"
        reasoning = f"All active modalities ({', '.join([m.upper() for m in active_modalities])}) returned UNCERTAIN due to low confidence (<60%). Human review recommended."
    elif num_decisive == 1:
        mod, v = list(decisive_verdicts.items())[0]
        final_v = "TRUSTED" if v == "AUTHENTIC" else "HIGH RISK"
        uncertain_note = f" (Excluded UNCERTAIN modalities: {', '.join([m.upper() for m in uncertain_modalities])})" if uncertain_modalities else ""
        reasoning = f"Single decisive modality ({mod.upper()}: {v}){uncertain_note}."
        consistency = "SINGLE_MODALITY"
    else:
        decisive_values = set(decisive_verdicts.values())
        uncertain_note = f" (Excluded UNCERTAIN modalities: {', '.join([m.upper() for m in uncertain_modalities])})" if uncertain_modalities else ""
        if len(decisive_values) == 1:
            unanimous_v = list(decisive_values)[0]
            if unanimous_v == "AUTHENTIC":
                final_v = "TRUSTED"
                consistency = "UNANIMOUS_AUTHENTIC"
                reasoning = f"Decisive modalities ({', '.join([m.upper() for m in decisive_verdicts.keys()])}) agreed AUTHENTIC.{uncertain_note}"
            else:
                final_v = "HIGH RISK"
                consistency = "UNANIMOUS_MANIPULATED"
                reasoning = f"Decisive modalities ({', '.join([m.upper() for m in decisive_verdicts.keys()])}) agreed MANIPULATED.{uncertain_note}"
        else:
            final_v = "REVIEW REQUIRED"
            consistency = "MIXED"
            flagged_fake = [m.upper() for m, val in decisive_verdicts.items() if val == "MANIPULATED"]
            flagged_real = [m.upper() for m, val in decisive_verdicts.items() if val == "AUTHENTIC"]
            reasoning = f"Cross-modal disagreement detected. Flagged MANIPULATED by [{', '.join(flagged_fake)}], AUTHENTIC by [{', '.join(flagged_real)}].{uncertain_note}"

    avg_overall_conf = float(np.mean(list(confidences.values()))) if confidences else 0.0

    return {
        "final_verdict": final_v,
        "cross_modal_consistency": consistency,
        "overall_confidence": avg_overall_conf,
        "num_modalities_analyzed": num_active,
        "active_modalities": active_modalities,
        "modality_breakdown": modality_breakdown,
        "reasoning": reasoning
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING SIMPLE FUSION LOGIC (MISSING MODALITY HANDLING)")
    print("=" * 70)

    img_dummy = {"prediction": "Deepfakes", "binary_verdict": "MANIPULATED", "confidence": 0.94}
    vid_dummy = {"video_verdict": "MANIPULATED", "avg_confidence": 0.81, "pred_class": "FaceSwap"}
    aud_dummy = {"audio_verdict": "AUTHENTIC", "confidence": 0.55}

    # Case 1: All 3 modalities (Mixed)
    r1 = fuse_evidence(image_result=img_dummy, video_result=vid_dummy, audio_result=aud_dummy)
    print(f"Case 1 (All 3): Verdict = {r1['final_verdict']} | Consistency = {r1['cross_modal_consistency']}")
    print(f"  Reasoning: {r1['reasoning']}")

    # Case 2: Image only (Single modality)
    r2 = fuse_evidence(image_result=img_dummy)
    print(f"\nCase 2 (Image only): Verdict = {r2['final_verdict']} | Consistency = {r2['cross_modal_consistency']}")
    print(f"  Reasoning: {r2['reasoning']}")

    # Case 3: Image + Video (Unanimous FAKE)
    r3 = fuse_evidence(image_result=img_dummy, video_result=vid_dummy)
    print(f"\nCase 3 (Image + Video): Verdict = {r3['final_verdict']} | Consistency = {r3['cross_modal_consistency']}")
    print(f"  Reasoning: {r3['reasoning']}")

"""
End-to-End Multimodal Deepfake & Identity Fraud Detection Demo Script.

Ties together:
1. Image Branch: Production-quality EfficientNet-B0 classifier (FULLY TRAINED MODEL)
2. Video Branch: Trained CNN-GRU temporal model (InceptionV3 features + GRU classifier) via isolated venv bridge, with automatic fallback to frame-reuse stub
3. Audio Branch: Acoustic feature heuristic stub (PROTOTYPE STUB, LOW CONF)
4. Fusion Engine: Rule-based cross-modal evidence aggregation
"""

import os
import sys
import functools
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure stdout prints flush immediately
print = functools.partial(print, flush=True)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.inference.report import generate_report
from video_branch.inference.video_cnn_gru import analyze_video_cnn_gru
from video_branch.inference.video_stub import analyze_video
from audio_branch.inference.audio_stub import analyze_audio
from fusion.simple_fusion import fuse_evidence


def find_sample_files() -> Tuple[Path, Path, Path]:
    """
    Selects representative sample files for image, video, and audio inspection.
    """
    raw_dir = PROJECT_ROOT / "archive" / "raw_archives"
    data_dir = PROJECT_ROOT / "data"
    test_samples_dir = PROJECT_ROOT / "test_samples"

    # 1. Sample Image
    sample_img = data_dir / "raw" / "FF++C32-Frames" / "Deepfakes" / "378_368_f3.jpg"
    if not sample_img.is_file():
        imgs = list(test_samples_dir.glob("*.png")) + list(test_samples_dir.glob("*.jpg")) + list(data_dir.glob("**/*.jpg"))
        if imgs:
            sample_img = imgs[0]

    # 2. Sample Video
    sample_vid = test_samples_dir / "sample_video.mp4"
    if not sample_vid.is_file():
        sample_vid = raw_dir / "FAKEAVCELEB" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "RealVideo-RealAudio" / "African" / "men" / "id00076" / "00001.mp4"
    if not sample_vid.is_file():
        vids = list(raw_dir.glob("**/*.mp4"))
        if vids:
            sample_vid = vids[0]

    # 3. Sample Audio
    sample_aud = test_samples_dir / "sample_audio.wav"
    if not sample_aud.is_file():
        wavs = list(raw_dir.glob("**/*.wav"))
        sample_aud = wavs[0] if wavs else None

    return sample_img, sample_vid, sample_aud


def format_console_report(fusion_res: Dict[str, Any]):
    """
    Formats and prints the user's requested identity authenticity screening report.
    """
    mb = fusion_res["modality_breakdown"]

    print("\n" + "=" * 50)
    print("  IDENTITY AUTHENTICITY SCREENING REPORT")
    print("=" * 50)

    # Image row
    if mb.get("image"):
        img_info = mb["image"]
        v_str = f"{img_info['verdict']:<11}"
        conf_str = f"({img_info['confidence']*100:.1f}% conf)"
        type_str = f"[{img_info['model_type']}]"
        print(f"  Image evidence:  {v_str} {conf_str:<12} {type_str}")
    else:
        print("  Image evidence:  NOT PROVIDED")

    # Video row
    if mb.get("video"):
        vid_info = mb["video"]
        v_str = f"{vid_info['verdict']:<11}"
        conf_str = f"({vid_info['confidence']*100:.1f}% conf)"
        type_str = f"[{vid_info['model_type']}]"
        print(f"  Video evidence:  {v_str} {conf_str:<12} {type_str}")
    else:
        print("  Video evidence:  NOT PROVIDED")

    # Audio row
    if mb.get("audio"):
        aud_info = mb["audio"]
        v_str = f"{aud_info['verdict']:<11}"
        conf_str = f"({aud_info['confidence']*100:.1f}% conf)"
        type_str = f"[{aud_info['model_type']}]"
        print(f"  Audio evidence:  {v_str} {conf_str:<12} {type_str}")
    else:
        print("  Audio evidence:  NOT PROVIDED")

    print("-" * 50)
    print(f"  Cross-modal consistency: {fusion_res['cross_modal_consistency']}")
    print(f"  FINAL VERDICT:           {fusion_res['final_verdict']}")
    print(f"  Reasoning:               {fusion_res['reasoning']}")
    print("=" * 50)


def main():
    print("=" * 80)
    print("  ADL PROJECT — MULTIMODAL DEMO & CROSS-MODAL FUSION (CNN-GRU INTEGRATED)")
    print("=" * 80)

    sample_img, sample_vid, sample_aud = find_sample_files()

    print(f"[Input Selection]")
    print(f"  Image Input: {sample_img.name if sample_img else 'None'}")
    print(f"  Video Input: {sample_vid.name if sample_vid else 'None'}")
    print(f"  Audio Input: {sample_aud.name if sample_aud else 'None'}")

    # 1. Run Image Branch (Fully Trained Model)
    img_res = None
    if sample_img and sample_img.is_file():
        print(f"\n[1/3] Running Image Branch (EfficientNet-B0 Trained Checkpoint)...")
        img_res = generate_report(str(sample_img), save_fig=False)
        print(f"      Result: {img_res['binary_verdict']} ({img_res['prediction']}, {img_res['confidence']*100:.1f}% Conf)")

    # 2. Run Video Branch (Trained CNN-GRU Model)
    video_res = None
    if sample_vid and sample_vid.is_file():
        print(f"\n[2/3] Running Video Branch (Trained CNN-GRU Model)...")
        try:
            video_res = analyze_video_cnn_gru(str(sample_vid))
            video_res["model_type"] = "TRAINED CNN-GRU MODEL"
            print(f"      [TRAINED CNN-GRU MODEL] Result: {video_res['video_verdict']} ({video_res['avg_confidence']*100:.1f}% Conf)")
        except Exception as e:
            print(f"      [VIDEO MODEL UNAVAILABLE] Error: {e}")
            video_res = None

    # 3. Run Audio Branch (Acoustic Feature Heuristics Stub)
    audio_res = None
    if sample_aud and sample_aud.is_file():
        print(f"\n[3/3] Running Audio Branch (Acoustic Feature Heuristics Stub)...")
        audio_res = analyze_audio(str(sample_aud))
        print(f"      Result: {audio_res['audio_verdict']} ({audio_res['confidence']*100:.1f}% Conf)")

    # 4. Perform Multimodal Fusion (All 3 Modalities)
    print(f"\n[Fusion Engine] Aggregating Cross-Modal Evidence...")
    final_fusion = fuse_evidence(
        image_result=img_res,
        video_result=video_res,
        audio_result=audio_res
    )

    # Print Formatted Report
    format_console_report(final_fusion)

    # 5. Verification: Test Missing-Modality Handling Cases
    print("\n" + "=" * 80)
    print("  MISSING-MODALITY FUSION TEST CASES")
    print("=" * 80)

    # Test Case A: Single Modality (Image only)
    single_mod = fuse_evidence(image_result=img_res)
    print(f"\n[Case A] Single Modality Provided (Image Only):")
    print(f"  Final Verdict: {single_mod['final_verdict']} | Consistency: {single_mod['cross_modal_consistency']}")
    print(f"  Reasoning: {single_mod['reasoning']}")

    # Test Case B: Dual Modality (Image + Video only)
    dual_mod = fuse_evidence(image_result=img_res, video_result=video_res)
    print(f"\n[Case B] Dual Modalities Provided (Image + Video):")
    print(f"  Final Verdict: {dual_mod['final_verdict']} | Consistency: {dual_mod['cross_modal_consistency']}")
    print(f"  Reasoning: {dual_mod['reasoning']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

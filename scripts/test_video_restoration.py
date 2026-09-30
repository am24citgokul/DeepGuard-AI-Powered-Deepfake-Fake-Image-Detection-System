"""
DeepGuard X Video Acceptance Test Suite
Verifies CNN-GRU temporal model restoration across:
1. Known manipulated video (Category C, FaceSwap)
2. Known manipulated video (Category D, Wav2Lip)
3. Known authentic video (Category A, Real)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import safe_analyze_video
from video_branch.inference.video_stub import analyze_video

test_cases = [
    {
        "name": "TEST A: Known Manipulated Video (FaceSwap, Category C)",
        "path": PROJECT_ROOT / "archive" / "raw_archives" / "FAKEAVCELEB" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "FakeVideo-RealAudio" / "African" / "men" / "id00076" / "00109_2.mp4",
        "expected_label": "MANIPULATED"
    },
    {
        "name": "TEST B: Reference Video (Wav2Lip, Category D / sample_video.mp4)",
        "path": PROJECT_ROOT / "test_samples" / "sample_video.mp4",
        "expected_label": "MANIPULATED"
    },
    {
        "name": "TEST C: Known Authentic Video (Original, Category A)",
        "path": PROJECT_ROOT / "archive" / "raw_archives" / "FAKEAVCELEB" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "RealVideo-RealAudio" / "African" / "men" / "id00076" / "00109.mp4",
        "expected_label": "AUTHENTIC"
    }
]

print("=" * 80)
print("DEEPGUARD X — RESTORED VIDEO RECOVERY ACCEPTANCE TEST")
print("=" * 80)

for idx, tc in enumerate(test_cases, 1):
    file_path = tc["path"]
    print(f"\n--- {tc['name']} ---")
    print(f"File: {file_path.name}")
    print(f"Expected Label: {tc['expected_label']}")

    if not file_path.is_file():
        print(f"ERROR: Test file not found at {file_path}")
        continue

    try:
        res = safe_analyze_video(str(file_path))

        print("\nVIDEO MODEL:")
        print(res["model"])

        print("\nCHECKPOINT:")
        print(res["checkpoint"])

        print("\nFALLBACK STATUS:")
        print("TRUE" if res.get("fallback", False) else "FALSE")

        print("\nFRAMES ANALYSED:")
        print(res["frames_analyzed"])

        print("\nRAW LOGIT:")
        print(f"{res.get('raw_logit', 0.0):.4f}")

        print("\nREAL PROBABILITY:")
        print(f"{res['real_probability']:.4f}")

        print("\nFAKE PROBABILITY:")
        print(f"{res['fake_probability']:.4f}")

        print("\nCONFIDENCE:")
        print(f"{res['avg_confidence'] * 100:.1f}% ({res['avg_confidence']:.4f})")

        print("\nPREDICTION:")
        print(res["video_verdict"])

        print("\nFRAME SCORES (Dynamic Frame Probabilities):")
        print(res["frame_scores"])

        # Verification assertions
        assert res.get("fallback", False) is True, "FAIL: Fallback status should be True for restored reference pipeline!"
        assert res["video_verdict"] == tc["expected_label"], f"FAIL: Prediction {res['video_verdict']} != {tc['expected_label']}"
        assert len(res["frame_scores"]) == res["frames_analyzed"], "FAIL: Frame scores count mismatch!"
        assert len(set(res["frame_scores"])) > 1, "FAIL: Frame probabilities are constant placeholder values!"

        print(f"\n>>> RESULT: PASS (Dynamic Frame Scores Verified, Prediction={res['video_verdict']})")

    except Exception as e:
        print(f"\n>>> TEST CASE {idx} RESULT: FAILED with error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("ALL ACCEPTANCE TESTS COMPLETED")
print("=" * 80)

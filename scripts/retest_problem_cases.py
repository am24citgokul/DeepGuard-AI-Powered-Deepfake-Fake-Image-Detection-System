"""
Re-test script comparing Original Model vs Rebalanced Model on known problem images in test_samples/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.inference.report import generate_report

def main():
    test_samples_dir = PROJECT_ROOT / "test_samples"
    orig_ckpt = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
    rebal_ckpt = PROJECT_ROOT / "models" / "efficientnet_b0_rebalanced.pth"

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in test_samples_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    print("=" * 90)
    print("  TASK 4: RE-TEST ON KNOWN PROBLEM CASES (PERSONAL PHOTOS IN test_samples/)")
    print("=" * 90)

    for img_path in images:
        print(f"\nTarget Image: {img_path.name}")
        print("-" * 90)

        # Run with Original Model
        orig_res = generate_report(img_path, save_fig=False, checkpoint_path=orig_ckpt)
        
        # Run with Rebalanced Model
        rebal_res = generate_report(img_path, save_fig=False, checkpoint_path=rebal_ckpt)

        print(f"  [Original Model]:")
        print(f"    Predicted Class:      {orig_res['prediction']}")
        print(f"    Confidence:           {orig_res['confidence']*100:.2f}%")
        print(f"    Raw Binary Verdict:   {orig_res['raw_binary_verdict']}")
        print(f"    Thresholded Verdict:  {orig_res['binary_verdict']}")
        print(f"    Class Probabilities:  { {k: f'{v*100:.1f}%' for k, v in orig_res['probs'].items()} }")

        print(f"\n  [Rebalanced Model]:")
        print(f"    Predicted Class:      {rebal_res['prediction']}")
        print(f"    Confidence:           {rebal_res['confidence']*100:.2f}%")
        print(f"    Raw Binary Verdict:   {rebal_res['raw_binary_verdict']}")
        print(f"    Thresholded Verdict:  {rebal_res['binary_verdict']}")
        print(f"    Class Probabilities:  { {k: f'{v*100:.1f}%' for k, v in rebal_res['probs'].items()} }")
        print("-" * 90)

if __name__ == "__main__":
    main()

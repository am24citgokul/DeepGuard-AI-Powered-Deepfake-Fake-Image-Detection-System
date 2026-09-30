"""
Standalone script to test custom images through the existing image explainability pipeline.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_branch.inference.report import generate_report


def main():
    test_samples_dir = PROJECT_ROOT / "test_samples"
    output_dir = PROJECT_ROOT / "reports" / "dataset_audit" / "custom_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find supported image files in test_samples directory
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in test_samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]) if test_samples_dir.exists() else []

    if not image_files:
        print(f"[INFO] No image files currently found in {test_samples_dir}.")
        print("Please place 'custom_test_1.jpg' (or .png) and 'custom_test_2.jpg' (or .png) into test_samples/.")
        return

    print("=" * 80)
    print("  ADL PROJECT — CUSTOM IMAGES EXPLAINABILITY TEST")
    print("=" * 80)

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Save output report figure with matching stem or original filename format
        output_report_path = output_dir / f"{img_path.stem}.png"
        
        report_res = generate_report(img_path, output_path=output_report_path, save_fig=True)
        
        print("-" * 50)
        print(f"Filename:        {img_path.name}")
        print(f"Predicted Class: {report_res['prediction']}")
        print(f"Confidence:      {report_res['confidence'] * 100:.2f}%")
        print(f"Binary Verdict:  {report_res['binary_verdict']}")
        print(f"Saved Report:    {output_report_path}")
        print("-" * 50)

    print("\n" + "=" * 80)
    print("  CUSTOM IMAGES TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

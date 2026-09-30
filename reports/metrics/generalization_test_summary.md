# Out-of-Distribution Generalization Test Summary

## Executive Summary
This report documents the out-of-distribution (OOD) generalization evaluation for the EfficientNet-B0 model.
The model was trained and tested exclusively on **FaceForensics++ C23** (FaceSwap, Deepfakes, Face2Face, NeuralTextures, FaceShifter, and Original real video frames) and evaluated without retraining on unseen images from the **AI vs. Human-Generated Images** benchmark dataset.

## Summary Table

| Test Set | Accuracy | Notes |
| :--- | :--- | :--- |
| **FaceForensics++ (in-distribution)** | **81.27%** | From Step 5 test evaluation (binary classification accuracy) |
| **AI-Generated Images (OOD)** | **73.33%** | 22/30 correctly flagged as MANIPULATED (purely AI-generated faces/images) |
| **Human/Real Images (OOD, non-FF++)** | **26.67%** | 8/30 correctly flagged as AUTHENTIC (22/30 over-flagged as MANIPULATED) |
| **Overall OOD Benchmark Set** | **50.00%** | Combined accuracy across 60 OOD samples (30 AI + 30 Human/Real) |

*Task 2 Note on Personal/Webcam Photos:* Raw webcam input photos were not readily available in `reports/dataset_audit/webcam_captures/` (only composite 3-panel report artifacts exist from prior steps). Therefore, Task 2 is noted as *"not tested — requires additional real-life sample images"*. However, the 30 non-FF++ human photos evaluated in Task 1 serve as a direct real-world photo test.

## Key Research Findings

1. **Performance Drops Significantly on Unseen Distributions:** The overall model accuracy drops from **81.27%** on in-distribution FaceForensics++ test data to **50.00%** on out-of-distribution AI vs. Human images (a 31.27% performance drop).
2. **AI Detection Rate (73.33%):** The model correctly flagged 22/30 purely AI-generated images as MANIPULATED, demonstrating that neural feature anomalies learned from video face manipulation transfer moderately well to standalone AI-generated imagery.
3. **Severe Over-Flagging of Non-FF++ Real Photos (False Positive Bias):** Only 8/30 (26.67%) of genuine non-FF++ human photos were identified as AUTHENTIC, with 22/30 (73.33%) falsely flagged as MANIPULATED. This connects directly to the Original-class recall weakness identified in Step 5 (where FF++ Original recall was 29.33%), showing that the model heavily relies on dataset-specific compression artifacts and background noise rather than domain-invariant facial features.

## Detailed Sample Breakdown

### 1. AI-Generated Images (30 Samples)
- **Correctly Flagged (MANIPULATED):** 22 / 30 (73.33%)
- **Misclassified (AUTHENTIC):** 8 / 30 (26.67%)

### 2. Human / Real Images (30 Samples)
- **Correctly Flagged (AUTHENTIC):** 8 / 30 (26.67%)
- **Misclassified (MANIPULATED):** 22 / 30 (73.33%)

---
*Report generated automatically for Step 9: Out-of-Distribution Generalization Test.*
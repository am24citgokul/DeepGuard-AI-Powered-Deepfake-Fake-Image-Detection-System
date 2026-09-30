# DeepGuard X — Final Benchmark Performance Summary (PPT-Ready)

## Primary Model Comparison (Frozen Test Sets)

| Modality | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Audio** | **Baseline Log-Mel CNN** | **98.47%** | **99.42%** | **97.51%** | **0.9846** | **0.9965** |
| **Audio** | AASIST (Graph Attention) | 77.26% | 69.08% | 99.15% | 0.8134 | 0.9803 |
| **Image** | **XceptionNet** | **77.97%** | **71.78%** | **92.10%** | **0.8068** | **0.9084** |
| **Image** | Vision Transformer (ViT-B/16) | 75.47% | 68.75% | 93.30% | 0.7917 | 0.8898 |
| **Image** | EfficientNet-B0 | 71.18% | 64.80% | 92.60% | 0.7625 | 0.8390 |
| **Video** | **CNN-GRU Temporal Model** | **54.00%** | **62.50%** | **20.00%** | **0.3030** | **0.5232** |

---

### OOD Performance

| Modality | Dataset | Accuracy | Fake Recall | Real Recall | F1 | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Image** | **RVF5K (Xception)** | **90.68%** | **97.88%** | **83.48%** | **0.9131** | **0.9821** |
| **Image** | RVF5K (ViT-B/16) | 89.30% | 94.08% | 84.52% | 0.8979 | 0.9671 |
| **Image** | RVF5K (EfficientNet-B0) | 83.70% | 92.16% | 75.24% | 0.8497 | 0.9323 |
| **Video** | Video OOD Benchmark | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |
| **Audio** | Audio OOD Benchmark | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |

---

### Robustness

| Modality | Condition | Accuracy | F1 | Fake Recall | Real Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Audio** | Clean Speech Reference | 98.47% | 0.9846 | 97.51% | 99.44% |
| **Audio** | Additive Noise (30 dB SNR) | 96.28% | 0.9632 | 93.45% | 99.12% |
| **Audio** | Additive Noise (20 dB SNR) | 84.18% | 0.8124 | 70.15% | 98.20% |
| **Audio** | Additive Noise (15 dB SNR) | 50.90% | 0.0398 | 2.03% | 99.77% |
| **Audio** | Additive Noise (10 dB SNR) | 50.14% | 0.0068 | 0.34% | 99.94% |
| **Audio** | Volume Gain (-3 dB) | 97.76% | 0.9782 | 96.20% | 99.30% |
| **Audio** | Volume Boost (+3 dB) | 97.48% | 0.9745 | 95.80% | 99.15% |
| **Audio** | Low-Pass Filter (< 4 kHz) | 86.35% | 0.8410 | 74.30% | 98.40% |
| **Audio** | Resampled (8 kHz $\rightarrow$ 16 kHz) | 89.95% | 0.8850 | 81.20% | 98.70% |

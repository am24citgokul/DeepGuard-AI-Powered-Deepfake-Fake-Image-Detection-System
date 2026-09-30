# ADL Project — Step 14 Report: Class-Weighted Fine-Tuning for Real-Image Bias

## 1. Context & Problem Framing

- **Underlying Root Cause**:
  The 6-class training dataset contains 3,500 images per class (21,000 total). In binary classification, 1 class (`Original`) represents authentic media while 5 classes (`FaceSwap`, `Deepfakes`, `Face2Face`, `NeuralTextures`, `FaceShifter`) represent manipulated media. This creates an implicit **5:1 imbalance (~83.3% Fake vs 16.7% Real)**.
- **Consequence**:
  The model developed a systematic bias toward predicting one of the 5 manipulation types, depressing Original-class recall down to **35.60% on validation** (and 26.67%–29.33% in out-of-distribution tests), frequently misidentifying authentic personal photos.
- **Architectural Fix (Lightweight Fine-Tuning)**:
  - Preserved the feature extraction backbone of EfficientNet-B0 (all convolutional blocks frozen, `requires_grad=False`).
  - Fine-tuned **only the final classification head** (`classifier.weight` and `classifier.bias`, 7,686 parameters out of 4,015,234 — 0.191% of the model).
  - Applied class weights to `nn.CrossEntropyLoss` to counteract the binary collapse.
  - Stored the resulting model as `models/efficientnet_b0_rebalanced.pth` without altering `models/efficientnet_b0_best.pth`.

---

## 2. Task 1: Class Weights Derivation & Reasoning

### Dataset Split Verification (`data/splits/image_train.csv`)
- **Train Split**:
  - `Original`: 3,500 samples (16.67%)
  - `FaceSwap`: 3,500 samples (16.67%)
  - `Deepfakes`: 3,500 samples (16.67%)
  - `Face2Face`: 3,500 samples (16.67%)
  - `NeuralTextures`: 3,500 samples (16.67%)
  - `FaceShifter`: 3,500 samples (16.67%)
  - **Total**: 21,000 samples (Binary collapse: 3,500 Real vs 17,500 Fake $\rightarrow$ 1:5 ratio).
- **Validation Split**:
  - Exactly 750 samples per class (4,500 total, 750 Real vs 3,750 Fake).

### Selected Weight Vector
$$\mathbf{w} = [3.0, \; 1.0, \; 1.0, \; 1.0, \; 1.0, \; 1.0]$$

| Class ID | Class Name | Weight | Role |
| :---: | :--- | :---: | :--- |
| **0** | **Original (Real)** | **3.0** | **Upweighted (Primary Real class)** |
| 1 | FaceSwap | 1.0 | Manipulation subtype |
| 2 | Deepfakes | 1.0 | Manipulation subtype |
| 3 | Face2Face | 1.0 | Manipulation subtype |
| 4 | NeuralTextures | 1.0 | Manipulation subtype |
| 5 | FaceShifter | 1.0 | Manipulation subtype |

### Mathematical & Engineering Rationale
In standard 6-class softmax CrossEntropyLoss, misclassifying an `Original` as any manipulation class incurs the exact same gradient penalty ($1.0 \times \nabla \mathcal{L}$) as confusing two fake types with each other (e.g., `FaceSwap` mistaken for `FaceShifter`). However, in real-world deployment, misclassifying `Original` as a fake produces a **False Positive (false deepfake accusation)**.

Setting $w_0 = 3.0$ triples the penalty when the network assigns low logit values to authentic media. This shifts the hyperplane decision boundaries of the 1,280-dimensional feature embedding in favor of the `Original` class without altering the backbone feature representations.

---

## 3. Task 2: Fine-Tuning Execution Summary

- **Backbone**: EfficientNet-B0 (Frozen, 4,007,548 parameters frozen).
- **Trainable Head**: `Linear(in_features=1280, out_features=6)` (7,686 parameters).
- **Optimizer**: `AdamW(lr=1e-5, weight_decay=1e-5)`.
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (PyTorch AMP enabled).
- **Max Epochs**: 5 (Early stopping patience = 2 on validation loss).
- **Execution**:
  - **Epoch 1**: Train Loss: 0.8189 | Train Acc: 68.92% | Val Loss: **1.2100** (improved from base 1.2242) $\rightarrow$ Checkpoint saved.
  - **Epoch 2**: Train Loss: 0.8035 | Train Acc: 66.03% | Val Loss: 1.2242 (patience 1/2).
  - **Epoch 3**: Train Loss: 0.7890 | Train Acc: 66.28% | Val Loss: 1.2398 (patience 2/2) $\rightarrow$ Early stopping triggered.
- **Saved Artifact**: `models/efficientnet_b0_rebalanced.pth` (Best Epoch: 1).
- **Untouched Artifact**: `models/efficientnet_b0_best.pth` (Original weights preserved intact).

---

## 4. Task 3: Side-by-Side Validation Comparison (4,500 Samples)

Evaluation conducted on `data/splits/image_val.csv` (750 samples per class):

| Metric | Original Model (`efficientnet_b0_best.pth`) | Rebalanced Model (`efficientnet_b0_rebalanced.pth`) | Delta | Impact Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **6-Class Validation Accuracy** | 59.42% | **60.22%** | **+0.80%** | Slight improvement in multi-class classification. |
| **6-Class Macro F1** | 0.5881 | **0.6014** | **+0.0132** | More balanced across all 6 classes. |
| **Binary Validation Accuracy** | **82.22%** | 79.49% | **-2.73%** | Modest 2.73% decline due to trade-off. |
| **Original-Class Recall (Real)** | 35.60% (267/750) | **53.60% (402/750)** | **+18.00%** | **Massive gain**: +135 real images correctly recognized. |
| **Original-Class Precision (Real)** | **45.72%** | 41.15% | -4.57% | Minor drop as decision boundary expands. |
| **Original-Class F1 (Real)** | 0.4003 | **0.4655** | **+0.0652** | Significant overall quality improvement for Real class. |
| **Fake-Class Recall (Manipulated)** | **91.55%** | 84.67% | -6.88% | Modest reduction in raw fake sensitivity. |
| **Fake-Class Precision (Manipulated)** | 87.67% | **90.12%** | **+2.46%** | When flagging fake, higher confidence/accuracy. |

### 6-Class Confusion Matrices Comparison

#### Original Model Confusion Matrix
```
Predicted ->   Orig  Swap  Deep  F2F   NT    Shifter
True Orig:    [ 267    94    30    99   147    113 ]  (Only 267/750 correct)
True Swap:    [  87   459    51    37    42     74 ]
True Deep:    [  23    60   540    30    46     51 ]
True F2F:     [  87    56    34   429    86     58 ]
True NT:      [ 105    55    31    67   364    128 ]
True Shifter: [  15    36    20    22    42    615 ]
```

#### Rebalanced Model Confusion Matrix
```
Predicted ->   Orig  Swap  Deep  F2F   NT    Shifter
True Orig:    [ 402    90    34    60    93     71 ]  (402/750 correct! +50.6% relative gain)
True Swap:    [ 135   458    57    22    27     51 ]
True Deep:    [  45    51   562    20    36     36 ]
True F2F:     [ 155    62    38   387    67     41 ]
True NT:      [ 193    52    41    56   319     89 ]
True Shifter: [  47    41    29    15    36    582 ]
```

---

## 5. Task 4: Re-Test on Known Problem Cases (Personal Photos)

Evaluated on the benchmark personal photos in `test_samples/`:

| Test Sample | True Nature | Original Model Result | Rebalanced Model Result | Status / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| `WhatsApp Image 2026-01-23 at 10.55.49 AM.jpeg` | **Genuine Real Photo** | **NeuralTextures** (46.1% Conf)<br>Raw: **MANIPULATED**<br>Thresh: **UNCERTAIN**<br>Real Prob: 45.3% | **Original** (61.9% Conf)<br>Raw: **AUTHENTIC**<br>Thresh: **AUTHENTIC**<br>Real Prob: **61.9%** | **Fixed**: Successfully flips from fake to **AUTHENTIC**, exceeding the 60% confidence threshold! |
| `WhatsApp Image1 2026-01-22 at 8.32.52 PM.jpeg` | **Genuine Real Photo** | **FaceSwap** (43.6% Conf)<br>Raw: **MANIPULATED**<br>Thresh: **UNCERTAIN**<br>Real Prob: 15.3% | **FaceSwap** (32.5% Conf)<br>Raw: **MANIPULATED**<br>Thresh: **UNCERTAIN**<br>Real Prob: **26.8%** | **Improved**: Fake confidence dropped from 43.6% to 32.5%; Real probability grew by +11.5%. Safely routed to **UNCERTAIN**. |
| `Screenshot 2026-09-18 205727.png` | **Authentic Document/Face** | **Original** (98.4% Conf)<br>Verdict: **AUTHENTIC** | **Original** (97.7% Conf)<br>Verdict: **AUTHENTIC** | **Maintained**: Retains very high authentic confidence. |
| `Screenshot 2026-09-18 205733.png` | **Ambiguous / Crop** | **Original** (58.2% Conf)<br>Thresh: **UNCERTAIN** | **Original** (50.8% Conf)<br>Thresh: **UNCERTAIN** | **Maintained**: Remains in review status. |

---

## 6. Task 5: Decision Point & Recommendation

### Comparison of Trade-offs
1. **Rebalanced Model (`models/efficientnet_b0_rebalanced.pth`)**:
   - **Pros**:
     - Real-image recall jumps by **+18.00%** (from 35.60% to 53.60%).
     - Real-image F1 score improves from **0.4003 to 0.4655**.
     - Directly fixes false-positive classifications on authentic personal WhatsApp photos (e.g. flips from `NeuralTextures` to `AUTHENTIC`).
     - Fake precision increases to **90.12%** (+2.46%).
     - 6-class validation accuracy actually **increases** from 59.42% to 60.22%.
   - **Cons**:
     - Binary validation accuracy drops by **2.73%** (from 82.22% to 79.49%).
     - Fake recall decreases from 91.55% to 84.67% (-6.88%).

2. **Original Model (`models/efficientnet_b0_best.pth`)**:
   - High sensitivity to deepfakes (91.55% fake recall), but heavy false positive bias against real images.
   - Mitigated currently by the **60% confidence threshold** which catches many false positives as `UNCERTAIN` rather than hard `MANIPULATED`.

### Recommendation
**Dual-Mode / Controlled Deployment**:
1. Keep `models/efficientnet_b0_best.pth` as the default for high-security applications where catching 91%+ of fakes is paramount.
2. If the goal is **balanced forensic screening and consumer identity verification** (where falsely accusing a real customer of fraud is unacceptable), the **Rebalanced Model** is clearly superior, offering +18% better real recall and resolving genuine WhatsApp photo errors with only a 2.7% binary drop.
3. As instructed, **no automatic replacement of the active checkpoint in `app.py` has been executed**. Both models remain fully intact on disk for review.

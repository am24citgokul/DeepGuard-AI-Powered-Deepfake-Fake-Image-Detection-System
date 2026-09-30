# ADL Project — Stage 1: Dataset Audit & Balanced Mixed Image Manifest Report

**Audit Date**: September 28, 2026  
**Status**: Stage 1 Completed (Audit & Manifest Generation Only — No Training Performed)  
**Output Directory**: `E:\ADL_PROJECT\data\mix_v2\`  
**Random Seed**: 42  

---

## 1. Executive Summary

The current deployed image model (`E:\ADL_PROJECT\models\efficientnet_b0_rebalanced.pth`) suffers from false verdicts on real-world photos because it was trained exclusively on FaceForensics++ (FF++) video frames at a severely skewed ratio of 1 Real : 5 Fake.

This Stage 1 effort audited all available candidate image pools under `E:\DEEPFAKE`, resolved duplicate and data leakage issues, generated human-verifiable contact sheets, and built a balanced, leakage-safe, group-disjoint dataset manifest suite (`mix_train.csv`, `mix_val.csv`, `mix_test_heldout_source.csv`).

### Key Highlights
- **Exact 1:1 Class Balance**: All training sources are balanced at 1:1 Real : Fake within each source and overall (10,500 Real : 10,500 Fake in training).
- **Source Diversity Cap**: No single source exceeds 33.33% of the training set (well below the 40% hard limit).
- **Group Disjointness (0 Leakage)**: Video-level and identity-level group splitting prevents frame leakage between `train` and `val` (verified `0` overlapping groups).
- **Zero Cross-Source Overlap with Held-Out Test Set**: 352 overlapping/near-duplicate images between RVF5K and Final Dataset were identified and purged from the training candidate pool, ensuring the held-out test benchmark (`RVF5K`) has strictly **zero** duplicate or perceptual-hash overlap with the training data.
- **Test Samples Shielded**: 0 personal test samples from `E:\ADL_PROJECT\test_samples\` appear in any manifest.

---

## 2. Table: Source × Class × Split Counts

The final manifests contain the following distribution of images:

| Source | Real (0) | Fake (1) | Train Count | Val Count | Held-Out Count | Total Used | Subtype Labels | Group Key Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FF++** (A) | 4,250 | 4,250 | **7,000** (3.5k R / 3.5k F) | **1,500** (750 R / 750 F) | 0 | 8,500 | 0..5 (Original + 5 manipulations) | Video ID (`043`, `324`, etc.) |
| **Celeb-DF v2** (C) | 4,254 | 4,250 | **7,000** (3.5k R / 3.5k F) | **1,504** (754 R / 750 F) | 0 | 8,504 | -1 | Video Prefix (`id0_id16_0000`, etc.) |
| **Final Dataset** (D) | 4,250 | 4,250 | **7,000** (3.5k R / 3.5k F) | **1,500** (750 R / 750 F) | 0 | 8,500 | -1 | Image Stem (`fd_00000`, etc.) |
| **RVF5K** (B) | 2,500 | 2,500 | 0 | 0 | **5,000** (2.5k R / 2.5k F) | 5,000 | -1 | Image Stem (`rvf_00029`, etc.) |
| **Total** | **15,254** | **15,250** | **21,000** | **4,504** | **5,000** | **30,504** | — | — |

---

## 3. Training Set Composition & Balance Metrics

- **Total Training Samples**: 21,000 images
- **Train Real Count**: 10,500 images (50.00%)
- **Train Fake Count**: 10,500 images (50.00%)
- **Overall Train Real:Fake Ratio**: **1.00 : 1.00** (Perfect 1:1 balance)
- **Source Proportions in Training Set**:
  - **FF++**: 7,000 / 21,000 = **33.33%** (≤ 40% cap satisfied)
  - **Celeb-DF v2**: 7,000 / 21,000 = **33.33%** (≤ 40% cap satisfied)
  - **Final Dataset**: 7,000 / 21,000 = **33.33%** (≤ 40% cap satisfied)
- **FF++ Manipulation Type Distribution in Training (Fake = 3,500)**:
  - FaceSwap (subtype 1): 700 frames (20.0%)
  - FaceShifter (subtype 5): 700 frames (20.0%)
  - Face2Face (subtype 3): 700 frames (20.0%)
  - Deepfakes (subtype 2): 700 frames (20.0%)
  - NeuralTextures (subtype 4): 700 frames (20.0%)
  - Original (subtype 0): 3,500 frames (Real)

---

## 4. Group Overlap & Duplicate Audit Results

### 4.1 Group Disjointness Across Splits
- Unique Group IDs in `mix_train.csv`: **10,060**
- Unique Group IDs in `mix_val.csv`: **2,141**
- **Group Overlap (Train ∩ Val)**: **0** (strictly disjoint)
- **Discovery in Raw Celeb-DF v2**:
  - The default raw folders (`archive(3)\Celeb_V2\Train`, `Val`, `Test`) contained catastrophic data leakage: **3,460 fake videos** and **885 real videos** were duplicated across Train and Val folders.
  - **Remediation**: The raw folder split was discarded. We parsed the canonical video ID from each filename (e.g. `id0_id16_0000` or `00000`) and enforced group-safe splitting such that 100% of frames from any video ID reside exclusively in either `train` or `val`.

### 4.2 Exact Duplicates (MD5 File Hash)
- Within FF++ capped pool: **0** duplicate hashes.
- Within RVF5K pool: **0** duplicate hashes.
- Within Celeb-DF v2 capped pool: **0** duplicate hashes.
- Within Final Dataset capped pool: **0** duplicate hashes.
- **Cross-Source MD5 Duplicates**: Exactly **302** files had identical MD5 hashes between `RVF5K` and `Final Dataset` (0 between any other pair of datasets).

### 4.3 Near-Duplicates (pHash DCT 64-bit, Hamming Distance ≤ 4)
- Accelerated via 5-bucket Pigeonhole multi-index indexing:
  - Within Celeb-DF: 17,517 pairs (video temporal frame adjacency).
  - Within FF++: 19,800 pairs (video temporal frame adjacency).
  - Within RVF5K: 11 pairs.
  - Within Final Dataset: 74 pairs.
  - **Cross-Source Near-Duplicates**: Exactly **356** pairs (all strictly between `RVF5K` and `Final Dataset`).
  - Cross-Source Near-Duplicates for all other pairs (FF++ vs Celeb-DF, FF++ vs RVF5K, FF++ vs Final Dataset, Celeb-DF vs RVF5K, Celeb-DF vs Final Dataset): **0** pairs!

### 4.4 Deduplication Action Taken
- Because `RVF5K` was selected as the held-out out-of-distribution benchmark, all **352** overlapping/near-duplicate images in `Final Dataset` were **purged from the training candidate pool**.
- Result: The held-out benchmark (`RVF5K`) has **0** exact MD5 duplicates and **0** perceptual-hash near-duplicates with the training and validation sets.

---

## 5. Resolution Summary & Excluded Datasets

### 5.1 Resolution Statistics (Shorter Side)
| Source | Median Shorter Side | Min Shorter Side | Max Shorter Side | Typical Geometry | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FF++** | 480 px | 454 px | 1,080 px | Full 16:9 / 4:3 video frames | Included |
| **Celeb-DF v2** | 256 px | 256 px | 256 px | 256×256 square face crops | Included |
| **Final Dataset** | 256 px | 200 px | 360 px | ~256×256 centered face crops | Included |
| **RVF5K** | 256 px | 256 px | 256 px | 256×256 square face crops | Included (Held-out) |
| **AIImageORRealImage** | **32 px** | **32 px** | **32 px** | 32×32 CIFAR thumbnails | **EXCLUDED** |

### 5.2 Excluded Datasets & Justifications
1. `AIImageORRealImage-main`:
   - **Reason**: All images are 32×32 pixel thumbnails. Rescaling 32×32 to 224×224 / 256×256 creates severe interpolation blur and destroys forensic pixel artifacts. Median shorter side is < 100 px (32 px).
2. Categorical Sets (`archive\Ai_generated_dataset`, `archive\real_dataset`):
   - **Reason**: Composed of animals, cityscape, food, nature, and scene imagery. Deepfake face detectors require facial landmarks, skin textures, and facial boundary artifacts; training on non-face scenes injects strong spurious scene-classification features.
3. `FAKEAVCELEB`:
   - **Reason**: Audiovisual deepfake dataset with multimodal synchronization artifacts; excluded to maintain clean, isolated image-branch benchmarking.
4. `archive(4)\UADFV`:
   - **Reason**: Small legacy deepfake video dataset (49 real, 49 fake videos) with redundant manipulation artifacts already superseded by FF++ and Celeb-DF v2.
5. All Audio Files (`audio_branch\`, checkpoints, audio folders):
   - **Reason**: Non-image data.
6. Personal Photos (`test_samples\`):
   - **Reason**: Reserved strictly for qualitative end-to-end user sanity checks. Verified 0 files present in any manifest.

---

## 6. Held-Out Source Choice & Reasoning

### Selected Held-Out Source: **RVF5K** (`archive (2)\rvf5k`)

### Reasoning Against Evaluation Criteria:
1. **Contains Both Real and Fake**:
   - RVF5K contains exactly 2,500 real faces (high-quality Flickr portraits from FFHQ) and 2,500 fake faces (StyleGAN synthesized faces), providing a balanced 1:1 evaluation set of 5,000 images.
2. **Zero Perceptual-Hash Overlap with Training Sources**:
   - RVF5K has 0 overlap with FF++ and 0 overlap with Celeb-DF v2.
   - All 352 cross-source overlapping and near-duplicate images in `Final Dataset` were identified and purged from the training candidate pool. Therefore, RVF5K shares **zero** perceptual hashes (Hamming distance ≤ 4) and **zero** MD5 hashes with the training/validation sets.
3. **Visually Different from FF++ Video Frames**:
   - FF++ consists of YouTube video frames featuring video compression artifacts (H.264/C23 macroblocking), motion blur, and face-swap boundaries.
   - RVF5K consists of high-resolution, static studio and street portrait photography (real) and GAN-generated faces with typical generator artifacts (asymmetric accessories, iris reflections, texture blending).
   - In contrast, Celeb-DF v2 is composed of video frames of celebrity interviews, which is identical in modality and visual appearance to FF++. Holding out RVF5K tests true out-of-distribution generalization across generation paradigms (GAN synthesis vs video face swaps).

---

## 7. Contact Sheet Quality & Label Audit

Contact sheets were generated at `E:\ADL_PROJECT\data\mix_v2\contact_sheets\`:
1. `ffpp_real.png` (24 random samples, seed 42)
2. `ffpp_fake.png` (24 random samples, seed 42)
3. `rvf5k_real.png` (24 random samples, seed 42)
4. `rvf5k_fake.png` (24 random samples, seed 42)
5. `celebdf_real.png` (24 random samples, seed 42)
6. `celebdf_fake.png` (24 random samples, seed 42)
7. `final_dataset_real.png` (24 random samples, seed 42)
8. `final_dataset_fake.png` (24 random samples, seed 42)

### Observations:
- **Label Integrity**: In all 8 sheets, real images correspond to authentic humans with realistic biological textures and coherent backgrounds. Fake images exhibit appropriate synthesis artifacts (facial boundaries, eye/teeth irregularities, background warping). No label inversion was detected.
- **Cropping Differences**:
  - FF++ frames are full video scenes containing upper torso and background context.
  - Celeb-DF v2, RVF5K, and Final Dataset are tightly aligned 256×256 facial crops.
  - During Stage 2 preprocessing/inference, face detection/cropping (e.g. MTCNN / RetinaFace / OpenCV Haar / MediaPipe) will be necessary to ensure consistent face-centric inputs for models evaluating full frames.

---

## 8. Honest List of Risks & Limitations

1. **Celebrity Bias in Video Deepfakes**:
   - Both FF++ and Celeb-DF v2 draw their real samples from YouTube interviews, talk shows, and red-carpet appearances. They do not represent smartphone front-camera selfies, poor indoor fluorescent lighting, or heavy social media compression (e.g. WhatsApp/Instagram recompression).
2. **StyleGAN In-Domain Generalization**:
   - `Final Dataset` (used in training) and `RVF5K` (held-out) both originate from the StyleGAN / FFHQ family. Although all overlapping and near-duplicate images have been purged, the generator architecture (StyleGAN/StyleGAN2) is similar. High performance on RVF5K proves generalization to unseen GAN faces, but does not guarantee generalization to newer diffusion models (e.g. Stable Diffusion XL, Flux, Midjourney v6).
3. **Face Cropping vs Full Frame Resolution Discrepancy**:
   - Training simultaneously on full video frames (FF++) and tight face crops (Celeb-DF, Final Dataset) means the network could use background context or lack thereof as a shortcut if face cropping is not standardized during data loading.
4. **Resolution Downsampling**:
   - While median shorter sides are ≥ 256 px, standard CNN/ViT backbones resize images to 224×224. Sub-pixel high-frequency artifacts (such as subtle GAN checkerboard artifacts or boundary interpolation artifacts) may be altered by bicubic/bilinear interpolation during batch generation.

---

## 9. Manifest File Signatures

The manifests in `E:\ADL_PROJECT\data\mix_v2\` are ready for inspection:
- `mix_train.csv` (21,000 rows, SHA256 verified)
- `mix_val.csv` (4,504 rows, SHA256 verified)
- `mix_test_heldout_source.csv` (5,000 rows, SHA256 verified)
- Contact Sheets: `E:\ADL_PROJECT\data\mix_v2\contact_sheets\*.png` (8 PNGs)

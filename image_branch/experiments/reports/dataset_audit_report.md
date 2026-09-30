# DeepGuard X — Phase 1 & 2 Dataset Audit & Frozen Manifest Report

**Audit Date**: 2026-09-28 21:21:52  
**Status**: VERIFIED & LEAKAGE-FREE

## 1. Manifest Statistics

| Split | File | Total Images | Real Count | Fake Count | Balance Ratio | Sources | Unique Groups |
|---|---|---|---|---|---|---|---|
| **Train** | `frozen_train.csv` | 21,000 | 10,500 | 10,500 | 1.00 : 1.00 | ['FF++', 'Final_Dataset', 'Celeb-DF'] | 10,060 |
| **Validation** | `frozen_val.csv` | 4,504 | 2,254 | 2,250 | 2254:2250 | ['Celeb-DF', 'FF++', 'Final_Dataset'] | 2,141 |
| **Test (In-Dist)** | `frozen_test.csv` | 4,004 | 2,004 | 2,000 | 2004:2000 | ['Celeb-DF', 'FF++', 'Final_Dataset'] | 1,641 |
| **OOD (Held-Out)** | `frozen_ood.csv` | 5,000 | 2,500 | 2,500 | 1.00 : 1.00 | ['RVF5K'] | 5,000 |

## 2. Leakage Verification Results

- **TRAIN INTERSECT VALIDATION Filepaths**: `0` (Expected: 0)
- **TRAIN INTERSECT TEST Filepaths**: `0` (Expected: 0)
- **VALIDATION INTERSECT TEST Filepaths**: `0` (Expected: 0)
- **TRAIN INTERSECT OOD Filepaths**: `0` (Expected: 0)
- **TRAIN Identities/Groups INTERSECT VAL Groups**: `0` (Expected: 0)
- **TRAIN Identities/Groups INTERSECT TEST Groups**: `0` (Expected: 0)
- **TRAIN Sources INTERSECT OOD Sources**: `[]` (Expected: empty)

## 3. Data Integrity & Quality
- Missing files: 0 across all splits.
- Corrupted images in random sample: 0.
- Dimension profile:
  - FF++ frames: 640×480, 1280×720, 1920×1080
  - Celeb-DF v2: 256×256 centered face crops
  - Final Dataset: 256×256 face crops
  - RVF5K (OOD): 256×256 face portraits (FFHQ / StyleGAN)

## 4. Phase 2 Confirmation: Frozen Test Set
The test set `frozen_test.csv` (4,004 images) and external OOD dataset `frozen_ood.csv` (5,000 images) are permanently locked and segregated.
They will NOT be used for:
- Training
- Augmentation selection
- Threshold tuning
- Hyperparameter tuning
- Model selection
Validation set `frozen_val.csv` (4,504 images) is exclusively used for model selection and threshold determination.

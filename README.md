# Audio-based Gender Classification with Uncertainty Handling

## Overview
This project builds a gender classification system using speech audio features.
Rather than forcing a binary decision for every input, the model outputs
**probabilistic predictions** and supports an **"uncertain"** category when
confidence is low.  
This design reflects real-world audio scenarios where domain mismatch and
ambiguous samples are common.

---

## Project Pipeline

1. **Data Cleaning**
   - Filter audio samples with valid gender labels
2. **Feature Extraction**
   - MFCCs and spectral statistics extracted from audio
3. **Dataset Construction**
   - All features aggregated into a single CSV file
4. **Feature Selection**
   - Correlation-based feature filtering
5. **Model Training**
   - StandardScaler + SMOTE
   - RandomForest and XGBoost comparison
6. **Evaluation & Uncertainty Analysis**
   - Validation/test-based uncertainty ratio analysis
   - Real audio inference with uncertainty-aware output

---

## File Description

### `clean_mp3.py`
Cleans the raw dataset by **retaining only audio samples with valid gender labels**.
This step ensures that noisy or unlabeled data does not propagate into the dataset.

---

### `build_dataset.py`
Extracts audio features from the cleaned dataset and constructs a unified CSV file.

**Extracted features include:**
- Duration
- Zero-crossing rate (mean, std)
- Spectral centroid (mean, std)
- Spectral bandwidth (mean, std)
- Spectral rolloff (mean, std)
- MFCC 1–13 (mean)

The output is saved as `voice_df.csv`.

---

### `var_choose.py`
Analyzes feature correlations and selects **highly relevant features** for model
training.  
This step reduces redundancy and improves model stability.

---

### `gender_classifier.py`
Initial experiment where **class weighting** was applied to address class imbalance.
Only scaling was applied to the dataset at this stage (before professor feedback).

---

### `scaler_smote.py`
Main training and comparison script.

**Key responsibilities:**
- Train / validation / test split
- Apply `StandardScaler` (fit on train only)
- Apply `SMOTE` (train set only)
- Train and compare:
  - RandomForest
  - XGBoost
- Analyze:
  - Scaling before vs after
  - SMOTE before vs after
  - Validation performance

Validation data is explicitly separated and not used for training.

> Note: In some experiments, RandomForest achieved a higher ROC-AUC than XGBoost.
This is attributed to RandomForest’s robustness on tabular features, while XGBoost
was later preferred for probability-based uncertainty analysis due to better
calibration.

---

### `testing_gender.py`
Inference and uncertainty analysis script.

**Features:**
- Loads trained model, scaler, and selected feature columns
- Extracts features from new audio
- Outputs:
  - `male`
  - `female`
  - `uncertain` (for low-confidence cases)

**Uncertainty analysis result (test set):**
[INFO] 학습에 사용된 feature 개수: 9
Uncertain ratio (test): 0.074
Uncertain samples: 42 / 565

[INFO] 학습에 사용된 feature 개수: 9
Uncertain ratio (test): 0.074
Uncertain samples: 42 / 565


This indicates that the model abstains from prediction on approximately **7.4%**
of ambiguous samples, which is considered a healthy range for uncertainty-aware
classification.

---

## Dataset Structure
data/
├─ raw_5.wav
├─ raw_6.wav
├─ raw_7.wav
├─ raw_8.wav
├─ X_test.csv
├─ X_val.csv
├─ y_test.csv
└─ y_val.csv


- Test and validation sets are **stored explicitly** to ensure reproducibility.
- Scaling is **never applied before saving** these datasets.

---

## Models

Stored in `models/`:
- `smote_rf.pkl`
- `smote_xgb.pkl`
- `smote_scaler.pkl`
- `feature_cols.pkl`

All inference uses the same feature order and scaler as training.

---

## Uncertainty Handling

Instead of forcing binary classification, the model defines an uncertainty region:

- **Male:** probability ≥ 0.75
- **Female:** probability ≤ 0.25
- **Uncertain:** otherwise

This approach prevents overconfident predictions on ambiguous or out-of-distribution
audio samples.

---

## How to Run

```bash
# 1. Build dataset
python build_dataset.py

# 2. Train models and compare
python scaler_smote.py

# 3. Run inference and uncertainty analysis
python testing_gender.py


[Notes]
SMOTE is applied only to the training set
Validation and test sets are never used for threshold tuning
RandomForest is used for baseline comparison
XGBoost is preferred for probability-based uncertainty analysis

[Conclusion]
This project emphasizes not only classification accuracy but also model
confidence and reliability.
By explicitly handling uncertainty, the system better reflects real-world audio
classification challenges.
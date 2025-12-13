# testing_gender.py

import librosa
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


model = joblib.load("models/smote_xgb.pkl")
scaler = joblib.load("models/smote_scaler.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

def extract_features_full(path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(path, sr=sr, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = mfcc.mean(axis=1)

    feats = {
        "duration": float(duration),
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
        "centroid_mean": float(centroid.mean()),
        "centroid_std": float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "bandwidth_std": float(bandwidth.std()),
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_std": float(rolloff.std()),
    }

    for i, v in enumerate(mfcc_means, start=1):
        feats[f"mfcc{i}_mean"] = float(v)

    return feats

def predict_gender(file_path):
    file_path = Path(file_path)
    print(f"\n🎧 file: {file_path}")

    feats = extract_features_full(str(file_path))
    feat_df = pd.DataFrame([feats])
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X_new = feat_df[feature_cols]
    X_new_scaled = scaler.transform(X_new)
    proba_male = model.predict_proba(X_new_scaled)[0, 1]  # male 확률

    if proba_male > 0.75:
        gender_str = "male"
        pred_label = 1
    elif proba_male < 0.25:
        gender_str = "female"
        pred_label = 0
    else:
        gender_str = "uncertain"
        pred_label = None

    print(f"Male probability: {proba_male:.3f}")
    print(f"Predicted gender: {gender_str}")

# if __name__ == "__main__":
#     file_path = "./data/raw_8.wav"  # 네가 녹음한 파일 경로
#     predict_gender(file_path)

X_val = pd.read_csv("data/X_val.csv")
y_val = pd.read_csv("data/y_val.csv")
proba = model.predict_proba(X_val)[:, 1] 


uncertain_mask = (proba > 0.45) & (proba < 0.55)
uncertain_ratio = uncertain_mask.mean()

print(f"Uncertain ratio (test): {uncertain_ratio:.3f}")
print(f"Uncertain samples: {uncertain_mask.sum()} / {len(proba)}")

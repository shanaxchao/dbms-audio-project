# testing_gender.py

import librosa
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ===== 1) 모델 / 스케일러 / feature_cols 로드 =====
model = joblib.load("models/gender_classifier.pkl")
scaler = joblib.load("models/gender_scaler.pkl")
feature_cols = joblib.load("models/gender_feature_cols.pkl")

print(f"[INFO] 학습에 사용된 feature 개수: {len(feature_cols)}")

# ===== 2) build_dataset.py와 동일한 feature 추출 함수 =====
def extract_features_full(path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(path, sr=sr, mono=True)

    # duration
    duration = librosa.get_duration(y=y, sr=sr)

    # 기본 스펙트럼 특징
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # MFCC (첫 13개 평균)
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

    # mfcc1_mean ~ mfcc13_mean
    for i, v in enumerate(mfcc_means, start=1):
        feats[f"mfcc{i}_mean"] = float(v)

    return feats


# ===== 3) 예측 함수 =====
def predict_gender(file_path):
    file_path = Path(file_path)
    print(f"\n🎧 file: {file_path}")

    # (1) feature dict 생성 (build_dataset와 동일)
    feats = extract_features_full(str(file_path))

    # (2) DataFrame으로 만들고, 학습 때 feature_cols 순서 그대로 맞추기
    feat_df = pd.DataFrame([feats])

    # 혹시 빠진 컬럼 있으면 채워주기 (NaN -> 0)
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X_new = feat_df[feature_cols]

    # (3) 스케일링
    X_new_scaled = scaler.transform(X_new)

    # (4) 예측
    predicted_label = int(model.predict(X_new_scaled)[0])
    probs = model.predict_proba(X_new_scaled)[0]

    label_map = {0: "female", 1: "male"}

    print("predicted gender:", label_map.get(predicted_label, predicted_label))
    print(f"confidence: {probs[predicted_label]:.4f}")

    return predicted_label, probs


if __name__ == "__main__":
    file_path = "./data/raw_3.wav"  # 네가 녹음한 파일 경로
    predict_gender(file_path)

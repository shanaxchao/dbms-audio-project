# build_dataset.py

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# ===== 1) 경로 설정 (VS Code relative path) =====
BASE_DIR = Path(__file__).resolve().parent

# meta.tsv가 build_dataset.py 옆에 있다고 가정
META_PATH = BASE_DIR / "data" / "meta.tsv"

# 오디오 파일 폴더
AUDIO_DIR = BASE_DIR / "data" / "korean_2"


# ===== 2) 라벨 매핑 설정 =====
GENDER_MAP = {
    "female_feminine": 0,
    "male_masculine": 1,
}

AGE_MAP = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
}


# ===== 3) feature 추출 =====
def extract_features(path, sr=22050):
    y, sr = librosa.load(path, sr=sr)

    duration = librosa.get_duration(y=y, sr=sr)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
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


# ===== 4) meta 읽기 =====
def load_meta(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path, sep="\t")  # path, age, gender 헤더라고 가정

    # ✅ 문자열 컬럼 공백 제거 – .str.strip() 사용해야 함
    df["path"] = df["path"].astype(str).str.strip()
    df["age"] = df["age"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip()

    return df



def main():
    # meta.tsv 불러오기
    meta_df = load_meta(META_PATH)
    rows = []

    print("총 샘플 수:", len(meta_df))

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        filename = row["path"]         # ex) common_voice_ko_12345.mp3
        age_str = row["age"]
        gender_str = row["gender"]

        audio_path = AUDIO_DIR / filename

        if not audio_path.exists():
            print(f"[WARN] 파일 없음: {audio_path}")
            continue

        try:
            feats = extract_features(str(audio_path))
        except Exception as e:
            print(f"[ERROR] {audio_path} 처리 실패: {e}")
            continue

        # meta 기반 라벨 추가
        feats.update({
            "filename": filename,
            "age": age_str,
            "gender": gender_str,
            "age_label": AGE_MAP.get(age_str, -1),
            "gender_label": GENDER_MAP.get(gender_str, -1),
        })

        rows.append(feats)

    df = pd.DataFrame(rows)

    # 상대경로로 결과 저장
    out_path = BASE_DIR / "voice_df.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("저장 완료 ->", out_path)
    print("shape:", df.shape)


if __name__ == "__main__":
    main()

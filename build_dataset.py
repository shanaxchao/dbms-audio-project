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
AUDIO_DIR2 = BASE_DIR / "data" / "english"

# 영어 화자 이름 (파일명 앞부분과 일치한다고 가정: bea_001.wav -> "bea")
male_speakers_en = ["bea", "jenie"]
female_speakers_en = ["josh", "sam"]


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

    # 문자열 컬럼 공백 제거
    df["path"] = df["path"].astype(str).str.strip()
    df["age"] = df["age"].astype(str).str.strip()
    df["gender"] = df["gender"].astype(str).str.strip()

    return df


def main():
    rows = []

    # ---------- 4-1) 한국어 meta 기반 ----------
    meta_df = load_meta(META_PATH)
    print("총 한국어 샘플 수:", len(meta_df))

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Korean"):
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
            "language": "ko",
        })

        rows.append(feats)

    # ---------- 4-2) 영어 폴더 전체 스캔 ----------
    en_files = list(AUDIO_DIR2.rglob("*.wav")) + list(AUDIO_DIR2.rglob("*.mp3"))
    print("총 영어 샘플 수 (파일 기준):", len(en_files))

    for audio_path in tqdm(en_files, desc="English"):
        filename = audio_path.name

        # 화자 이름: 파일명에서 맨 앞 토큰 사용 (예: bea_001.wav -> 'bea')
        speaker = audio_path.stem.split("(")[0]

        # 영어 성별을 한국어 라벨 체계에 맞춰 설정
        if speaker in male_speakers_en:
            gender_str = "male_masculine"
        elif speaker in female_speakers_en:
            gender_str = "female_feminine"
        else:
            print(f"[WARN] 성별 미지정 화자, 스킵: {filename} (speaker={speaker})")
            continue

        age_str = "thirties"  # 영어는 전부 30대

        try:
            feats = extract_features(str(audio_path))
        except Exception as e:
            print(f"[ERROR] {audio_path} 처리 실패: {e}")
            continue

        feats.update({
            "filename": filename,
            "age": age_str,
            "gender": gender_str,  # ← 한국어와 동일한 문자열
            "age_label": AGE_MAP.get(age_str, -1),
            "gender_label": GENDER_MAP.get(gender_str, -1),
            "language": "en",
        })

        rows.append(feats)

    # ---------- 4-3) raw 데이터 추가 ----------
    # ---------- 4-3) raw 데이터 추가 ----------
    RAW_DIR = BASE_DIR / "data"

    raw_gender_map = {
        "raw_1.wav": ("female_feminine", 0),
        "raw_2.wav": ("female_feminine", 0),
        "raw_3.wav": ("female_feminine", 0),
        "raw_4.wav": ("female_feminine", 0),
        "raw_5.wav": ("male_masculine", 1),
    }

    for fname, (gender_str, gender_label) in raw_gender_map.items():
        raw_path = RAW_DIR / fname

        if not raw_path.exists():
            print(f"[WARN] raw 파일 없음: {raw_path}")
            continue

        try:
            feats = extract_features(str(raw_path))
        except Exception as e:
            print(f"[ERROR] {raw_path} 처리 실패: {e}")
            continue

        feats.update({
            "filename": fname,
            "age": "thirties",                    # raw는 나이 정보 없으므로 임의로 통일
            "gender": gender_str,
            "age_label": AGE_MAP.get("thirties", 2),
            "gender_label": gender_label,
            "language": "raw",                    # raw 표시
        })

        rows.append(feats)

    print("[INFO] raw 파일 추가 완료:", list(raw_gender_map.keys()))



    # ---------- 5) DataFrame 생성 및 저장 ----------
    df = pd.DataFrame(rows)

    out_path = BASE_DIR / "voice_df.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("저장 완료 ->", out_path)
    print("shape:", df.shape)


if __name__ == "__main__":
    main()

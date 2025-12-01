# train_gender_xgb.py

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# ===== 1. 경로 설정 =====

# ===== 1) 경로 설정 (VS Code relative path) =====
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "voice_df.csv"
MODEL_PATH = BASE_DIR / "gender_xgb_model.pkl"
SCALER_PATH = BASE_DIR / "gender_scaler.pkl"
FEATURE_COLS_PATH = BASE_DIR / "gender_feature_cols.pkl"

# ===== 2. 데이터 불러오기 =====
df = pd.read_csv(CSV_PATH)

# ===== 3. feature / label 분리 =====
# gender_label: 0=female, 1=male
y = df["gender_label"]

# 모델에 넣을 feature만 선택 (라벨/문자열 정보 제외)
drop_cols = ["gender_label", "gender", "age", "filename", "path"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]

print("사용 feature 개수:", len(feature_cols))
print("feature 예시:", feature_cols[:10])

# ===== 4. train / test 분리 =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===== 5. 스케일링 =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 6. XGBoost 모델 정의 & 학습 =====
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    tree_method="hist"  # 속도 빠른 옵션
)

print("모델 학습 중...")
model.fit(X_train_scaled, y_train)

# ===== 7. 평가 =====
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("\n🎯 정확도:", acc)
print("\n📌 분류 리포트:\n", classification_report(y_test, y_pred))

# ===== 8. 모델/스케일러/feature 목록 저장 =====
# 모델 저장 폴더 만들기
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/gender_xgb_model.pkl")
joblib.dump(scaler, "models/gender_scaler.pkl")
joblib.dump(feature_cols, "models/gender_feature_cols.pkl")

print("모델 저장 완료 → models/")


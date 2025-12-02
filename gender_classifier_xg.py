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
feature_cols = ["mfcc8_mean", "mfcc13_mean", "mfcc12_mean", 
 "mfcc2_mean", "mfcc11_mean", "mfcc10_mean", 
 "mfcc6_mean", "mfcc4_mean", "duration"]

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

from collections import Counter

cnt = Counter(y_train)
# y_train: 0=female, 1=male
# → female(0) 쪽에 weight를 줘서 female misclassification을 더 큰 패널티로 만듦
w_female = cnt[1] / cnt[0]    # 남자/여자 비율로 weight 계산

sample_weight = y_train.astype(float).copy()
sample_weight[y_train == 0] = w_female  # female 가중치 ↑
sample_weight[y_train == 1] = 1.0

print(f"여자 클래스 가중치 적용됨: w_female={w_female:.3f}")

# ===== 6-2. 학습 =====
print("모델 학습 중...")
model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

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



from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred, target_names=["female(0)", "male(1)"]))
print(confusion_matrix(y_test, y_pred))

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "voice_df.csv"

df = pd.read_csv(CSV_PATH)

X = df[["mfcc8_mean", "mfcc13_mean", "mfcc12_mean", 
 "mfcc2_mean", "mfcc11_mean", "mfcc10_mean", 
 "mfcc6_mean", "mfcc4_mean", "duration"]]
y = df['gender_label']
feature_cols =  X.columns.tolist()

from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size = 0.2)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size = 0.25,
    stratify = y_train_full,
    random_state = 22
)

#print("Before SMOTE:", y_train.value_counts())

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

# print("Train mean (before scaling):", X_train.mean(axis=0))
# print("Train std (before scaling):", X_train.std(axis=0))
# print("Train mean (after scaling):", X_train_scaled.mean(axis=0))
# print("Train std  (after scaling):", X_train_scaled.std(axis=0))

from imblearn.over_sampling import SMOTE
X_train_res, y_train_res = SMOTE(random_state = 22).fit_resample(X_train_scaled, y_train)

#print("After SMOTE:", y_train_res.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

rf = RandomForestClassifier(n_estimators = 200, random_state = 22)
rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:,1]

print("---RandomForest---")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators = 300,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8, 
    eval_metric = "logloss",
    random_state = 22
) 
xgb.fit(X_train_res, y_train_res)

y_pred_xgb = xgb.predict(X_test_scaled)
y_prob_xgb = xgb.predict_proba(X_test_scaled)[:,1]

print("=== XGBoost ===")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

import joblib
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(rf, MODEL_DIR / "smote_rf.pkl")
joblib.dump(xgb, MODEL_DIR / "smote_xgb.pkl")
joblib.dump(std_scaler, MODEL_DIR / "smote_scaler.pkl")
joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

X_val.to_csv("X_val.csv", index = False)
y_val.to_csv("y_val.csv", index = False)


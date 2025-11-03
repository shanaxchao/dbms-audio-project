from glob import glob
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_speakers_en = ["bea", "josh", "sam"]
test_speakers_en = ["jenie"]

english_dir = "./data/english"
korean_dir = "./data/korean"

def extract_features(file):
    y_audio, sr = librosa.load(file, sr = 16000, mono = True)
    mfccs = librosa.feature.mfcc(y = y_audio, sr = sr, n_mfcc = 13)
    mfccs_mean = np.mean(mfccs, axis = 1)
    mfccs_std = np.std(mfccs, axis = 1)
    return np.concatenate((mfccs_mean, mfccs_std)).astype(np.float32)

#english
english_files = glob(f"{english_dir}/*.wav")
eng_train, eng_test = [], []

for f in english_files:
    name = Path(f).stem.lower()
    if any(name.startswith(spk.lower()) for spk in train_speakers_en):
        eng_train.append(f)
    else:
        eng_test.append(f)


#korean
korean_files = glob(f"{korean_dir}/*.wav")
kor_train = [f for f in korean_files if Path(f).stem.lower().startswith("a_")]
kor_test = [f for f in korean_files if Path(f).stem.lower().startswith("b_")]

#print(len(eng_test), len(kor_test))

#data split
X_train, y_train, X_test, y_test = [], [], [], []
for f in eng_train:
    X_train.append(extract_features(f))
    y_train.append(0) # 0 == eng
for f in eng_test:
    X_test.append(extract_features(f))
    y_test.append(0)

for f in kor_train:
    X_train.append(extract_features(f))
    y_train.append(1) # 1 == kor
for f in kor_test:
    X_test.append(extract_features(f))
    y_test.append(1)

# train XGBoost
model = xgb.XGBClassifier(
    n_estimators = 300, 
    learning_rate = 0.1, 
    max_depth = 6, 
    subsample = 0.9,
    colsample_bytree = 0.9,
    eval_metric = 'logloss',
    random_state = 42
)

model.fit(X_train, y_train)

# predict labels for test set
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# detailed report
print(classification_report(y_test, y_pred, 
                            labels = [0, 1],
                            target_names = ["english", "korean"], 
                            zero_division = 0))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# save model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/language_classifier.pkl")
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

# gather all audio files
audio_files = glob("./data/*/*.wav")

X = []  # features
y = []  # labels

for file in audio_files:
    path = Path(file)
    label = path.parent.name  # get folder name, e.g. 'korean'
    y.append(label)
    
    # load audio
    y_audio, sr = librosa.load(file, sr=16000, mono=True)
    
    # extract features, e.g., MFCCs and average over time
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # shape (13,)
    mfccs_std = np.std(mfccs, axis = 1)

    features = np.concatenate((mfccs_mean, mfccs_std))

    X.append(features)

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(y)

# split into training and testing sets
for seed in [1, 2, 3, 4, 5]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = seed, stratify = y
    )

# train random forest
#model = RandomForestClassifier(n_estimators=100, random_state=40)
model = xgb.XGBClassifier(
    n_estimators = 300, 
    learning_rate = 0.3, 
    max_depth = 6, 
    subsample = 1,
    eval_metric = 'logloss'
)

model.fit(X_train, y_train)

# predict labels for test set
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# detailed report
print(classification_report(y_test, y_pred, target_names = le.classes_))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

# save model
Path("models").mkdir(exist_ok=True)
joblib.dump({"model":model, "label_encoder":le}, "models/language_classifier.pkl")
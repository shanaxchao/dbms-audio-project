from glob import glob
import librosa
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

audio_files = glob("./data/*/*.wav")

X = []  # features
y = []  # labels

for file in audio_files:
    # extract label from folder name
    # example: './data/korean/kor_0001.wav' → 'korean'
    label = file.split("/")[2]  # index 2 is the folder name
    y.append(label)
    
    # load audio
    y_audio, sr = librosa.load(file, sr=None)
    
    # extract features, e.g., MFCCs and average over time
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # shape (13,)
    
    X.append(mfccs_mean)

X = np.array(X)
y = np.array(y)


# split into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train random forest
model = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees
model.fit(X_train, y_train)

# predict labels for test set
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# detailed report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/language_classifier.pkl")
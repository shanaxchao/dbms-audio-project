#same model
#audio file of us 가져와야됨
#모델에 넣어서 성능확인

import librosa
import numpy as np
import joblib

model = joblib.load("models/language_classifier.pkl")

file_path = "./data/raw_3.wav"
y_audio, sr = librosa.load(file_path, sr=16000, mono=True)

mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc = 13)
mfccs_mean = np.mean(mfccs, axis=1)
mfccs_std = np.std(mfccs, axis=1)
features = np.concatenate((mfccs_mean, mfccs_std))

X_new = features.reshape(1, -1)

predicted_label = model.predict(X_new)[0]
label = {0 : "english", 1 : "korean"}

print("predicted language:", label.get(predicted_label, predicted_label))
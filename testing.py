#same model
#audio file of us 가져와야됨
#모델에 넣어서 성능확인

import librosa
import numpy as np
import joblib
model = joblib.load("models/language_classifier.pkl")

file_path = "./data/korean/kor(1).wav"
y_audio, sr = librosa.load(file_path, sr=16000, mono=True)

mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc = 13)
mfccs_mean = np.mean(mfccs, axis=1)

X_new = mfccs_mean.reshape(1, -1)

predicted_label = model.predict(X_new)
print("predicted language : ", predicted_label[0])
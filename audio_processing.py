# Practice Audio Data Processing via Python
# Following a YouTube tutorial: https://www.youtube.com/watch?v=ZqpSb5p1xQo
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import sounddevice as sd

from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# list of audio files
audio_files = glob("./data/*/*.wav")

y, sr = librosa.load(audio_files[0])

# play the first audio file
# sd.play(y, sr)
# sd.wait()

# plot raw waveform
# pd.Series(y).plot(figsize=(10, 5), 
#                   lw = 1, 
#                   title = "Raw Audio Example",
#                   color = color_pal[0]) # easier to plot with pandas
# plt.show()

# plot trimmed waveform
# y_trimmed, _ = librosa.effects.trim(y)
# pd.Series(y).plot(figsize=(10, 5), 
#                   lw = 1, 
#                   title = "Trimmed Audio Example",
#                   color = color_pal[1]) # easier to plot with pandas
# plt.show()

# plot zoomed in waveform
# pd.Series(y[30000:30500]).plot(figsize=(10, 5), 
#                   lw = 1, 
#                   title = "Raw Audio Zoomed In Example",
#                   color = color_pal[2]) # easier to plot with pandas
# plt.show()

# see the different frequencies (fourier transform)
# D = librosa.stft(y) # short-time fourier transform
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # convert to decibels

# # plot spectogram
# fig, ax = plt.subplots(figsize=(10, 5))
# img = librosa.display.specshow(S_db, 
#                                x_axis='time', 
#                                y_axis='log',
#                                ax=ax)
# ax.set_title("Spectogram Example",
#              fontsize=20)
# fig.colorbar(img, ax=ax, format=f"%0.2f")
# plt.show()

# plot mel spectogram (mel = melodic)
S = librosa.feature.melspectrogram(y, 
                               sr=sr, 
                               n_mels=256)
S_db_mel = librosa.amplitude_to_db(np.abs(S), ref=np.max) # convert to decibels

fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db_mel, 
                               x_axis='time', 
                               y_axis='log',
                               ax=ax)
ax.set_title("Mel Spectogram Example",
             fontsize=20)
fig.colorbar(img, ax=ax, format=f"%0.2f")
plt.show()


import pandas as pd

df = pd.read_csv("voice_df.csv")

# gender_label이 0/1 형태라고 가정
y = df["gender_label"]

# 숫자형 feature만 선택
numeric_df = df.select_dtypes(include=["float", "int"]).drop(columns=["gender_label"])
corr = numeric_df.apply(lambda col: col.corr(y))
corr = corr.sort_values(ascending=False)
print(corr)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
corr.plot(kind="bar")
plt.title("Correlation with Gender (female=0, male=1)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(corr.to_frame().T, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap: Feature vs Gender")
plt.yticks([])
plt.tight_layout()
plt.show()

topN = corr.abs().sort_values(ascending=False).head(10)
topN.plot(kind="barh", figsize=(8,6))
plt.title("Top 10 Gender-Correlated Features")
plt.xlabel("Correlation")
plt.gca().invert_yaxis()
plt.show()

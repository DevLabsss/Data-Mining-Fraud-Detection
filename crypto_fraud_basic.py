# crypto_fraud_basic.py (fixed)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs_basic")
OUT.mkdir(exist_ok=True)

# 1) Generate data sintetis (dibesarkan & rule dilonggarkan agar fraud muncul)  # <--
np.random.seed(42)
n_samples = 5000                           # dari 2000 -> 5000               # <--
amount = np.random.exponential(scale=100, size=n_samples)
freq = np.random.poisson(lam=3, size=n_samples)
age = np.random.randint(1, 1000, n_samples)
is_weekend = np.random.choice([0, 1], size=n_samples)

# Label fraud (rule dilonggarkan supaya proporsinya tidak nol di test)          # <--
labels = ((amount > 120) & (age < 100) & (freq > 5)).astype(int)

df = pd.DataFrame({
    "amount": amount,
    "transaction_freq_24h": freq,
    "account_age_days": age,
    "is_weekend": is_weekend,
    "is_fraud": labels
})
df.to_csv(OUT / "fraud_dataset.csv", index=False)
print(f"✅ Dataset tersimpan: {OUT/'fraud_dataset.csv'} (rows={len(df)})")
print("Class balance:", dict(zip(*np.unique(labels, return_counts=True))))      # <--

# 2) Split data (stratify supaya komposisi kelas terjaga)
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 4) Evaluasi (pakai labels=[0,1] supaya report selalu 2 kelas)                 # <--
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n📊 Hasil Evaluasi")
print("Akurasi:", acc)

cm = confusion_matrix(y_test, y_pred, labels=[0,1])                             # <--
print("Confusion Matrix:\n", cm)

rep = classification_report(
    y_test, y_pred,
    labels=[0,1],                                                                # <--
    target_names=["Normal","Fraud"],                                             # <--
    zero_division=0                                                               # <--
)
print("Classification Report:\n", rep)

# 5) Simpan hasil (buat bahan PPT)
pd.DataFrame(
    cm,
    index=["Actual_Normal","Actual_Fraud"],
    columns=["Pred_Normal","Pred_Fraud"]
).to_csv(OUT/"confusion_matrix.csv", index=True)

with open(OUT/"classification_report.txt","w") as f:
    f.write(rep)
with open(OUT/"metrics.txt","w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")

# Simpan gambar confusion matrix
plt.figure(figsize=(4.5,4))
plt.imshow(cm)           # warna default (sesuai aturan, tidak set colormap)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Pred Normal","Pred Fraud"])
plt.yticks([0,1], ["Actual Normal","Actual Fraud"])
for (i,j), val in np.ndenumerate(cm):
    plt.text(j, i, int(val), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT/"confusion_matrix.png", dpi=200)
plt.close()

print(f"\n📁 Output disimpan di folder: {OUT.resolve()}")
print(" - fraud_dataset.csv")
print(" - confusion_matrix.csv / confusion_matrix.png")
print(" - classification_report.txt")
print(" - metrics.txt")

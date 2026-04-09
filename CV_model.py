"""
Train an MLP classifier on MediaPipe hand landmarks to recognize ASL fingerspelling.

Input:  landmarks.csv  — rows of (label, x0, y0, z0, ..., x20, y20, z20)
Output: model.pkl      — trained sklearn MLPClassifier, ready for inference.py

Architecture: 63 → 128 → 64 → 29 classes
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

LANDMARKS_CSV = "./landmarks.csv"
MODEL_OUT     = "./model.pkl"
ENCODER_OUT   = "./label_encoder.pkl"
TEST_SIZE     = 0.15
RANDOM_STATE  = 42

# --- Load data ---
print("Loading landmarks.csv...")
df = pd.read_csv(LANDMARKS_CSV)
print(f"  {len(df)} rows, {df['label'].nunique()} classes: {sorted(df['label'].unique())}")

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# Augment with mirrored (left-hand) versions of every sample by negating x coordinates.
# x columns are at indices 0, 3, 6, ..., 60 (every 3rd starting at 0).
X_mirrored = X.copy()
X_mirrored[:, 0::3] *= -1
X = np.concatenate([X, X_mirrored], axis=0)
y = np.concatenate([y, y], axis=0)
print(f"  After mirroring augmentation: {len(X)} total samples")

# --- Encode labels to integers ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# --- Train/val split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"\nTrain: {len(X_train)} samples | Val: {len(X_val)} samples")

# --- Train MLP ---
print("\nTraining MLP...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    max_iter=500,
    random_state=RANDOM_STATE,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nValidation accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nPer-class report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# --- Confusion matrix ---
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.title("Confusion Matrix — Validation Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to confusion_matrix.png")

# --- Save model and encoder ---
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
with open(ENCODER_OUT, "wb") as f:
    pickle.dump(le, f)
print(f"Model saved to {MODEL_OUT}")
print(f"Label encoder saved to {ENCODER_OUT}")

import os

# Set environment variables before importing TabPFN / PyTorch-backed code
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

# Load dataset
cols = [f"sensor_{i}" for i in range(1, 25)] + ["target"]
df = pd.read_csv("sensor_readings_24.data", header=None, names=cols)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:")
print(df["target"].value_counts())

# Split features / target
X = df.drop(columns=["target"])
y = df["target"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Reduce training size for Mac MPS memory stability
# You can increase this later if it runs fine
max_train_size = 1000
if len(X_train) > max_train_size:
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_train_size,
        random_state=42,
        stratify=y_train
    )
else:
    X_train_small, y_train_small = X_train, y_train

print("\nTrain shape:", X_train_small.shape)
print("Test shape:", X_test.shape)

# Train model
clf = TabPFNClassifier()
clf.fit(X_train_small, y_train_small)

# Predict in very small batches to avoid MPS out-of-memory
batch_size = 10
predictions = []

for i in range(0, len(X_test), batch_size):
    batch = X_test.iloc[i:i + batch_size]
    batch_pred = clf.predict(batch)
    predictions.extend(batch_pred)
    print(f"Predicted rows {i} to {min(i + batch_size, len(X_test)) - 1}")

predictions = np.array(predictions)

# Evaluate
acc = accuracy_score(y_test, predictions)
print("\nAccuracy:", acc)
print("Sample predictions:", predictions[:10])
print("Sample actuals:    ", y_test.iloc[:10].to_numpy())
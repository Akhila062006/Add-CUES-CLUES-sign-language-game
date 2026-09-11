import os
import csv
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# Dataset folder
DATASET_FOLDER = "dataset"

# Model file
MODEL_FILE = "knn_model.pkl"

# Gestures
GESTURES = ["YES", "WATER", "HELP"]


# Load dataset
X = []
y = []

print("Loading dataset...")

for gesture in GESTURES:

    for filename in os.listdir(DATASET_FOLDER):

        if filename.startswith(gesture + "_") and filename.endswith(".csv"):

            filepath = os.path.join(DATASET_FOLDER, filename)

            with open(filepath, "r") as file:

                reader = csv.reader(file)

                for row in reader:

                    if row:
                        landmarks = [float(value) for value in row]

                        X.append(landmarks)
                        y.append(gesture)


print(f"Total samples loaded: {len(X)}")


# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create KNN model
print("Training KNN model...")

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(X_train, y_train)


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Save model
joblib.dump(model, MODEL_FILE)

print(f"Model saved as: {MODEL_FILE}")

print("Training completed successfully!")
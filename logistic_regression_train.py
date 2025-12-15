import os
import cv2
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Training_logistic_regression")
SAVE_PATH =os.path.join(BASE_DIR, "Model.pkl")

CLASSES = ["Banana Lady Finger 1", "Banana 1", "Apple Red 1", "Apple Red 2", "Strawberry 1"]

warnings.filterwarnings("ignore")

print("Loading images...")
images, labels = [], []

for label_id, fruit_name in enumerate(CLASSES):
    folder = os.path.join(TRAIN_DIR, fruit_name)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img.flatten() / 255.0)
            labels.append(label_id)

X = np.array(images)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

print("\nTraining model...")
model = LogisticRegression(max_iter=25, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)

print(f"Training Accuracy: {train_acc:.2%}")
print(f"Validation Accuracy: {val_acc:.2%}")

iterations = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20,25]
train_scores = []
val_scores = []

for max_iter in iterations:
    temp_model = LogisticRegression(max_iter=max_iter, random_state=42)
    temp_model.fit(X_train, y_train)
    train_scores.append(temp_model.score(X_train, y_train))
    val_scores.append(temp_model.score(X_val, y_val))

plt.figure(figsize=(7, 5))
plt.plot(iterations, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(iterations, val_scores, 's-', label='Validation Accuracy', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# --- Save Model ---
joblib.dump(model, SAVE_PATH)
print(f"\nModel saved to: {SAVE_PATH}")
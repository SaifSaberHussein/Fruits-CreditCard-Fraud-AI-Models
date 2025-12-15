import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model.pkl")
TEST_DIR = os.path.join(BASE_DIR, "Testing_logistic_regression")

CLASSES = ["Banana Lady Finger 1", "Banana 1", "Apple Red 1", "Apple Red 2", "Strawberry 1"]

print("Loading model...")
model = joblib.load(MODEL_PATH)

# --- Load Test Data ---
print("Loading test images...")
X_test, y_test = [], []

for label_id, fruit_name in enumerate(CLASSES):
    folder = os.path.join(TEST_DIR, fruit_name)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            X_test.append(img.flatten() / 255.0)
            y_test.append(label_id)

X_test = np.array(X_test)
y_test = np.array(y_test)

# --- Make Predictions ---
print("Making predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# --- 1. Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# --- 2. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

# --- 3. ROC Curve ---
y_test_bin = label_binarize(y_test, classes=range(len(CLASSES)))

plt.figure(figsize=(10, 8))
for i in range(len(CLASSES)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{CLASSES[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()

# --- 4. Test Single Random Image ---
print("\n--- Testing Random Image ---")
random_class = np.random.choice(CLASSES)
folder = os.path.join(TEST_DIR, random_class)
random_file = np.random.choice(os.listdir(folder))
img_path = os.path.join(folder, random_file)

# Load and predict
img = cv2.imread(img_path)
img_flat = img.flatten() / 255.0
prediction = model.predict([img_flat])[0]
predicted_class = CLASSES[prediction]

print(f"Actual: {random_class}")
print(f"Predicted: {predicted_class}")

plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"True: {random_class}\nPredicted: {predicted_class}")
plt.axis('off')
plt.tight_layout()
plt.show()
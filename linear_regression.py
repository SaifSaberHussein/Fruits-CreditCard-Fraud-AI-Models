import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc


data = pd.read_csv(r"F:\.Coding\Visula Studio Code\Python\Machine Learning\ML Project\creditcard_2023.csv")


X = data.drop("Class", axis=1).values
y = data["Amount"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


m, n = X_train.shape
weights = np.zeros(n)
bias = 0

learning_rate = 0.01
epochs = 500
losses = []

for _ in range(epochs):
    y_pred = np.dot(X_train, weights) + bias

    
    loss = np.mean((y_train - y_pred) ** 2)
    losses.append(loss)

    
    dw = (-2 / m) * np.dot(X_train.T, (y_train - y_pred))
    db = (-2 / m) * np.sum(y_train - y_pred)

    
    weights -= learning_rate * dw
    bias -= learning_rate * db


y_test_pred = np.dot(X_test, weights) + bias

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.show()

plt.scatter(y_test, y_test_pred, s=5)
plt.xlabel("Actual Class")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()
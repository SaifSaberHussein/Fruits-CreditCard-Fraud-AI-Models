import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r"F:\.Coding\Visula Studio Code\Python\Machine Learning\ML Project\creditcard_2023.csv")

# Drop columns not needed
data = data.drop(columns=["id", "Class"])


#Define features and target

# Features: V1 to V28 and target (amount)
X = data.drop(columns=["Amount"])
y = data["Amount"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)


# -----------------------------------------------------
# Feature scaling
# -----------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train
# -----------------------------------------------------
knn = KNeighborsRegressor(
    n_neighbors=20, #best mse is 2000 but we use small number to make graph look a bit better
    weights="distance"
)

knn.fit(X_train_scaled, y_train)


#PREDICTIOn
y_pred = knn.predict(X_test_scaled)



# calculations
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print("===== y_test amount =====")
print("Min:", y_test.min())
print("Mean:", round(y_test.mean(),2))
print("Max:", y_test.max())

print("\n===== y_pred amount statistics =====")
print("Min:", y_pred.min())
print("Mean:", round(y_pred.mean(),2))
print("Max:", y_pred.max())

print("==== mean square error(mse) and Rootmse =====")
print("Mean Squared Error (MSE)", round(mse,2))
print("RMSE:", round(rmse,2))



# 8. y_test vs y_pred PLOT (TEST DATA)
#    Sampling is used ONLY FOR PLOT
# -----------------------------------------------------
sample_size = 500  # number of test points to plot change if wanted to show more/less
sample_indices = np.random.choice(
    len(y_test),
    size=sample_size,
    replace=False )

y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]

# Scatter plot of sampled test points

#------------------------------------------ comment the above lines in the box if u want to run the entire y test vs y predict
#and the plt scatter down too 
 
 
 # Defines limites for regression line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())


plt.figure(figsize=(10, 7))
#---------------------------- comment the plt scatter to hide sample
plt.scatter(
    y_test_sample,
    y_pred_sample,
   alpha=0.3 ) #alpha to make points transparent
#-------------------------------------------------------------------



#to plot full test data not sample only (keep just in case DONT DELETE)
#-----------------------------------
#plt.scatter(
  #  y_test,       # full test data
 #   y_pred,       # predictions for full test data
#    alpha=0.3   ) 
#-----------------------------------

# diagonal line
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="-",
    linewidth=3,
    color="red"
)

plt.xlabel("Actual Amount (y_test)")
plt.ylabel("Predicted Amount (y_pred)")
plt.title("KNN Regressor: y_test vs y_pred (Test dataa)")
plt.grid(True)
plt.show()

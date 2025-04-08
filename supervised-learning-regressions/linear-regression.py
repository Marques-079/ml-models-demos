import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data-preprocessing/scaled_numerical1.csv")

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal equation: w = (X^T X)^-1 X^T y
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  
        return X_b @ self.weights

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def rmse(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        mask = y != 0
        relative_error = np.abs((y_pred[mask] - y[mask]) / y[mask])
        within_x_percent = relative_error <= 0.10  # 10% tolerance
        return np.mean(within_x_percent)


#Running model here:

features = [
    'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)'
]
target = 'Apparent Temperature (C)'


X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("R² Score:", model.score(X_test, y_test))
print("RMSE:", model.rmse(X_test, y_test))

acc = model.accuracy(X_test, y_test)
print(f"Accuracy within 5% tolerance: {acc * 100:.2f}%")




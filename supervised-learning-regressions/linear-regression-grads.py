import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class LinearRegressionGD:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):
            y_pred = X @ self.weights + self.bias

            dw = -(2 / n_samples) * (X.T @ (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)

            '''It may be easier to think y-ypred is the MSE and X.T is the transpose of the X matrix. After multiplication
            the result is similar to attention scores in transformers where the error is applied to each feature, thus 
            we can calaculate our graidents and update our weights accordingly, negative will reinforce a posiotive gradient decent
            taking the partial derivative of the loss allows to move towards local (seen) minima. (I think)
            
            By multiplying the transpose of X with the error, we are effectively distributing the error across all features, so
            if big weight made big error it will update big weight more'''

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if epoch % 50 == 0 or epoch == epochs - 1:
                mse = np.mean((y - y_pred) ** 2)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

    def predict(self, X):
        return X @ self.weights + self.bias

    def rmse(self, X, y):
        return np.sqrt(np.mean((self.predict(X) - y) ** 2))

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        mask = y != 0
        relative_error = np.abs((y_pred[mask] - y[mask]) / y[mask])
        within_x_percent = relative_error <= 0.10  # 10% tolerance
        return np.mean(within_x_percent)

df = pd.read_csv("data-preprocessing/scaled_numerical1.csv")

features = [
    'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)'
]
target = 'Apparent Temperature (C)'


X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = LinearRegressionGD()
model.fit(X_train, y_train, learning_rate=0.01, epochs=2000)


acc = model.accuracy(X_test, y_test)
print(f"Accuracy within 5% tolerance: {acc * 100:.2f}%")
print("RMSE:", model.rmse(X_test, y_test))
print("R² Score:", model.r2_score(X_test, y_test))

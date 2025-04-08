import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

class LassoRegressionGD:
    def __init__(self, alpha=0.1, learning_rate=0.01, epochs=1000):
        self.alpha = alpha  # L1 penalty strength
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y - y_pred

            # Gradients
            dw = (-2 / n_samples) * (X.T @ error)
            db = (-2 / n_samples) * np.sum(error)

            '''
            So what L1 regularization does, qualitatively, is it adds an extra force during gradient descent 
            that is constant in magnitude but opposite in direction to the sign of each weight. 
            Unlike L2, which adds a penalty proportional to the current value of the weight, 
            L1 adds a flat penalty that always pushes back with the same intensity.
            For positive weights, it subtracts a small fixed value (moves them closer to 0)
            For negative weights, it adds a small fixed value (again, toward 0)
            For weights near 0, this pull is strong enough to completely flatten them out to exactly 0
            this is applying constant force in the form of allha * sign(weights)
            
            it adds a sign multiplier and a small value proporitonal to the already existing weights. 
            dw += self.alpha * np.sign(self.weights)'''

            # Add subgradient of L1
            dw += self.alpha * np.sign(self.weights)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Optional: print loss occasionally
            if epoch % 100 == 0 or epoch == self.epochs - 1:
                loss = np.mean(error**2) + self.alpha * np.sum(np.abs(self.weights))
                print(f"Epoch {epoch} - Loss: {loss:.4f}")

    def predict(self, X):
        return X @ self.weights + self.bias

    def rmse(self, X, y):
        return np.sqrt(np.mean((y - self.predict(X)) ** 2))

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

model = LassoRegressionGD(alpha=0.1, learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)


acc = model.accuracy(X_test, y_test)
print(f"Accuracy within 5% tolerance: {acc * 100:.2f}%")
print("RMSE:", model.rmse(X_test, y_test))
print("R²:", model.r2_score(X_test, y_test))


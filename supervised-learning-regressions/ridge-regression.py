import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RidgeRegressionGD:
    def __init__(self, alpha=1.0, learning_rate=0.01, epochs=1000):
        self.alpha = alpha
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent loop
        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y - y_pred

            # Gradients
            dw = (-2 / n_samples) * (X.T @ error) + 2 * self.alpha * self.weights
            db = (-2 / n_samples) * np.sum(error)

            '''So what this does qualaitaitverly is adds an increase proportional to already present weight values 
            before the multiplication of the learning rate (prior to gradient descent). This increase in magnitude means 
            the update moves towards zero at a faster weight and if moving away from zero for whatever reason would act to 
            reduce the magnitude of the update (neg - pos = lessneg), thus pinning our weights around 0.
            
            This extra layer of regularization is what makes weights more stable by punishing big weights and prevent overfitting
            and relliance on certain features. I guess you can think about the gradient decent more like a Vase than a Bowl where
            the bottom point is the local minma.
            '''

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse:.4f}")

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

    
features = [
    'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)'
]
target = 'Apparent Temperature (C)'

df = pd.read_csv("data-preprocessing/scaled_numerical1.csv")

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RidgeRegressionGD(alpha=0.5, learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)

print("RMSE:", model.rmse(X_test, y_test))
print("R²:", model.r2_score(X_test, y_test))
acc = model.accuracy(X_test, y_test)
print(f"Accuracy within 5% tolerance: {acc * 100:.2f}%")

'''
-regression.py
Epoch 0: MSE = 237.9130
Epoch 100: MSE = 17.3821
Epoch 200: MSE = 13.3069
Epoch 300: MSE = 12.9935
Epoch 400: MSE = 12.9396
Epoch 500: MSE = 12.9284
Epoch 600: MSE = 12.9259
Epoch 700: MSE = 12.9254
Epoch 800: MSE = 12.9253
Epoch 900: MSE = 12.9252
Epoch 999: MSE = 12.9252
RMSE: 3.597973501613889
R²: 0.8859843166097213
Accuracy within 5% tolerance: 22.93%

#This bad performance compared to the linear regression is due to the fact that the data is clear and thus can afford to rely
on specifc features for predictions. By adding regularization we are focing the model to use features more equally which
is self-sabotage for this (more) noiseless dataset
'''

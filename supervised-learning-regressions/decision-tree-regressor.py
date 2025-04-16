import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):

        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        # Combine X and y for convenience
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self._build_tree(dataset, depth=0)

    def _build_tree(self, dataset, depth):
        # Separate features and target
        X = dataset[:, :-1]
        y = dataset[:, -1]
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        best_split = self._find_best_split(dataset, n_features)
        # If no valid split was found, return a leaf node.
        if best_split['mse'] is None:
            leaf_value = np.mean(y)
            return Node(value=leaf_value)
        
        # Recursively build the left and right subtrees.
        left_subtree = self._build_tree(best_split['left_dataset'], depth + 1)
        right_subtree = self._build_tree(best_split['right_dataset'], depth + 1)
        
        return Node(feature_index=best_split['feature_index'],
                    threshold=best_split['threshold'],
                    left=left_subtree,
                    right=right_subtree)

    def _find_best_split(self, dataset, n_features):
        best_mse = float('inf')
        best_split = {'feature_index': None, 'threshold': None,
                      'left_dataset': None, 'right_dataset': None, 'mse': None}
        
        for feature_index in range(n_features):
            feature_values = dataset[:, feature_index]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                # Split dataset based on the threshold
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold
                
                # Skip splits that don't divide the dataset
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                left_dataset = dataset[left_indices]
                right_dataset = dataset[right_indices]
                
                # Calculate MSE for the left and right splits
                y_left = left_dataset[:, -1]
                y_right = right_dataset[:, -1]
                mse_left = np.mean((y_left - np.mean(y_left)) ** 2)
                mse_right = np.mean((y_right - np.mean(y_right)) ** 2)
                mse = (len(y_left) * mse_left + len(y_right) * mse_right) / len(dataset)
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_dataset': left_dataset,
                        'right_dataset': right_dataset,
                        'mse': mse
                    }
        
        # If no valid split is found, mse remains infinity.
        if best_mse == float('inf'):
            best_split['mse'] = None
        
        return best_split

    def predict(self, X):
        predictions = [self._predict_sample(sample, self.root) for sample in X]
        return np.array(predictions)

    def _predict_sample(self, sample, node):
      
        # If this is a leaf node, return its value.
        if node.value is not None:
            return node.value
        # Otherwise, choose branch based on the threshold.
        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
        
    def accuracy(self, X, y, tolerance=0.05):
        y_pred = self.predict(X)
        y_true = np.array(y)
        
        # Avoid division by zero by creating a mask for non-zero true values.
        mask = y_true != 0  
        if np.sum(mask) == 0:
            return 0  # No valid entries for computing accuracy.
        relative_error = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
        return np.mean(relative_error <= tolerance)


#Evaluation metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot



csv_file = "data-preprocessing/scaled_numerical1.csv"
df = pd.read_csv(csv_file)

features = [
    'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)'
]
target = 'Apparent Temperature (C)'
X = df[features].values
y = df[target].values

indices = np.random.permutation(len(y))
X = X[indices]
y = y[indices]

split_idx = int(len(y) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

acc = tree.accuracy(X_test, y_test, tolerance=0.05)
print(f"Accuracy within 5% tolerance: {acc * 100:.2f}%")
print("RMSE:", rmse(y_test, predictions))
print("R²:", r2_score(y_test, predictions))

'''
Interestingly shows at 60.99% accuracy compared to a linear regression model which only shows 46.34% 
(with a confidence interval of 5%) - This shows a 32% performance imporvement.
I hypothesise this is because the tree can capture non linear patterns in the data over past models,
furthermore, the tree is less sensitive to outliers and noise in the data and thus better captures relationships in the data 
at a multitude of levels higher than a 'basic' linear regression or variations
'''

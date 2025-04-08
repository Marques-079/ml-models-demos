import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("data-preprocessing/weatherHistory.csv")

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# ========== 1. Distribution plots for numerical features ==========
numerical_cols = df.select_dtypes(include='number').columns

for col in numerical_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True, bins=40, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== 2. Box plots to check for outliers ==========
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color='lightcoral')
    plt.title(f"Boxplot of {col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== 3. Categorical value counts ==========
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
    plt.title(f"Count of categories in {col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

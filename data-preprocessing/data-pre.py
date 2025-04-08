import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df = pd.read_csv("data-preprocessing/weatherHistory.csv")
print(df.info())
print(df.describe())

def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.75 * IQR
        upper = Q3 + 1.75 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

#selecting column that only have numeric values
columns = [
    'Temperature (C)',
    'Apparent Temperature (C)',
    'Humidity',
    'Wind Speed (km/h)',
    'Wind Bearing (degrees)',
    'Visibility (km)',
    'Loud Cover',
    'Pressure (millibars)'
]

df = remove_outliers_iqr(df, columns)
df.to_csv('unscaled_full.csv', index = False)

features_to_scale = df.select_dtypes(include='number').columns

df_scaled = pd.DataFrame(
    scaler.fit_transform(df[features_to_scale]),
    columns=features_to_scale
)

df_scaled.to_csv('scaled_numerical.csv', index = False)





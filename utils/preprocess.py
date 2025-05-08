import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()
    df = df.drop(['CustomerID', 'Gender'], axis=1, errors='ignore')  # Optional
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

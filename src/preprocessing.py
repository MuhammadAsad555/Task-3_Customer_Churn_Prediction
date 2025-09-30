
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Basic cleaning:
     - Drop customerID
     - Convert TotalCharges to numeric (coerce errors)
     - Strip spaces from object columns if present
    """
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # Trim whitespace from string columns
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # Convert TotalCharges to numeric (some rows are blank)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing TotalCharges with MonthlyCharges * tenure as a simple estimate
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
        else:
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

def encode_features(df, drop_first=True):
    """Encode categorical features and extract X, y.
    - target column expected to be 'Churn' with Yes/No
    - returns X (dataframe) and y (Series)
    """
    df = df.copy()
    if 'Churn' not in df.columns:
        raise ValueError("DataFrame must contain 'Churn' column as target.")
    y = df['Churn'].map({'Yes':1, 'No':0})
    X = df.drop(columns=['Churn'])
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    # One-hot encode categoricals
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

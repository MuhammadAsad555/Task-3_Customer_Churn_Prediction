
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.preprocessing import load_data, clean_data, encode_features, split_data

def train_and_save(data_path, model_path='models/logistic_model.joblib'):
    df = load_data(data_path)
    df = clean_data(df)
    X, y = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Build pipeline with scaler + logistic regression
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(X_train, y_train)
    # Evaluate on train and test
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'pipeline': pipe, 'feature_columns': X.columns.tolist()}, model_path)
    print(f"Saved model to {model_path}")
    return model_path, X_test, y_test

if __name__ == '__main__':
    # Default expected CSV filename — update path if needed
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    train_and_save(data_path)

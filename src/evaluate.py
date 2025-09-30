
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.preprocessing import load_data, clean_data, encode_features, split_data

def evaluate_model(model_path, data_path, output_dir='outputs'):
    # Load model
    obj = joblib.load(model_path)
    pipe = obj['pipeline']
    feature_columns = obj.get('feature_columns', None)
    # Load and prepare data
    df = load_data(data_path)
    df = clean_data(df)
    X, y = encode_features(df)
    # Align columns (in case the dataset used for eval has different dummies)
    if feature_columns is not None:
        # Add missing columns with zeros
        for c in feature_columns:
            if c not in X.columns:
                X[c] = 0
        # Remove any extra columns
        X = X[feature_columns]
    # Predict
    y_pred = pipe.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, digits=4)
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)
    # Plot and save confusion matrix
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    # Put numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(outpath)
    print(f"Saved confusion matrix to {outpath}")
    return acc, cm, report

if __name__ == '__main__':
    model_path = 'models/logistic_model.joblib'
    data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    evaluate_model(model_path, data_path)

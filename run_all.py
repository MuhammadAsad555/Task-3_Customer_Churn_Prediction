
from src.train_model import train_and_save
from src.evaluate import evaluate_model
import os

DATA_PATH = 'data/Telco-Customer-Churn.csv'
MODEL_PATH = 'models/logistic_model.joblib'
OUTPUT_DIR = 'outputs'

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please download the Telco Customer Churn CSV and place it there.")
        return
    model_path, X_test, y_test = train_and_save(DATA_PATH, model_path=MODEL_PATH)
    evaluate_model(MODEL_PATH, DATA_PATH, output_dir=OUTPUT_DIR)

if __name__ == '__main__':
    main()

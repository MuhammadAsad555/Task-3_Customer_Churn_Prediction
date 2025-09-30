# Customer_Churn_Prediction
Customer Churn Prediction

Overview:

This project predicts customer churn (whether a customer will leave the service or not) using machine learning models. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation. The goal is to help businesses reduce churn by identifying at-risk customers in advance.


Models Implemented:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting (XGBoost/LightGBM, if used)

Support Vector Machine (if used)

Evaluation Metrics:

Accuracy

Precision, Recall, F1-Score

ROC-AUC


Results:

Train accuracy: 0.8065
Test  accuracy: 0.8070
Saved model to models/logistic_model.joblib
Accuracy: 0.8066

Classification report:
               precision    recall  f1-score   support

           0     0.8481    0.8976    0.8721      5174 
           1     0.6618    0.5548    0.6036      1869 

    accuracy                         0.8066      7043 
   macro avg     0.7549    0.7262    0.7379      7043 
weighted avg     0.7986    0.8066    0.8009      7043 

Confusion matrix:
 [[4644  530]
 [ 832 1037]]
Saved confusion matrix to outputs\confusion_matrix.png

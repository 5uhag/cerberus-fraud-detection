# model.py
# Author: Senior Python Data Scientist
# Description: Backend Machine Learning Script for Credit Card Fraud Detection.

import json
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

def build_fraud_model(data_path='creditcard.csv'):
    # 1. Data Ingestion
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please place the file in the project directory.")
        return

    data = pd.read_csv(data_path)
    print(f"Dataset Ingested Successfully: {data.shape} transactions found.")

    # 2. Preprocessing & Feature Engineering
    scaler = StandardScaler()
    
    # Scale both simultaneously so the scaler remembers both!
    scaled_features = scaler.fit_transform(data[['Amount', 'Time']])
    data['scaled_amount'] = scaled_features[:, 0]
    data['scaled_time'] = scaled_features[:, 1]
    
    # Remove raw columns to prevent redundancy
    data.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Define features (X) and target label (y)
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Save the order of features to ensure the dashboard feeds data correctly
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, 'features_list.pkl')

    # 3. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Balancing Classes with SMOTE
    print("Applying SMOTE to balance the minority fraud class...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"New Class Distribution: {np.bincount(y_train_res)}")

    # 5. Model Training (Random Forest)
    print("Initializing Random Forest Classifier training...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # 6. Evaluation and Performance Logging
    predictions = model.predict(X_test)
    report_text = classification_report(y_test, predictions, zero_division=0)
    report_data = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)
    print("\nModel Evaluation Report:")
    print(report_text)

    fraud_metrics = report_data.get('1', {})
    metrics = {
        'precision': round(float(fraud_metrics.get('precision', 0.0)), 4),
        'recall': round(float(fraud_metrics.get('recall', 0.0)), 4),
        'f1_score': round(float(fraud_metrics.get('f1-score', 0.0)), 4),
        'support': int(fraud_metrics.get('support', 0)),
        'accuracy': round(float(report_data.get('accuracy', 0.0)), 4),
        'confusion_matrix': matrix.tolist(),
    }

    with open('metrics.json', 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print('Saved metrics to metrics.json')

    # 7. Model Serialization
    joblib.dump(model, 'fraud_model.pkl')
    joblib.dump(scaler, 'data_scaler.pkl')
    print("Success: Model and Scaler have been serialized to disk.")

if __name__ == "__main__":
    build_fraud_model()
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def run_model():
    project_root = os.path.abspath(os.path.join(os.path.dirname(_file_), '..'))
    os.chdir(project_root)

    # Create output directories upfront
    os.makedirs("Data/model", exist_ok=True)
    os.makedirs("plots/model", exist_ok=True)
    
    # Step 1 - Load Cleaned Data
    print("Loading cleaned data...")
    X_train = pd.read_csv("Data/X_train_clean.csv")
    X_test  = pd.read_csv("Data/X_test_clean.csv")
    y_train = pd.read_csv("Data/y_train.csv").squeeze()
    y_test  = pd.read_csv("Data/y_test.csv").squeeze()

    print("Train shape:", X_train.shape)
    print("Test shape: ", X_test.shape)

    # Step 2 - Feature Engineering
    print("Adding engineered feature: fuel_cost_to_income_ratio...")
    X_train["fuel_cost_to_income_ratio"] = (
        X_train["monthly_fuel_cost_inr"] / (X_train["income_inr_month"] + 1e-9)
    )
    X_test["fuel_cost_to_income_ratio"] = (
        X_test["monthly_fuel_cost_inr"] / (X_test["income_inr_month"] + 1e-9)
    )

    scaler = StandardScaler()
    X_train[["fuel_cost_to_income_ratio"]] = scaler.fit_transform(
        X_train[["fuel_cost_to_income_ratio"]]
    )
    X_test[["fuel_cost_to_income_ratio"]] = scaler.transform(
        X_test[["fuel_cost_to_income_ratio"]]
    )

    print("Feature engineering done. New feature: fuel_cost_to_income_ratio")
    print("Updated train shape:", X_train.shape)

    # Step 3 - Train Logistic Regression Model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

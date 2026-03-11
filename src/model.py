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
    
        # Step 4 - Evaluate on Test Set
    print("\nEvaluating model on test set...")
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Unlikely", "Likely"]))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Step 5 - Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unlikely", "Likely"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix — Logistic Regression")
    plt.savefig("plots/model/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved to plots/model/confusion_matrix.png")

    # Step 6 - Feature Importance (Coefficients)
    coef_df = pd.DataFrame({
        "feature":     X_train.columns,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", ascending=False)

    print("\nTop 10 features driving Likely Adoption:")
    print(coef_df.head(10).to_string(index=False))

    print("\nTop 10 features driving Unlikely Adoption:")
    print(coef_df.tail(10).to_string(index=False))

    plt.figure(figsize=(10, 8))
    top20  = pd.concat([coef_df.head(10), coef_df.tail(10)])
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top20["coefficient"]]
    plt.barh(top20["feature"], top20["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 20 Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("plots/model/feature_coefficients.png", bbox_inches="tight")
    plt.close()
    print("Feature coefficients plot saved to plots/model/feature_coefficients.png")

    # Step 7 - Save Model and Predictions
    joblib.dump(model, "Data/model/logistic_regression_model.pkl")
    print("Model saved to Data/model/logistic_regression_model.pkl")

    results_df = X_test.copy()
    results_df["actual_label"]         = y_test.values
    results_df["predicted_label"]      = y_pred
    results_df["adoption_probability"] = y_pred_proba

    results_df.to_csv("Data/model/predictions.csv", index=False)
    print("Predictions saved to Data/model/predictions.csv")

    # Step 8 - Save Metrics Summary
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(auc, 4)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("Data/model/model_metrics.csv", index=False)
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    print("Metrics saved to Data/model/model_metrics.csv")

if _name_ == "_main_":
    run_model()

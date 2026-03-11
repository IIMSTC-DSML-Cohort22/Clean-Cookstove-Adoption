import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ZONE_WEIGHTS = {
    "Central": 1.4,   # MP, Chhattisgarh — low income, biomass dependent
    "East":    1.5,   # WB, Odisha, Jharkhand, Assam — poorest zone, highest inertia
    "North":   1.2,   # UP, Bihar, Rajasthan — large population, mixed access
    "South":   0.8,   # Karnataka, TN, Telangana — already high adoption baseline
    "West":    0.9,   # Maharashtra, Gujarat — urban-heavy, high income
}

FUEL_WEIGHTS = {
    "current_fuel_type_Firewood":        1.5,
    "current_fuel_type_Crop Residue":    1.5,
    "current_fuel_type_Cow Dung Cake":   1.4,
    "current_fuel_type_Kerosene":        1.1,
    "current_fuel_type_LPG":             0.8,
}


def get_zone_label(row, zone_cols):
    """Reconstruct zone name from one-hot encoded columns."""
    for col in zone_cols:
        if row[col] == 1:
            return col.replace("zone_", "")
    return "Central"  # dropped reference category


def build_sample_weights(X, zone_cols):
    """
    Compute per-sample training weights combining:
    1. Zone-level economic disparity weight
    2. Fuel availability weight
    """
    weights = np.ones(len(X))

    # Apply zone weights
    zone_labels = X.apply(lambda row: get_zone_label(row, zone_cols), axis=1)
    for zone, w in ZONE_WEIGHTS.items():
        weights[zone_labels == zone] *= w

    # Apply fuel weights
    for fuel_col, w in FUEL_WEIGHTS.items():
        if fuel_col in X.columns:
            weights[X[fuel_col] == 1] *= w

    return weights


def run_fine_tuning():
    project_root = os.path.abspath(os.path.join(os.path.dirname(_file_), ".."))
    os.chdir(project_root)

    os.makedirs("Data/fine_tuned", exist_ok=True)
    os.makedirs("plots/fine_tuned", exist_ok=True)

    # ── Load Data ─────────────────────────────────────────────────────────────
    print("Loading cleaned data...")
    X_train = pd.read_csv("Data/X_train_clean.csv")
    X_test  = pd.read_csv("Data/X_test_clean.csv")
    y_train = pd.read_csv("Data/y_train.csv").squeeze()
    y_test  = pd.read_csv("Data/y_test.csv").squeeze()

    # ── Feature Engineering (same as model.py) ────────────────────────────────
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

    zone_cols = [c for c in X_train.columns if c.startswith("zone_")]

    # ── Baseline Model (no fine-tuning) ───────────────────────────────────────
    print("\nTraining baseline model...")
    baseline_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    baseline_model.fit(X_train, y_train)
    y_proba_baseline = baseline_model.predict_proba(X_test)[:, 1]
    auc_baseline     = roc_auc_score(y_test, y_proba_baseline)
    print(f"Baseline AUC : {auc_baseline:.4f}")

    # ── Fine-Tuned Model (with sample weights) ────────────────────────────────
    print("\nBuilding regional sample weights...")
    sample_weights = build_sample_weights(X_train, zone_cols)
    print(f"Sample weight range: {sample_weights.min():.2f} — {sample_weights.max():.2f}")
    print(f"Mean weight        : {sample_weights.mean():.4f}")

    print("\nTraining fine-tuned model...")
    finetuned_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    finetuned_model.fit(X_train, y_train, sample_weight=sample_weights)
    y_proba_finetuned = finetuned_model.predict_proba(X_test)[:, 1]
    auc_finetuned     = roc_auc_score(y_test, y_proba_finetuned)
    print(f"Fine-Tuned AUC : {auc_finetuned:.4f}")
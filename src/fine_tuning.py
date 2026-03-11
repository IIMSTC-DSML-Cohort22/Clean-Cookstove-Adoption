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
    # ── Plot 1: ROC Curve — Baseline vs Fine-Tuned ────────────────────────────
    print("\nPlotting ROC Curve comparison...")
    fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_baseline)
    fpr_f, tpr_f, _ = roc_curve(y_test, y_proba_finetuned)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_b, tpr_b, color="#e74c3c", lw=2,
            label=f"Baseline       (AUC = {auc_baseline:.4f})")
    ax.plot(fpr_f, tpr_f, color="#2ecc71", lw=2,
            label=f"Fine-Tuned     (AUC = {auc_finetuned:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
            label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr_f, tpr_f, alpha=0.07, color="#2ecc71")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Baseline vs Fine-Tuned ROC Curve\nRegional Economic Disparity Adjustment", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig("plots/fine_tuned/roc_baseline_vs_finetuned.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/fine_tuned/roc_baseline_vs_finetuned.png")

    # ── Plot 2: AUC per Zone — Baseline vs Fine-Tuned ─────────────────────────
    print("\nPlotting per-zone AUC comparison...")

    def get_zone_labels(X):
        return X.apply(lambda row: get_zone_label(row, zone_cols), axis=1)

    X_test_zone = get_zone_labels(X_test)
    zones        = sorted(X_test_zone.unique())
    auc_b_zones  = []
    auc_f_zones  = []

    for zone in zones:
        mask = X_test_zone == zone
        if mask.sum() < 10:
            auc_b_zones.append(None)
            auc_f_zones.append(None)
            continue
        auc_b_zones.append(roc_auc_score(y_test[mask], y_proba_baseline[mask]))
        auc_f_zones.append(roc_auc_score(y_test[mask], y_proba_finetuned[mask]))

    x      = np.arange(len(zones))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, auc_b_zones, width, label="Baseline",   color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, auc_f_zones, width, label="Fine-Tuned", color="#2ecc71", alpha=0.85)
    ax.set_xlabel("Zone", fontsize=12)
    ax.set_ylabel("AUC Score", fontsize=12)
    ax.set_title("AUC by Zone — Baseline vs Fine-Tuned\nImpact of Regional Disparity Adjustment", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(zones, fontsize=11)
    ax.set_ylim([0.5, 1.0])
    ax.axhline(0.85, color="grey", linestyle="--", lw=1, label="AUC = 0.85 reference")
    ax.legend(fontsize=10)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/fine_tuned/auc_by_zone_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/fine_tuned/auc_by_zone_comparison.png")

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("       FINE-TUNING SUMMARY")
    print("=" * 55)
    print(f"  Baseline AUC    : {auc_baseline:.4f}")
    print(f"  Fine-Tuned AUC  : {auc_finetuned:.4f}")
    delta = auc_finetuned - auc_baseline
    print(f"  AUC Delta       : {delta:+.4f}  {'✅ Improved' if delta >= 0 else '⚠️ Reduced'}")

    print("\n  Per-Zone AUC:")
    print(f"  {'Zone':<12} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>8}")
    print("  " + "-" * 44)
    for zone, ab, af in zip(zones, auc_b_zones, auc_f_zones):
        if ab is None:
            continue
        print(f"  {zone:<12} {ab:>10.4f} {af:>12.4f} {af - ab:>+8.4f}")

    print("\n  Fine-Tuned Classification Report:")
    y_pred_ft = finetuned_model.predict(X_test)
    print(classification_report(y_test, y_pred_ft, target_names=["Unlikely", "Likely"]))
    print("=" * 55)

    # ── Save Outputs ──────────────────────────────────────────────────────────
    joblib.dump(finetuned_model, "Data/fine_tuned/finetuned_model.pkl")
    print("Fine-tuned model saved → Data/fine_tuned/finetuned_model.pkl")

    zone_summary = pd.DataFrame({
        "zone":         zones,
        "auc_baseline": auc_b_zones,
        "auc_finetuned": auc_f_zones,
        "delta":        [af - ab if ab else None for ab, af in zip(auc_b_zones, auc_f_zones)]
    })
    zone_summary.to_csv("Data/fine_tuned/zone_auc_comparison.csv", index=False)
    print("Zone AUC comparison saved → Data/fine_tuned/zone_auc_comparison.csv")


if __name__ == "__main__":
    run_fine_tuning()
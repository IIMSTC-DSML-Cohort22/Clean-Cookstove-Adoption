import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, roc_auc_score


# ── Deployment Strategy Definitions ──────────────────────────────────────────
STRATEGIES = {
    "Mass Outreach":        {"threshold": 0.35, "color": "#e67e22",
                             "description": "Awareness campaigns — low cost per contact"},
    "Balanced Deployment":  {"threshold": 0.50, "color": "#3498db",
                             "description": "Field agent visits — moderate cost"},
    "Precision Deployment": {"threshold": 0.65, "color": "#2ecc71",
                             "description": "Subsidised stove distribution — limited budget"},
}


def compute_strategy_metrics(y_true, y_proba, threshold):
    """Compute deployment metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    targeted   = tp + fp
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    wasted     = fp
    missed     = fn

    return {
        "targeted":         targeted,
        "true_adopters":    tp,
        "wasted_visits":    wasted,
        "missed_adopters":  missed,
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1_score":         round(f1, 4),
    }


def run_deployment_strategy():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    os.makedirs("Data/deployment", exist_ok=True)
    os.makedirs("plots/deployment", exist_ok=True)

    # ── Load Predictions ──────────────────────────────────────────────────────
    print("Loading predictions...")
    df           = pd.read_csv("Data/model/predictions.csv")
    y_test       = df["actual_label"]
    y_proba      = df["adoption_probability"]
    total        = len(y_test)
    total_actual = int(y_test.sum())

    print(f"Total households in test set : {total}")
    print(f"Actual adopters              : {total_actual}")
    print(f"Actual non-adopters          : {total - total_actual}")

    # ── Compute Metrics Per Strategy ──────────────────────────────────────────
    rows = []
    for name, cfg in STRATEGIES.items():
        m = compute_strategy_metrics(y_test, y_proba, cfg["threshold"])
        rows.append({
            "Strategy":            name,
            "Threshold":           cfg["threshold"],
            "Description":         cfg["description"],
            "Households Targeted": m["targeted"],
            "True Adopters Reached": m["true_adopters"],
            "Wasted Visits":       m["wasted_visits"],
            "Missed Adopters":     m["missed_adopters"],
            "Precision":           m["precision"],
            "Recall":              m["recall"],
            "F1 Score":            m["f1_score"],
        })

    results_df = pd.DataFrame(rows)

    # ── Print Deployment Simulation Table ─────────────────────────────────────
    print("\n" + "=" * 75)
    print("          DEPLOYMENT STRATEGY SIMULATION")
    print("=" * 75)
    print(f"  {'Strategy':<22} {'Threshold':>9} {'Targeted':>9} {'Reached':>9} "
          f"{'Wasted':>8} {'Missed':>8} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    print("  " + "-" * 73)
    for _, row in results_df.iterrows():
        print(f"  {row['Strategy']:<22} {row['Threshold']:>9.2f} "
              f"{row['Households Targeted']:>9} "
              f"{row['True Adopters Reached']:>9} "
              f"{row['Wasted Visits']:>8} "
              f"{row['Missed Adopters']:>8} "
              f"{row['Precision']:>10.1%} "
              f"{row['Recall']:>8.1%} "
              f"{row['F1 Score']:>7.4f}")
    print("=" * 75)
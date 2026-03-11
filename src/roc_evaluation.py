import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import joblib
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

def evaluate_roc(
    predictions_path="Data/model/predictions.csv",
    model_path="Data/model/logistic_regression_model.pkl",
    output_dir="plots/roc"
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    
    os.makedirs(output_dir, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────────────────────
    print("Loading predictions...")
    df = pd.read_csv(predictions_path)

    y_test       = df["actual_label"]
    y_pred_proba = df["adoption_probability"]
    y_pred       = df["predicted_label"]

    # ── 1. ROC Curve ──────────────────────────────────────────────────────────
    print("Plotting ROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Find the default threshold (0.5) operating point
    default_idx = np.argmin(np.abs(thresholds - 0.5))

    # Find optimal threshold using Youden's J statistic (maximise TPR - FPR)
    youden_idx      = np.argmax(tpr - fpr)
    optimal_thresh  = thresholds[youden_idx]
    optimal_fpr     = fpr[youden_idx]
    optimal_tpr     = tpr[youden_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2ecc71", lw=2.5,
            label=f"Logistic Regression (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--",
            label="Random Classifier (AUC = 0.50)")

    # Default threshold point
    ax.scatter(fpr[default_idx], tpr[default_idx],
               color="#e74c3c", zorder=5, s=100,
               label=f"Threshold = 0.50  (FPR={fpr[default_idx]:.2f}, TPR={tpr[default_idx]:.2f})")

    # Optimal threshold point
    ax.scatter(optimal_fpr, optimal_tpr,
               color="#3498db", zorder=5, s=100, marker="D",
               label=f"Optimal Threshold = {optimal_thresh:.2f}  (FPR={optimal_fpr:.2f}, TPR={optimal_tpr:.2f})")

    ax.fill_between(fpr, tpr, alpha=0.08, color="#2ecc71")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ROC Curve — Clean Cookstove Adoption\nDeployment Strategy Evaluation", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC Curve saved → plots/roc_curve.png  |  AUC = {auc_score:.4f}")

    # ── 2. Threshold vs Precision / Recall / F1 ───────────────────────────────
    print("Plotting Threshold Analysis...")
    thresh_range = np.arange(0.1, 0.91, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresh_range:
        y_t = (y_pred_proba >= t).astype(int)
        tp = ((y_t == 1) & (y_test == 1)).sum()
        fp = ((y_t == 1) & (y_test == 0)).sum()
        fn = ((y_t == 0) & (y_test == 1)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    best_f1_thresh = thresh_range[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_range, precisions, label="Precision", color="#3498db", lw=2)
    ax.plot(thresh_range, recalls,    label="Recall",    color="#e67e22", lw=2)
    ax.plot(thresh_range, f1s,        label="F1 Score",  color="#2ecc71", lw=2)
    ax.axvline(0.5,              color="#e74c3c", linestyle="--", lw=1.5,
               label="Default Threshold (0.50)")
    ax.axvline(best_f1_thresh,   color="#9b59b6", linestyle="--", lw=1.5,
               label=f"Best F1 Threshold ({best_f1_thresh:.2f})")
    ax.axvline(optimal_thresh,   color="#3498db", linestyle=":",  lw=1.5,
               label=f"Youden Optimal ({optimal_thresh:.2f})")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold\nDeployment Threshold Selection", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Threshold Analysis saved → plots/threshold_analysis.png  |  Best F1 Threshold = {best_f1_thresh:.2f}")

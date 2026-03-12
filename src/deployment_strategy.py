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
    # ── Plot 1: Annotated ROC Curve with 3 Strategy Points ───────────────────
    print("\nPlotting annotated ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(fpr, tpr, color="#2c3e50", lw=2.5,
            label=f"Logistic Regression (AUC = {auc_score:.4f})", zorder=2)
    ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--",
            label="Random Classifier", zorder=1)
    ax.fill_between(fpr, tpr, alpha=0.06, color="#2c3e50")

    for name, cfg in STRATEGIES.items():
        t     = cfg["threshold"]
        color = cfg["color"]
        idx   = np.argmin(np.abs(thresholds - t))
        ax.scatter(fpr[idx], tpr[idx], color=color, s=140, zorder=5)
        ax.annotate(
            f"{name}\n(t={t}, P={results_df.loc[results_df['Strategy']==name,'Precision'].values[0]:.0%},"
            f" R={results_df.loc[results_df['Strategy']==name,'Recall'].values[0]:.0%})",
            xy=(fpr[idx], tpr[idx]),
            xytext=(fpr[idx] + 0.06, tpr[idx] - 0.07),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1)
        )

    ax.set_xlabel("False Positive Rate (Wasted Visits)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Adopters Reached)", fontsize=12)
    ax.set_title("Deployment Strategy Evaluation — ROC Curve\nClean Cookstove Adoption | SDG 7",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig("plots/deployment/roc_deployment_strategies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/deployment/roc_deployment_strategies.png")

    # ── Plot 2: Strategy Comparison Bar Chart ─────────────────────────────────
    print("Plotting strategy comparison bar chart...")
    strategy_names = results_df["Strategy"].tolist()
    colors         = [STRATEGIES[s]["color"] for s in strategy_names]
    x              = np.arange(len(strategy_names))
    width          = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width,     results_df["Households Targeted"],    width, label="Households Targeted", color="#2c3e50", alpha=0.85)
    b2 = ax.bar(x,             results_df["True Adopters Reached"],  width, label="True Adopters Reached", color="#2ecc71", alpha=0.85)
    b3 = ax.bar(x + width,     results_df["Wasted Visits"],          width, label="Wasted Visits", color="#e74c3c", alpha=0.85)

    for bar in [b1, b2, b3]:
        for rect in bar:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 1,
                    str(int(rect.get_height())),
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, fontsize=11)
    ax.set_ylabel("Number of Households", fontsize=12)
    ax.set_title("Deployment Strategy Comparison\nHouseholds Targeted vs Reached vs Wasted",
                 fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/deployment/strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/deployment/strategy_comparison.png")

    # ── Plot 3: Precision vs Recall Trade-off ─────────────────────────────────
    print("Plotting Precision vs Recall trade-off...")
    fig, ax = plt.subplots(figsize=(8, 5))

    thresh_range = np.arange(0.1, 0.91, 0.01)
    precs, recs  = [], []
    for t in thresh_range:
        y_t = (y_proba >= t).astype(int)
        tp  = ((y_t == 1) & (y_test == 1)).sum()
        fp  = ((y_t == 1) & (y_test == 0)).sum()
        fn  = ((y_t == 0) & (y_test == 1)).sum()
        precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recs.append(tp / (tp + fn)  if (tp + fn)  > 0 else 0)

    ax.plot(recs, precs, color="#2c3e50", lw=2, label="Precision-Recall Curve")

    for name, cfg in STRATEGIES.items():
        m = compute_strategy_metrics(y_test, y_proba, cfg["threshold"])
        ax.scatter(m["recall"], m["precision"],
                   color=cfg["color"], s=120, zorder=5,
                   label=f"{name} (t={cfg['threshold']})")

    ax.set_xlabel("Recall (Adopters Reached)", fontsize=12)
    ax.set_ylabel("Precision (Targeting Accuracy)", fontsize=12)
    ax.set_title("Precision vs Recall — Deployment Trade-off\nClean Cookstove Adoption | SDG 7",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig("plots/deployment/precision_recall_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → plots/deployment/precision_recall_tradeoff.png")

    # ── Save Results Table ────────────────────────────────────────────────────
    results_df.to_csv("Data/deployment/deployment_strategy_results.csv", index=False)
    print("\nDeployment results saved → Data/deployment/deployment_strategy_results.csv")
    print("\nDone. All deployment strategy outputs generated.")


if __name__ == "__main__":
    run_deployment_strategy()
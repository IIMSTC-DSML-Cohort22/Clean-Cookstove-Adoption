import os
import sys
import time
import traceback

# Set working directory to project root before anything else
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

# Add src/ to path so all modules can be imported
sys.path.append(os.path.join(project_root, "src"))
  
from clean_data          import clean_data
from eda                 import run_eda
from model               import run_model
from roc_evaluation      import evaluate_roc
from fine_tuning         import run_fine_tuning
from deployment_strategy import run_deployment_strategy


# ── Pipeline Step Definitions ─────────────────────────────────────────────────
STEPS = [
    
    {
        "name":     "Step 1 — Data Cleaning",
        "function": clean_data,
        "args":     {},
    },
    {
        "name":     "Step 2 — Exploratory Data Analysis",
        "function": run_eda,
        "args":     {},
    },
    {
        "name":     "Step 3 — Feature Engineering + Model Training",
        "function": run_model,
        "args":     {},
    },
    {
        "name":     "Step 4 — ROC Curve Evaluation",
        "function": evaluate_roc,
        "args":     {},
    },
    {
        "name":     "Step 5 — Regional Fine-Tuning",
        "function": run_fine_tuning,
        "args":     {},
    },
    {
        "name":     "Step 6 — Deployment Strategy Evaluation",
        "function": run_deployment_strategy,
        "args":     {},
    },
]


# ── Pipeline Runner ───────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("   CLEAN COOKSTOVE ADOPTION — FULL PIPELINE")
    print("   SDG 7: Affordable and Clean Energy")
    print(f"   Working directory: {os.getcwd()}")
    print("=" * 60)

    total_start = time.time()
    results     = []

    for i, step in enumerate(STEPS, 1):
        print(f"\n{'─' * 60}")
        print(f"  {step['name']}")
        print(f"{'─' * 60}")

        step_start = time.time()
        try:
            step["function"](**step["args"])
            duration = round(time.time() - step_start, 2)
            print(f"  ✅ Completed in {duration}s")
            results.append({"step": step["name"], "status": "✅ Passed", "duration": duration})

        except Exception as e:
            duration = round(time.time() - step_start, 2)
            print(f"\n  ❌ FAILED: {step['name']}")
            print(f"  Error: {e}")
            print("\n  Full traceback:")
            traceback.print_exc()
            results.append({"step": step["name"], "status": f"❌ Failed — {e}", "duration": duration})
            print("\n  Pipeline stopped. Fix the error above and re-run.")
            break

    # ── Final Summary ─────────────────────────────────────────────────────────
    total_duration = round(time.time() - total_start, 2)
    print(f"\n{'=' * 60}")
    print("   PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Step':<45} {'Status':<20} {'Time':>6}")
    print(f"  {'─' * 57}")
    for r in results:
        print(f"  {r['step']:<45} {r['status']:<20} {r['duration']:>5}s")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_duration}s")

    passed = sum(1 for r in results if "Passed" in r["status"])
    total  = len(STEPS)
    print(f"  Steps completed: {passed} / {total}")

    if passed == total:
        print("\n  ✅ Full pipeline completed successfully.")
        print("\n  Output locations:")
        print("    Data/model/          → predictions, metrics, model")
        print("    Data/fine_tuned/     → fine-tuned model, zone AUC")
        print("    Data/deployment/     → deployment strategy results")
        print("    plots/eda/           → 9 EDA plots")
        print("    plots/model/         → confusion matrix, coefficients")
        print("    plots/fine_tuned/    → ROC comparison, zone AUC chart")
        print("    plots/deployment/    → annotated ROC, bar chart, PR curve")
    else:
        print("\n  ⚠️  Pipeline did not complete. See error above.")

    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
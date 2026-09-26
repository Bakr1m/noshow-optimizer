"""Day 56: register recorded experiments in MLflow (documented backfill).

Retraining (~10 min CPU) to regenerate identical numbers would be theater;
this logs the saved metrics as tagged backfill runs. Train scripts remain
the source of truth for future live-logged runs.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRACKING_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT = "noshow_ops"


def main():
    base = json.loads((PROJECT_ROOT / "models" / "baseline_metrics.json").read_text())
    tune = json.loads((PROJECT_ROOT / "models" / "tuning_metrics.json").read_text())
    sim = json.loads((PROJECT_ROOT / "models" / "simulation.json").read_text())
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="lr_baseline"):
        mlflow.log_param("model", "LogisticRegression-balanced")
        mlflow.log_metric("roc_auc", base["roc_auc"])
        mlflow.log_metric("pr_auc", base["pr_auc"])
        mlflow.log_metric("cost_at_0.5", base["cost_at_0.5"]["total"])
    for name in ("xgboost", "lightgbm"):
        m = tune["models"][name]
        with mlflow.start_run(run_name=f"{name}_tuned"):
            mlflow.log_param("model", name)
            mlflow.log_param("best_threshold", m["best_threshold"])
            mlflow.log_metric("roc_auc", m["roc_auc"])
            mlflow.log_metric("pr_auc", m["pr_auc"])
            mlflow.log_metric("test_cost", m["test_cost"]["total"])
    with mlflow.start_run(run_name="reminder_simulation"):
        mlflow.log_metric("best_policy_cost", sim["best"]["total"])
        mlflow.log_metric("baseline_cost", sim["baseline"]["total"])
    print(f"logged 4 runs to '{EXPERIMENT}'")


if __name__ == "__main__":
    main()

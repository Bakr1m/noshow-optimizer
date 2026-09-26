"""Day 53: baseline Logistic Regression (interpretable floor) + cost framing.

Pipeline (no leakage: transformers fit on train folds only):
numeric median-impute, categorical most-frequent + one-hot.
Output: models/lr_baseline.joblib + models/baseline_metrics.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from costs import COST_FN, COST_FP, cost_at_threshold
from features import LABEL, build_features, feature_columns

MODEL_PATH = PROJECT_ROOT / "models" / "lr_baseline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "baseline_metrics.json"
SEED = 42


def build_preprocessor(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )


def main():
    df = build_features()
    cols = feature_columns(df)
    X, y = df[cols], df[LABEL]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    pipe = Pipeline([
        ("prep", build_preprocessor(Xtr)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                   random_state=SEED)),
    ])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    c = cost_at_threshold(yte, proba, 0.5)
    out = {
        "n_train": len(Xtr), "n_test": len(Xte),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "pr_auc": round(float(average_precision_score(yte, proba)), 4),
        "f1_at_0.5": round(float(f1_score(yte, pred)), 4),
        "confusion_at_0.5": {"tn": int(tn), "fp": int(fp), "fn": int(fn),
                             "tp": int(tp)},
        "cost_at_0.5": c,
        "cost_fn": COST_FN, "cost_fp": COST_FP,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

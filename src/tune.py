"""Day 54: gradient boosting + cost-tuned threshold (never 0.5 by default).

- 64/16/20 train/val/test split (stratified): threshold tuned on VAL cost,
  final numbers reported on TEST (tuning on test would overfit the threshold).
- Winner = lowest val cost; threshold = argmin val cost over a grid.
Output: models/xgb.joblib, models/lgbm.joblib + models/tuning_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline import build_preprocessor
from costs import cost_at_threshold
from features import LABEL, build_features, feature_columns

GRID = [round(t, 2) for t in np.arange(0.05, 0.91, 0.05)]
SEED = 42


def best_threshold(y_val, proba):
    """Argmin validation cost over the grid (ties -> higher threshold)."""
    scored = [(t, cost_at_threshold(y_val, proba, t)["total"]) for t in GRID]
    best_cost = min(c for _, c in scored)
    return max(t for t, c in scored if c == best_cost)


def main():
    t0 = time.time()
    df = build_features()
    cols = feature_columns(df)
    X, y = df[cols], df[LABEL]
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.36, random_state=SEED, stratify=y)
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=5 / 9, random_state=SEED, stratify=ytmp)

    prep = build_preprocessor(Xtr)
    Xtr_p, Xva_p, Xte_p = prep.fit_transform(Xtr), prep.transform(Xva), prep.transform(Xte)
    pos_w = float((ytr == 0).sum() / (ytr == 1).sum())

    models = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, n_jobs=-1, random_state=SEED,
            eval_metric="logloss"),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1),
    }
    out = {"grid": GRID, "models": {}}
    for name, clf in models.items():
        # Serving needs the full chain (preprocessor + classifier): fit the
        # bare estimator for metrics, then refit an identical pipeline whose
        # artifact the API loads (same data/seed/config -> same model).
        clf.fit(Xtr_p, ytr)
        pipe = Pipeline([("prep", build_preprocessor(Xtr)), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        pv, pt = clf.predict_proba(Xva_p)[:, 1], clf.predict_proba(Xte_p)[:, 1]
        thr = best_threshold(yva, pv)
        val_cost = cost_at_threshold(yva, pv, thr)["total"]
        test_cost = cost_at_threshold(yte, pt, thr)
        out["models"][name] = {
            "roc_auc": round(float(roc_auc_score(yte, pt)), 4),
            "pr_auc": round(float(average_precision_score(yte, pt)), 4),
            "best_threshold": thr,
            "val_cost": val_cost,
            "test_cost": test_cost,
        }
        joblib.dump(clf, PROJECT_ROOT / "models" / f"{name}.joblib")
        joblib.dump(pipe, PROJECT_ROOT / "models" / f"{name}_pipeline.joblib")
        print(f"{name}: ROC={out['models'][name]['roc_auc']} "
              f"thr={thr} val_cost=${val_cost:,.0f} test_cost=${test_cost['total']:,.0f}",
              flush=True)
    winner = min(out["models"], key=lambda m: out["models"][m]["val_cost"])
    out["winner"] = winner
    out["elapsed_s"] = round(time.time() - t0, 1)
    (PROJECT_ROOT / "models" / "tuning_metrics.json").write_text(
        json.dumps(out, indent=2))
    print(f"winner: {winner} | saved models + tuning_metrics.json")


if __name__ == "__main__":
    main()

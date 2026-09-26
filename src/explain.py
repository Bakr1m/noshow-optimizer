"""Day 56: SHAP explainability for the winning XGBoost model.

TreeExplainer on a 500-row test sample (seeded); saves the summary plot and
top-15 mean-|SHAP| features. Answers: what drives flags, in business terms.
Output: docs/shap_summary.png + models/shap_top.json
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline import build_preprocessor
from features import LABEL, build_features, feature_columns

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"
PLOT_PATH = PROJECT_ROOT / "docs" / "shap_summary.png"
OUT_PATH = PROJECT_ROOT / "models" / "shap_top.json"
SEED = 42
SAMPLE = 500


def main():
    df = build_features()
    cols = feature_columns(df)
    X, y = df[cols], df[LABEL]
    Xtr, _, _, _ = train_test_split(
        X, y, test_size=0.36, random_state=SEED, stratify=y)
    prep = build_preprocessor(Xtr)
    Xtr_p = prep.fit_transform(Xtr)
    try:
        names = prep.get_feature_names_out().tolist()
    except Exception:  # noqa: BLE001 - older sklearn fallback
        names = [f"f{i}" for i in range(Xtr_p.shape[1])]
    rng = np.random.RandomState(SEED)
    n_rows = Xtr_p.shape[0]
    idx = rng.choice(n_rows, min(SAMPLE, n_rows), replace=False)
    Xs = Xtr_p[idx]
    if hasattr(Xs, "toarray"):
        Xs = Xs.toarray()
    clf = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xs)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    top = [{"feature": names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
           for i in order]
    OUT_PATH.write_text(json.dumps(top, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([t["feature"] for t in top][::-1],
            [t["mean_abs_shap"] for t in top][::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Top drivers of no-show flags (XGBoost, 500-row sample)")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(top[:8], indent=2))
    print(f"saved -> {PLOT_PATH}, {OUT_PATH}")


if __name__ == "__main__":
    main()

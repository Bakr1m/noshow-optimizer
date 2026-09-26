"""Day 55: reminder-intervention simulation on the held-out test set.

Policy: flag proba >= threshold -> send a $1 extra reminder. Flagged
would-be no-shows convert to shows with probability EFFECTIVENESS
(stated assumption to validate in a pilot, not a measured fact).
Compares: do-nothing, remind-flagged (per threshold), remind-everyone.

Output: models/simulation.json + docs/simulation.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline import build_preprocessor
from costs import COST_FN
from features import LABEL, build_features, feature_columns

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost.joblib"  # Day-54 val-cost winner
OUT_PATH = PROJECT_ROOT / "models" / "simulation.json"
PLOT_PATH = PROJECT_ROOT / "docs" / "simulation.png"

SMS_COST = 1.0
EFFECTIVENESS = 0.3  # assumption: reminder averts 30% of flagged no-shows
SEED = 42


def simulate(y_true, proba, threshold, sms_cost=SMS_COST,
             effectiveness=EFFECTIVENESS, cost_fn=COST_FN, seed=SEED):
    """Deterministic-ish intervention accounting (seeded conversion draws)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    tp_flag = flagged & (y_true == 1)
    converted = tp_flag & (rng.rand(len(y_true)) < effectiveness)
    prevented = int(converted.sum())
    missed = int(((y_true == 1) & ~converted).sum())
    sms_spend = round(float(flagged.sum() * sms_cost), 2)
    return {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "prevented": prevented,
        "missed": missed,
        "sms_spend": sms_spend,
        "miss_cost": round(missed * cost_fn, 2),
        "total": round(missed * cost_fn + sms_spend, 2),
    }


def main():
    df = build_features()
    cols = feature_columns(df)
    X, y = df[cols], df[LABEL]
    Xtr, Xtmp, _, ytmp = train_test_split(
        X, y, test_size=0.36, random_state=SEED, stratify=y)
    _, Xte, _, yte = train_test_split(
        Xtmp, ytmp, test_size=5 / 9, random_state=SEED, stratify=ytmp)
    # Preprocessor fits on TRAIN only (same split/seed as tune.py).
    prep = build_preprocessor(Xtr)
    prep.fit(Xtr)
    Xte_p = prep.transform(Xte)
    proba = joblib.load(MODEL_PATH).predict_proba(Xte_p)[:, 1]
    yte = np.asarray(yte)

    baseline = {"policy": "do-nothing",
                "total": round(float((yte == 1).sum() * COST_FN), 2)}
    rows = [simulate(yte, proba, t) for t in
            (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)]
    rows.append({**simulate(yte, np.ones_like(proba), 0.0),
                 "threshold": "remind-everyone"})
    best = min((r for r in rows if isinstance(r["threshold"], float)),
               key=lambda r: r["total"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [r["threshold"] for r in rows if isinstance(r["threshold"], float)]
    ax.plot(xs, [r["total"] for r in rows if isinstance(r["threshold"], float)],
            marker="o", label="remind-flagged total cost")
    ax.axhline(baseline["total"], color="red", linestyle="--", label="do-nothing")
    ax.set_xlabel("flag threshold")
    ax.set_ylabel("test-set cost ($)")
    ax.set_title("Reminder policy cost vs threshold (effectiveness=0.3 assumed)")
    ax.legend()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=100, bbox_inches="tight")
    plt.close(fig)

    out = {"baseline": baseline, "policies": rows, "best": best,
           "assumptions": {"sms_cost": SMS_COST, "effectiveness": EFFECTIVENESS,
                           "miss_cost": COST_FN}}
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps({"baseline": baseline, "best": best}, indent=2))
    print(f"saved -> {OUT_PATH}, {PLOT_PATH}")


if __name__ == "__main__":
    main()

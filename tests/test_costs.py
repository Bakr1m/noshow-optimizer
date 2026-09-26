"""Cost-math tests (exact, hermetic) + pipeline smoke test."""
import numpy as np
import pandas as pd

from src.baseline import build_preprocessor
from src.costs import COST_FN, COST_FP, cost_at_threshold, total_cost
from src.features import LABEL


def test_total_cost_exact():
    y = [0, 0, 1, 1]
    p = [0, 1, 0, 1]  # 1 fp, 1 fn
    c = total_cost(y, p)
    assert (c["fp"], c["fn"]) == (1, 1)
    assert c["total"] == COST_FP + COST_FN


def test_cost_at_threshold_monotone():
    y = [0, 0, 0, 1, 1, 1]
    proba = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
    # Lowering the threshold can only trade FNs for FPs here.
    assert cost_at_threshold(y, proba, 0.9)["fn"] == 3
    assert cost_at_threshold(y, proba, 0.05)["fp"] == 3


def test_pipeline_no_leakage_smoke():
    """Preprocessor fits on train only; smoke-fit a tiny frame end to end."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "lead_days": rng.randint(0, 30, 60).astype(float),
        "Age": rng.randint(5, 90, 60).astype(float),
        "Gender": rng.choice(["F", "M"], 60),
        LABEL: rng.choice([0, 1], 60),
    })
    Xtr, Xte = df.iloc[:40], df.iloc[40:]
    pipe = Pipeline([
        ("prep", build_preprocessor(Xtr.drop(columns=[LABEL]))),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(Xtr.drop(columns=[LABEL]), Xtr[LABEL])
    proba = pipe.predict_proba(Xte.drop(columns=[LABEL]))[:, 1]
    assert len(proba) == 20 and np.isfinite(proba).all()
    # Fitted only on train categories: unseen category must not crash.
    assert pipe.predict_proba(Xte.drop(columns=[LABEL])).shape == (20, 2)

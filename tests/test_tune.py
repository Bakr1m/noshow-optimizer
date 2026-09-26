"""Threshold-tuning tests (exact on synthetic scores, hermetic)."""
import numpy as np

from src.tune import GRID, best_threshold


def test_best_threshold_minimizes_cost():
    y = np.array([0] * 90 + [1] * 10)
    # Perfect ranking: scores separate classes; best = just below min positive.
    proba = np.array([0.1] * 90 + [0.9] * 10)
    t = best_threshold(y, proba)
    assert t == max(g for g in GRID if g <= 0.9)
    from src.costs import cost_at_threshold

    assert cost_at_threshold(y, proba, t)["total"] == 0.0  # perfect split, no errors


def test_best_threshold_beats_default_when_asymmetric():
    rng = np.random.RandomState(0)
    y = (rng.rand(1000) < 0.2).astype(int)
    proba = np.clip(rng.rand(1000) * 0.4 + y * 0.4, 0, 1)
    from src.costs import cost_at_threshold

    t = best_threshold(y, proba)
    assert t < 0.5  # 30:1 asymmetry must pull the threshold down
    assert cost_at_threshold(y, proba, t)["total"] <= cost_at_threshold(
        y, proba, 0.5)["total"]

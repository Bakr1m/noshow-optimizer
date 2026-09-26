"""Simulation accounting tests (exact arithmetic on synthetic outcomes)."""
import numpy as np

from src.costs import COST_FN
from src.simulate import SMS_COST, simulate


def test_simulate_exact_bookkeeping():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.9, 0.1, 0.9, 0.9, 0.1])  # flags rows 0, 2, 3
    r = simulate(y, proba, 0.5, seed=0)
    assert r["flagged"] == 3
    assert r["sms_spend"] == 3 * SMS_COST
    # Seeded draws decide conversion; total must reconcile exactly.
    assert r["total"] == round(r["missed"] * COST_FN + r["sms_spend"], 2)
    assert r["prevented"] + r["missed"] == 3  # all positives accounted for


def test_threshold_zero_flags_everyone():
    y = np.array([0, 1, 0, 1])
    r = simulate(y, np.array([0.2, 0.3, 0.1, 0.4]), 0.0, seed=1)
    assert r["flagged"] == 4
    assert r["prevented"] + r["missed"] == 2


def test_perfect_model_beats_nothing():
    y = np.array([0] * 80 + [1] * 20)
    proba = np.array([0.05] * 80 + [0.95] * 20)
    base = 20 * COST_FN
    r = simulate(y, proba, 0.5, seed=2)
    assert r["total"] < base  # reminders must save money here
    assert r["prevented"] > 0

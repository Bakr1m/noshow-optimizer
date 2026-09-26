"""Day 53: business cost framing (explicit, documented assumptions).

Intervention: flag high-risk appointments for extra reminders + standby
overbooking. Costs in USD per appointment outcome (Brazilian public-system
scale, stated as assumptions for stakeholders to challenge):
- False negative (missed no-show -> idle room + clinician time): $150.
- False positive (unnecessary reminder + overbooking friction): $5.
Ratio 30:1 — missing a no-show dwarfs a wasted nudge, so the optimal
threshold will sit far below 0.5. Day 54 tunes on this; Day 55 simulates it.
"""
COST_FN = 150.0
COST_FP = 5.0


def total_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    """Business cost of a hard prediction vector (lower is better)."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return {"fn": fn, "fp": fp, "total": fn * cost_fn + fp * cost_fp}


def cost_at_threshold(y_true, proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    import numpy as np

    return total_cost(y_true, (np.asarray(proba) >= threshold).astype(int),
                      cost_fn, cost_fp)

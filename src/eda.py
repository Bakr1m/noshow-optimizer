"""Day 51 EDA: what correlates with no-shows?

Loads the raw CSV, parses dates, builds lead time + prior-history rate,
and measures every candidate factor against the target.
Output: models/eda_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_PATH = PROJECT_ROOT / "data" / "noshow_raw.csv"
OUT_PATH = PROJECT_ROOT / "models" / "eda_summary.json"


def clean_frame(df):
    """Parse dates, binary target, drop impossible ages, add lead time."""
    df = df.copy()
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
    df["no_show"] = (df["No-show"] == "Yes").astype(int)
    df = df[df["Age"] >= 0].copy()  # negative ages are data-entry errors
    df["lead_days"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(
        lower=0
    )
    df["appt_dow"] = df["AppointmentDay"].dt.day_name()
    return df.reset_index(drop=True)


def load_clean(path=DATA_PATH):
    return clean_frame(pd.read_csv(path))


def prior_rate(df):
    """Per-appointment prior no-show rate for that patient (past only)."""
    df = df.sort_values(["PatientId", "AppointmentDay"])
    g = df.groupby("PatientId")["no_show"]
    return (g.cumsum().shift(1, fill_value=0) / g.cumcount().replace(0, np.nan)).fillna(
        0.0
    )


def point_biserial(x, y):
    """Correlation of a numeric feature with the binary target."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def rate_by(df, col, min_n=200):
    return (
        df.groupby(col)["no_show"]
        .agg(["mean", "count"])
        .query("count >= @min_n")
        .sort_values("mean", ascending=False)
    )


def main():
    df = load_clean()
    df["prior_rate"] = prior_rate(df).to_numpy()
    s = {
        "n_rows": len(df),
        "overall_rate": round(float(df["no_show"].mean()), 4),
        "by_sms": rate_by(df, "SMS_received", 1).to_dict(),
        "by_gender": rate_by(df, "Gender", 1).to_dict(),
        "by_dow": rate_by(df, "appt_dow", 1).to_dict(),
        "corr": {
            "lead_days": round(point_biserial(df["lead_days"], df["no_show"]), 4),
            "prior_rate": round(point_biserial(df["prior_rate"], df["no_show"]), 4),
            "age": round(point_biserial(df["Age"], df["no_show"]), 4),
            "sms": round(point_biserial(df["SMS_received"], df["no_show"]), 4),
        },
        "age_bins": df.assign(
            age_bin=pd.cut(df["Age"], [0, 18, 35, 55, 75, 200])
        )
        .groupby("age_bin", observed=True)["no_show"]
        .mean()
        .round(4)
        .astype(float)
        .to_dict(),
        "top_neighbourhoods": rate_by(df, "Neighbourhood").head(3).to_dict(),
        "bottom_neighbourhoods": rate_by(df, "Neighbourhood").tail(3).to_dict(),
    }
    s["age_bins"] = {str(k): v for k, v in s["age_bins"].items()}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "by_dow"}, indent=2,
                     default=str)[:1500])
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Day 52: feature engineering for no-show prediction.

- lead_days: booking-to-appointment gap (clipped >= 0; scheduled-after-
  appointment is a data-entry artifact, not time travel).
- prior_rate / prior_visits: per-patient history, strictly past-only
  (expanding mean shifted by one; first visit -> 0.0/0).
- dow / is_weekend / month: calendar effects (work/school/transport rhythms).
- age_group: coarse life-stage bins (scheduling behavior differs by stage).

Output: data/noshow_features.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eda import clean_frame

DATA_PATH = PROJECT_ROOT / "data" / "noshow_raw.csv"
OUT_PATH = PROJECT_ROOT / "data" / "noshow_features.parquet"

LABEL = "no_show"


def add_history(df):
    """Past-only prior rate + visit count per patient."""
    df = df.sort_values(["PatientId", "AppointmentDay"]).reset_index(drop=True)
    g = df.groupby("PatientId")["no_show"]
    df["prior_rate"] = (
        g.cumsum().shift(1, fill_value=0) / g.cumcount().replace(0, np.nan)
    ).fillna(0.0)
    df["prior_visits"] = g.cumcount()
    return df


def add_calendar(df):
    df["dow"] = df["AppointmentDay"].dt.day_name()
    df["is_weekend"] = (df["AppointmentDay"].dt.dayofweek >= 5).astype(int)
    df["month"] = df["AppointmentDay"].dt.month
    return df


def add_age_group(df):
    df["age_group"] = pd.cut(
        df["Age"], bins=[0, 18, 35, 55, 75, 200],
        labels=["child", "young", "mid", "senior", "elder"],
    )
    return df


def build_features(path=DATA_PATH):
    df = add_age_group(add_calendar(add_history(clean_frame(pd.read_csv(path)))))
    return df


def feature_columns(df):
    exclude = {"no_show", "No-show", "PatientId", "AppointmentID",
               "ScheduledDay", "AppointmentDay", "appt_dow"}
    return [c for c in df.columns if c not in exclude]


def main():
    df = build_features()
    df.to_parquet(OUT_PATH, index=False)
    feats = feature_columns(df)
    print(f"rows: {len(df)} | features: {len(feats)}")
    print("engineered:", ["lead_days", "prior_rate", "prior_visits", "dow",
                          "is_weekend", "month", "age_group"])
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

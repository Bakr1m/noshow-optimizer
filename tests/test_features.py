"""Hermetic feature tests (synthetic frames, no data/ needed)."""
import pandas as pd

from src.features import add_age_group, add_calendar, add_history, feature_columns


def tiny_frame():
    base = {
        "PatientId": [1, 1, 2],
        "AppointmentID": [10, 11, 20],
        "Gender": ["F", "F", "M"],
        "ScheduledDay": pd.to_datetime(
            ["2016-04-29T18:38:08Z"] * 3, utc=True
        ),
        "AppointmentDay": pd.to_datetime(
            ["2016-04-29", "2016-05-06", "2016-04-30"], utc=True
        ),
        "Age": [30, 30, 70],
        "Neighbourhood": ["A", "A", "B"],
        "Scholarship": [0, 0, 1],
        "Hipertension": [0, 0, 0],
        "Diabetes": [0, 0, 0],
        "Alcoholism": [0, 0, 0],
        "Handcap": [0, 0, 0],
        "SMS_received": [0, 1, 0],
        "No-show": ["No", "Yes", "No"],
        "no_show": [0, 1, 0],
        "lead_days": [0, 7, 1],
    }
    return pd.DataFrame(base)


def test_history_past_only_and_counts():
    df = add_history(tiny_frame())
    p1 = df[df["PatientId"] == 1].sort_values("AppointmentDay")
    assert p1["prior_rate"].tolist() == [0.0, 0.0]  # history [No] -> 0.0
    assert p1["prior_visits"].tolist() == [0, 1]
    assert df[df["PatientId"] == 2]["prior_visits"].tolist() == [0]


def test_calendar_and_weekend():
    df = add_calendar(tiny_frame())
    assert df.loc[0, "dow"] == "Friday"  # 2016-04-29 was a Friday
    assert df.loc[2, "dow"] == "Saturday"  # 2016-04-30 was a Saturday
    assert df["is_weekend"].tolist() == [0, 0, 1]
    assert df["month"].tolist() == [4, 5, 4]


def test_age_groups_and_column_contract():
    df = add_age_group(add_calendar(add_history(tiny_frame())))
    assert df["age_group"].tolist() == ["young", "young", "senior"]
    cols = feature_columns(df)
    for banned in ("no_show", "No-show", "PatientId", "AppointmentID",
                   "ScheduledDay", "AppointmentDay"):
        assert banned not in cols
    for kept in ("lead_days", "prior_rate", "prior_visits", "dow",
                 "is_weekend", "month", "age_group"):
        assert kept in cols

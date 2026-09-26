"""Hermetic EDA helper tests (synthetic frames, no data/ needed)."""
import pandas as pd

from src.eda import clean_frame, point_biserial, prior_rate


def tiny_frame():
    return pd.DataFrame(
        {
            "PatientId": [1, 1, 2, 2],
            "AppointmentID": [10, 11, 20, 21],
            "Gender": ["F", "F", "M", "M"],
            "ScheduledDay": ["2016-04-29T18:38:08Z"] * 4,
            "AppointmentDay": [
                "2016-04-29T00:00:00Z",
                "2016-05-06T00:00:00Z",
                "2016-04-29T00:00:00Z",
                "2016-05-06T00:00:00Z",
            ],
            "Age": [30, 30, 40, 40],
            "Neighbourhood": ["A", "A", "B", "B"],
            "Scholarship": [0, 0, 1, 1],
            "Hipertension": [0, 0, 0, 0],
            "Diabetes": [0, 0, 0, 0],
            "Alcoholism": [0, 0, 0, 0],
            "Handcap": [0, 0, 0, 0],
            "SMS_received": [0, 1, 0, 1],
            "No-show": ["No", "Yes", "No", "No"],
        }
    )


def test_load_clean_target_and_lead():
    df = clean_frame(tiny_frame())
    assert df["no_show"].tolist() == [0, 1, 0, 0]
    # .dt.days floors: 2016-04-29T18:38 -> 2016-05-06T00:00 is 6 full days.
    assert df["lead_days"].tolist() == [0, 6, 0, 6]


def test_prior_rate_past_only():
    df = clean_frame(tiny_frame())
    pr = prior_rate(df)
    # Patient 1: row0 first appt -> 0.0; row1 sees one prior No (0.0 rate).
    # Patient 2: no prior shows -> 0.0.
    assert pr.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_prior_rate_reflects_history():
    df = clean_frame(tiny_frame())
    df.loc[1, "no_show"] = 1  # already 1; add a third appt for patient 1
    extra = df.iloc[[1]].copy()
    extra["AppointmentDay"] += pd.Timedelta(days=30)
    df2 = pd.concat([df, extra], ignore_index=True)
    pr = prior_rate(df2)
    # Patient 1's third row sees history [No, Yes] -> rate 0.5.
    # (Locate by key: prior_rate sorts internally, so positional iloc is wrong.)
    latest_p1 = df2[df2["PatientId"] == 1].sort_values("AppointmentDay").index[-1]
    assert pr.loc[latest_p1] == 0.5


def test_point_biserial_sign():
    x = [1, 2, 3, 10, 11, 12]
    y = [0, 0, 0, 1, 1, 1]
    assert point_biserial(x, y) > 0.8
    assert point_biserial([1, 2, 3, 4], [1, 0, 1, 0]) < 0.5

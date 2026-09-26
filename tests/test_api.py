"""API contract tests: validation + flag consistency (stand-in pipeline).

Hermetic: a tiny LR pipeline stands in for xgboost.joblib (gitignored), so no
test touches disk artifacts. Production wiring verified by Docker parity.
"""
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import serve
from src.serve import THRESHOLD, app


def tiny_pipeline():
    import numpy as np
    import pandas as pd

    from src.baseline import build_preprocessor

    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "lead_days": rng.randint(0, 30, 60).astype(float),
        "Age": rng.randint(5, 90, 60).astype(float),
        "Gender": rng.choice(["F", "M"], 60),
        "prior_rate": rng.rand(60),
        "prior_visits": rng.randint(0, 5, 60).astype(float),
        "dow": rng.choice(["Monday", "Saturday"], 60),
        "is_weekend": rng.choice([0, 1], 60),
        "month": rng.randint(1, 13, 60),
        "age_group": rng.choice(["young", "senior"], 60),
        "Neighbourhood": rng.choice(["A", "B"], 60),
        "Scholarship": rng.choice([0, 1], 60),
        "Hipertension": rng.choice([0, 1], 60),
        "Diabetes": rng.choice([0, 1], 60),
        "Alcoholism": rng.choice([0, 1], 60),
        "Handcap": rng.choice([0, 1], 60),
        "SMS_received": rng.choice([0, 1], 60),
    })
    y = (df["lead_days"] > 10).astype(int)
    pipe = Pipeline([
        ("prep", build_preprocessor(df)),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(df, y)
    return pipe


@pytest.fixture(autouse=True)
def stand_in_pipeline():
    serve._pipe = tiny_pipeline()
    yield
    serve._pipe = None


client = TestClient(app)

PAYLOAD = {
    "Gender": "F",
    "ScheduledDay": "2016-04-29T18:38:08Z",
    "AppointmentDay": "2016-05-06T00:00:00Z",
    "Age": 30,
    "Neighbourhood": "A",
    "Scholarship": 0,
    "Hipertension": 0,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handcap": 0,
    "SMS_received": 1,
    "prior_rate": 0.0,
    "prior_visits": 1,
}


def test_health():
    assert client.get("/health").json() == {"status": "healthy"}


def test_predict_risk_and_flag_consistent():
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["no_show_risk"] <= 1.0
    assert body["flag"] == (body["no_show_risk"] >= THRESHOLD)
    assert body["threshold"] == THRESHOLD


def test_predict_rejects_bad_input():
    bad = dict(PAYLOAD)
    del bad["Age"]
    assert client.post("/predict", json=bad).status_code == 422
    bad2 = dict(PAYLOAD, Scholarship=2)
    assert client.post("/predict", json=bad2).status_code == 422

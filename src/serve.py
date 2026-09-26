"""Day 57: stateless no-show scoring API.

Client sends appointment facts + known history stats (EHR-side); the server
derives lead/calendar/age features and scores with the cost-tuned XGBoost
pipeline. No server state, lazy model load (import never touches disk).

POST /predict -> {"no_show_risk", "flag", "threshold"} (flag at 0.15).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import add_age_group, add_calendar

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_pipeline.joblib"
THRESHOLD = 0.15  # Day-54 cost-tuned operating point

app = FastAPI(title="No-Show Risk API")

_pipe = None


def get_pipeline():
    """Load once on first request (never at import: hermetic tests)."""
    global _pipe
    if _pipe is None:
        _pipe = joblib.load(MODEL_PATH)
    return _pipe


class AppointmentInput(BaseModel):
    """Appointment facts; history stats come from the EHR, not server memory."""

    Gender: str
    ScheduledDay: str
    AppointmentDay: str
    Age: float
    Neighbourhood: str
    Scholarship: int = Field(ge=0, le=1)
    Hipertension: int = Field(ge=0, le=1)
    Diabetes: int = Field(ge=0, le=1)
    Alcoholism: int = Field(ge=0, le=1)
    Handcap: int = Field(ge=0, le=1)
    SMS_received: int = Field(ge=0, le=1)
    prior_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    prior_visits: int = Field(default=0, ge=0)


class PredictionResponse(BaseModel):
    no_show_risk: float
    flag: bool
    threshold: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(appt: AppointmentInput):
    try:
        df = pd.DataFrame([appt.model_dump()])
        df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True)
        df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True)
        df["lead_days"] = (
            df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(lower=0)
        df = add_age_group(add_calendar(df))
        risk = float(get_pipeline().predict_proba(df)[:, 1][0])
        return PredictionResponse(
            no_show_risk=risk, flag=risk >= THRESHOLD, threshold=THRESHOLD)
    except Exception as e:  # noqa: BLE001 - scoring failures -> 400, never 500
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

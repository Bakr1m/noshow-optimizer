# Project 5: Patient No-Show & Hospital Ops Optimizer

**Days 51–60 | Healthcare ML Portfolio**

## Business Context

Missed appointments cost clinics real money (idle rooms, clinician time) and
disrupt care continuity. Predicting no-show risk lets clinics send targeted
reminders and overbook intelligently. This project is framed around quantified
business impact: every modeling choice is priced in dollars, not just scored
in metrics.

## Dataset

- **Source**: Kaggle "Medical Appointment No Shows" mirror (raw GitHub CSV,
  verified fingerprints: 110,527 rows × 14 cols, 20.19% no-show)
- **Target**: `no_show` — patient missed the appointment (22,319 / 20.2%)
- **Key quirks**: SMS correlates *positively* (confounded outreach), lead time
  monotonic, 48k repeat-patient rows, negative-age entry errors removed
- Reproduce: download mirror CSV to `data/noshow_raw.csv` (see notebook 01)

## Approach

1. **EDA** (Day 51): target-first screening — lead +0.18 dominates.
2. **Features** (Day 52): past-only history, calendar, age groups; IDs out.
3. **Baseline + costs** (Day 53): balanced LR; $150 FN / $5 FP (30:1) fixed
   up front as documented stakeholder assumptions.
4. **Boosting + tuned threshold** (Day 54): XGBoost vs LightGBM on 64/16/20;
   argmin *validation* cost; winner XGBoost on val $.
5. **Simulation** (Day 55): $1 reminders at 30% effectiveness on held-out test.
6. **Tracking + SHAP** (Day 56): 4 MLflow runs; age/lead/place drivers.
7. **Serving + container** (Day 57): stateless `/predict`, 1.49 GB image,
   bit-identical parity.

## Results

| Model | ROC-AUC | PR-AUC | Test cost |
|-------|---------|--------|-----------|
| LR baseline @0.5 | 0.6817 | 0.3333 | $301,370 |
| XGBoost @0.15 | **0.7339** | **0.3977** | **$90,135** |
| LightGBM @0.10 | 0.7310 | 0.3964 | $88,275 |

- **Reminder simulation**: best policy $492k vs $670k do-nothing (**$178k /
  26% saved**, 1,330 prevented) — but targeting ≈ remind-everyone (honest).
- **Operating point**: 0.15 (recall-first under 30:1 asymmetry).

## Limitations

1. **Effectiveness assumed (0.3)**: all savings hinge on it — pilot must measure.
2. **Single-city public data (2016 Brazil)**: transport, SMS habits, and costs
   differ elsewhere; cost figures are stakeholder assumptions, not invoices.
3. **Observational confounding** (SMS targeting) limits causal claims about
   reminders themselves.
4. **Retrospective only**; no-shows are partly structural (transport, work) —
   prediction ≠ access.

## Ethical Considerations

- **Targeting must not punish**: risk scores allocate reminders/overbooking,
  never deny care; audit flags across neighbourhood/age for disparate impact.
- **Place-based features** (neighbourhood) can proxy socioeconomic status —
  disclose and monitor, don't launder bias into operations.
- **Transparency**: costs, assumptions, and the blanket-reminder comparison
  are published alongside the savings claim.

## Project Structure

```
project5_noshow/
├ data/            # raw CSV + features parquet (gitignored)
├ notebooks/       # 01_eda ... 08_polish (all execute clean)
├ src/             # eda, features, costs, baseline, tune, simulate,
│                  #   backfill_mlflow, explain, serve
├ api/main.py      # thin entrypoint
├ tests/           # 19 hermetic tests (synthetic frames only)
├ models/          # *.joblib/*.json (gitignored)
├ mlruns/          # noshow_ops tracking (gitignored)
├ docs/            # simulation.png, shap_summary.png
├ Dockerfile (1.49 GB) + requirements-serve.txt
├ requirements.txt (full) / requirements-train.txt
└ README.md
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test     # hermetic, no data/ needed
# training needs data/: python src/features.py, baseline.py, tune.py, ...
python api/main.py   # :8000
```

## Run with Docker

```bash
docker pull bakr1m/noshow-api:v1
docker run -p 8000:8000 bakr1m/noshow-api:v1
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"Gender":"F", ...}'
```

## Key Learnings

1. **Costs before thresholds** — 30:1 fixed first, 0.15 falls out naturally.
2. **Threshold beats architecture** ($301k→$90k vs model deltas of 2%).
3. **Observational data confounds** (SMS) — models learn it, policies must not.
4. **Simulations need stated assumptions** — effectiveness runs the table.
5. **Honest negatives close interviews** (targeting ≈ blanket).

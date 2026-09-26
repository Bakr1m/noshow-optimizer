# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-26

### Added
- Cost-framed XGBoost no-show classifier (ROC-AUC 0.734, cost-tuned @0.15).
- Reminder-intervention simulation ($178k / 26% saved, assumptions stated).
- SHAP explainability (age/lead/place drivers) + MLflow `noshow_ops` runs.
- Stateless FastAPI `/predict` + CPU-only Docker image, parity-verified.
- 19 hermetic pytest tests; ruff-clean; executing notebooks.

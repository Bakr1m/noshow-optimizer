FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/xgboost_pipeline.joblib ./models/xgboost_pipeline.joblib

EXPOSE 8000

CMD ["python", "api/main.py"]

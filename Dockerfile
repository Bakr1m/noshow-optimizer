FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installed in three small layers (not one): a single 831 MB layer
# repeatedly broke mid-upload on thin uplinks; each blob below pushes alone.
# Installed one package per layer (not one RUN): single blobs over ~300 MB
# repeatedly die mid-upload on thin uplinks. Pins mirror
# requirements-serve-base.txt (the manifest); keep both in sync.
COPY requirements-serve*.txt ./
RUN pip install --no-cache-dir numpy==2.5.3
RUN pip install --no-cache-dir "pandas==3.0.6"
RUN pip install --no-cache-dir "scipy==1.18.1"
RUN pip install --no-cache-dir --no-deps -r requirements-serve-ml.txt
RUN pip install --no-cache-dir -r requirements-serve-api.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/xgboost_pipeline.joblib ./models/xgboost_pipeline.joblib

EXPOSE 8000

CMD ["python", "api/main.py"]

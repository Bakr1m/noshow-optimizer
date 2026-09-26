FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
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

# Production model: fetched by exact release version, SHA256-verified.
# Never a moving tag, never baked from a developer laptop. Provenance:
# https://github.com/Bakr1m/noshow-optimizer/releases/tag/v1.0.0
ARG MODEL_TAG=v1.0.0
ARG MODEL_SHA256=fb453a00b42863fa36eecf64931c70f4eb92e0fc0ae9a06d3d571ca58d2c1942
RUN mkdir -p models && \
    curl -fsSL -o models/xgboost_pipeline.joblib \
      "https://github.com/Bakr1m/noshow-optimizer/releases/download/${MODEL_TAG}/xgboost_pipeline.joblib" && \
    echo "${MODEL_SHA256}  models/xgboost_pipeline.joblib" | sha256sum -c - && \
    python -c "import joblib; m=joblib.load('models/xgboost_pipeline.joblib'); print('artifact OK:', type(m).__name__)"

EXPOSE 8000

CMD ["python", "api/main.py"]

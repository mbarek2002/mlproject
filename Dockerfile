# Same Python minor version as the venv the model was trained with (pickles depend on it)
FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# serving dependencies only (no xgboost / catboost / plotting libs), installed first:
# this layer is cached as long as requirements-prod.txt does not change
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY app.py .
COPY src/ src/
COPY templates/ templates/
COPY artifacts/model.pkl artifacts/preprocessor.pkl artifacts/

# fail the build now rather than at the first request if the model needs a missing library
RUN python -c "from src.pipeline.predict_pipeline import PredictPipeline; PredictPipeline.load_model()" \
    && rm -rf logs/*

# run as a non-root user, allowed to write the logs/ folder only
RUN useradd --create-home appuser \
    && mkdir -p logs \
    && chown appuser:appuser logs
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"

# waitress: production WSGI server (the Flask built-in server is for development only)
CMD ["python", "-m", "waitress", "--host=0.0.0.0", "--port=5000", "app:app"]

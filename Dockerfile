# Same Python minor version as the venv the model was trained with (pickles depend on it)
FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# tini: tiny init process (PID 1) forwarding stop signals, so the app shuts down immediately
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

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

# listening port: 5000 by default, overridden by the platform (Render sets PORT=10000)
ENV PORT=5000
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)"

# waitress: production WSGI server (the Flask built-in server is for development only)
# sh -c to expand $PORT, exec so waitress replaces the shell and receives the signals forwarded by tini
ENTRYPOINT ["tini", "--"]
CMD ["sh", "-c", "exec python -m waitress --host=0.0.0.0 --port=$PORT app:app"]

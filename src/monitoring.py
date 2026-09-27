'''
Prometheus metrics exposed on GET /metrics:

- service: request count / latency per route and status
- model:   which model is served, load time, prediction count / latency
- ML:      distribution of the inputs and of the predicted scores (data drift signals)
'''
import hmac
import os
import time

from flask import request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

SCORE_BUCKETS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
CATEGORICAL_FEATURES = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

# ---------------- service ----------------
HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

# ---------------- model ----------------
MODEL_INFO = Gauge(
    "model_info", "Model currently served (value is always 1)", ["source", "version"]
)
MODEL_LOAD_SECONDS = Gauge("model_load_seconds", "Duration of the last model load")
PREDICTIONS = Counter("predictions_total", "Successful predictions")
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds", "Model inference latency (includes the first model load)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 5, 15, 30),
)
INVALID_REQUESTS = Counter("invalid_requests_total", "Rejected prediction forms", ["reason"])

# ---------------- ML: data drift signals ----------------
PREDICTED_SCORE = Histogram("predicted_math_score", "Predicted math scores", buckets=SCORE_BUCKETS)
INPUT_SCORE = Histogram("input_score", "Input scores sent to the model", ["feature"], buckets=SCORE_BUCKETS)
INPUT_CATEGORY = Counter("input_category_total", "Categorical inputs sent to the model", ["feature", "value"])


def set_model_info(info, load_seconds=None):
    MODEL_INFO.clear()
    MODEL_INFO.labels(source=info.get("source") or "none", version=info.get("version") or "none").set(1)
    if load_seconds is not None:
        MODEL_LOAD_SECONDS.set(load_seconds)


def observe_prediction(features, prediction, duration):
    PREDICTIONS.inc()
    PREDICTION_LATENCY.observe(duration)
    PREDICTED_SCORE.observe(prediction)
    row = features.iloc[0]
    for feature in ["reading_score", "writing_score"]:
        INPUT_SCORE.labels(feature=feature).observe(float(row[feature]))
    for feature in CATEGORICAL_FEATURES:
        INPUT_CATEGORY.labels(feature=feature, value=str(row[feature])).inc()


def observe_invalid(error_message):
    # fixed, low-cardinality reasons (never the raw message)
    if error_message.startswith("Missing"):
        reason = "missing_field"
    elif "must be a number" in error_message:
        reason = "not_a_number"
    else:
        reason = "out_of_range"
    INVALID_REQUESTS.labels(reason=reason).inc()


def init_app(app):
    '''
    Records every request (except /metrics itself) and adds GET /metrics
    '''
    @app.before_request
    def _start_timer():
        request._start_time = time.perf_counter()

    @app.after_request
    def _record_request(response):
        # the route pattern (never the raw path) keeps the number of series bounded
        endpoint = request.url_rule.rule if request.url_rule else "unmatched"
        if endpoint != "/metrics":
            HTTP_REQUESTS.labels(request.method, endpoint, str(response.status_code)).inc()
            HTTP_LATENCY.labels(endpoint).observe(time.perf_counter() - request._start_time)
        return response

    @app.route("/metrics")
    def metrics():
        # public by default (local docker-compose), protected when METRICS_TOKEN is set (e.g. on Render)
        token = os.getenv("METRICS_TOKEN")
        if token and not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {token}"):
            return {"error": "unauthorized"}, 401
        return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

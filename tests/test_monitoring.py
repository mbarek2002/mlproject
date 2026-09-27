import pytest
from prometheus_client import REGISTRY

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def value(name, **labels):
    # Prometheus counters are global: tests compare values before / after an action
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.content_type.startswith("text/plain")
    body = response.get_data(as_text=True)
    for metric in ["http_requests_total", "predictions_total", "predicted_math_score", "input_score", "model_info"]:
        assert metric in body


def test_prediction_updates_model_and_data_metrics(client, valid_form):
    before_predictions = value("predictions_total")
    before_reading = value("input_score_count", feature="reading_score")
    before_gender = value("input_category_total", feature="gender", value="female")
    before_requests = value("http_requests_total", method="POST", endpoint="/predictdata", status="200")

    client.post("/predictdata", data=valid_form)

    assert value("predictions_total") == before_predictions + 1
    assert value("input_score_count", feature="reading_score") == before_reading + 1
    assert value("input_category_total", feature="gender", value="female") == before_gender + 1
    assert value("http_requests_total", method="POST", endpoint="/predictdata", status="200") == before_requests + 1
    # the registry is disabled in tests (see conftest): the pickle fallback is served
    assert value("model_info", source="pickle-fallback", version="none") == 1


@pytest.mark.parametrize(
    "changes, reason",
    [({"gender": ""}, "missing_field"), ({"reading_score": "abc"}, "not_a_number"), ({"writing_score": "150"}, "out_of_range")],
)
def test_invalid_form_is_counted_by_reason(client, valid_form, changes, reason):
    before = value("invalid_requests_total", reason=reason)

    client.post("/predictdata", data=dict(valid_form, **changes))

    assert value("invalid_requests_total", reason=reason) == before + 1


def test_unknown_paths_share_one_label(client):
    # raw paths must never become labels (unbounded number of series)
    before = value("http_requests_total", method="GET", endpoint="unmatched", status="404")

    client.get("/random-path-1")
    client.get("/random-path-2")

    assert value("http_requests_total", method="GET", endpoint="unmatched", status="404") == before + 2


def test_metrics_can_be_protected_with_a_token(client, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "metrics-secret")

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer metrics-secret"}).status_code == 200

import re

import pytest

from app import app
from src.pipeline.predict_pipeline import PredictPipeline


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def predicted_value(response):
    match = re.search(r"Predicted maths score: <strong>([^<]*)</strong>", response.get_data(as_text=True))
    assert match, "no prediction in the page"
    return float(match.group(1))


def test_index_page(client):
    assert client.get("/").status_code == 200


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_form_page(client):
    response = client.get("/predictdata")

    assert response.status_code == 200
    assert 'name="reading_score"' in response.get_data(as_text=True)


def test_valid_form_returns_prediction(client, valid_form):
    response = client.post("/predictdata", data=valid_form)

    assert response.status_code == 200
    assert 0 <= predicted_value(response) <= 100


def test_reading_and_writing_are_not_swapped(client, valid_form, make_student):
    form = dict(valid_form, reading_score="90", writing_score="40")

    from_app = predicted_value(client.post("/predictdata", data=form))

    expected = round(float(PredictPipeline().predict(make_student(reading_score=90, writing_score=40))[0]), 2)
    swapped = round(float(PredictPipeline().predict(make_student(reading_score=40, writing_score=90))[0]), 2)
    assert expected != swapped  # otherwise this test could not detect a swap
    assert from_app == expected


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"gender": ""}, "Missing fields: gender"),
        ({"reading_score": ""}, "Missing fields: reading_score"),
        ({"writing_score": "abc"}, "writing_score must be a number"),
        ({"reading_score": "150"}, "reading_score must be between 0 and 100"),
        ({"writing_score": "-5"}, "writing_score must be between 0 and 100"),
    ],
)
def test_invalid_form_returns_400(client, valid_form, changes, message):
    response = client.post("/predictdata", data=dict(valid_form, **changes))

    assert response.status_code == 400
    assert message in response.get_data(as_text=True)


def test_missing_field_in_raw_request_returns_400(client, valid_form):
    # request sent without the browser (no HTML "required" check): the field is absent, not empty
    form = {key: value for key, value in valid_form.items() if key != "lunch"}

    response = client.post("/predictdata", data=form)

    assert response.status_code == 400
    assert "Missing fields: lunch" in response.get_data(as_text=True)

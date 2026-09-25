import os

import mlflow.sklearn
import pandas as pd
import pytest

from src.pipeline.predict_pipeline import PredictPipeline

DATA_PATH = os.path.join("notebooks", "data", "stud.csv")
TARGET_COLUMN = "math_score"
NUMERICAL_COLUMNS = ["writing_score", "reading_score"]
CATEGORICAL_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]


@pytest.fixture(autouse=True)
def model_source(request, monkeypatch):
    '''
    Tests use the artifacts/*.pkl models (tracked in git) so they give the same result on every machine.
    Tests marked @pytest.mark.registry use the MLflow Model Registry instead.
    The model cache is reset around every test.
    '''
    PredictPipeline._model = None

    if "registry" not in request.keywords:
        def registry_disabled(*args, **kwargs):
            raise RuntimeError("MLflow registry disabled in tests")

        monkeypatch.setattr(mlflow.sklearn, "load_model", registry_disabled)

    yield
    PredictPipeline._model = None


@pytest.fixture
def raw_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def make_student():
    '''
    Returns a function building a one-row features DataFrame, as the Flask app sends it
    '''
    def _make_student(reading_score=72.0, writing_score=74.0, **overrides):
        row = {
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "reading_score": float(reading_score),
            "writing_score": float(writing_score),
        }
        row.update(overrides)
        return pd.DataFrame([row])

    return _make_student


@pytest.fixture
def valid_form():
    return {
        "gender": "female",
        "ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "reading_score": "72",
        "writing_score": "74",
    }

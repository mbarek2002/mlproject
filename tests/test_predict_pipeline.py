import os

import mlflow
import mlflow.sklearn
import pytest

from src.mlflow_config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
from tests.conftest import TARGET_COLUMN


def predict_one(df):
    return float(PredictPipeline().predict(df)[0])


def test_custom_data_has_training_columns(raw_df):
    df = CustomData(
        gender="male",
        race_ethnicity="group C",
        parental_level_of_education="high school",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=60,
        writing_score=65,
    ).get_data_as_data_frame()

    assert len(df) == 1
    assert set(df.columns) == set(raw_df.drop(columns=[TARGET_COLUMN]).columns)


def test_prediction_is_a_valid_score(make_student):
    prediction = predict_one(make_student(reading_score=72, writing_score=74))

    assert 0 <= prediction <= 100


def test_better_scores_give_better_prediction(make_student):
    weak = predict_one(make_student(reading_score=40, writing_score=40))
    strong = predict_one(make_student(reading_score=90, writing_score=90))

    assert strong > weak


def test_fallback_uses_artifacts_pickles(make_student):
    student = make_student()

    prediction = predict_one(student)

    model = load_object(os.path.join("artifacts", "model.pkl"))
    preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
    assert prediction == pytest.approx(float(model.predict(preprocessor.transform(student))[0]))
    assert [name for name, _ in PredictPipeline._model.steps] == ["preprocessor", "model"]


def test_model_is_loaded_once():
    assert PredictPipeline.load_model() is PredictPipeline.load_model()


@pytest.mark.registry
@pytest.mark.skipif(not os.path.exists("mlflow.db"), reason="no local MLflow registry (run the training pipeline first)")
def test_registry_champion_is_used(make_student):
    student = make_student()

    prediction = predict_one(student)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    champion = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    assert prediction == pytest.approx(float(champion.predict(student)[0]))

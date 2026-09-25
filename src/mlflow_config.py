import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "student-performance"
REGISTERED_MODEL_NAME = "student-math-model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

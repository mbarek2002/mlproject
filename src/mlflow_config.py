# import os

# import mlflow
# from dotenv import load_dotenv

# # local development: read MLFLOW_* variables from a .env file (never committed)
# load_dotenv()

# # remote server (e.g. DagsHub) when MLFLOW_TRACKING_URI is set, local SQLite registry otherwise.
# # Credentials are read by MLflow itself from MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# MLFLOW_EXPERIMENT_NAME = "student-performance"
# REGISTERED_MODEL_NAME = "student-math-model"
# CHAMPION_ALIAS = "champion"
# CHALLENGER_ALIAS = "challenger"

# # fail fast when the server is unreachable (MLflow defaults retry for minutes):
# # the app then falls back to artifacts/*.pkl instead of hanging
# os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
# os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")


# def setup_mlflow():
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


import os

import mlflow
from dotenv import load_dotenv

load_dotenv()

# Databricks: "databricks". Without the variable: local SQLite registry (mlflow.db)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
# Databricks Unity Catalog: "databricks-uc". Empty: same server as the tracking URI
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI") or None
# Databricks needs a workspace path, e.g. /Users/you@mail.com/student-performance
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "student-performance")
# Unity Catalog needs catalog.schema.model, e.g. workspace.default.student_math_model
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "student-math-model")
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# fail fast when the server is unreachable (MLflow defaults retry for minutes):
# the app then falls back to artifacts/*.pkl instead of hanging
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")


def configure_mlflow_uris():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if MLFLOW_REGISTRY_URI:
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)


def setup_mlflow():
    configure_mlflow_uris()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

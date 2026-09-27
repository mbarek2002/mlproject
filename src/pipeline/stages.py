'''
Pipeline stages, run by DVC (see dvc.yaml):

    python -m src.pipeline.stages ingest      # stud.csv -> train.csv / test.csv
    python -m src.pipeline.stages transform   # -> preprocessor.pkl, train_arr.npy, test_arr.npy
    python -m src.pipeline.stages train       # -> model.pkl, metrics.json, MLflow run + registry

Each stage reads its inputs from artifacts/ and writes its outputs there, so DVC can
re-run only the stages whose inputs (data, code or params.yaml) changed.
'''
import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.components.data_ingection import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.mlflow_config import setup_mlflow
from src.utils import read_params

TRAIN_ARRAY_PATH = os.path.join("artifacts", "train_arr.npy")
TEST_ARRAY_PATH = os.path.join("artifacts", "test_arr.npy")
METRICS_PATH = "metrics.json"
PARAMS_PATH = "params.yaml"


def ingest():
    params = read_params(PARAMS_PATH)["data"]

    ingestion = DataIngestion()
    ingestion.ingestion_config.test_size = params["test_size"]
    ingestion.ingestion_config.random_state = params["random_state"]
    return ingestion.initiate_data_ingestion()


def transform():
    config = DataIngestionConfig()
    train_arr, test_arr, preprocessor_path = DataTransformation().initiate_data_transformation(
        config.train_data_path, config.test_data_path
    )
    np.save(TRAIN_ARRAY_PATH, train_arr)
    np.save(TEST_ARRAY_PATH, test_arr)
    return preprocessor_path


def data_version(source_path):
    '''
    DVC hash of the raw dataset (from its .dvc file), None if the data is not tracked by DVC
    '''
    dvc_file = f"{source_path}.dvc"
    if not os.path.exists(dvc_file):
        return None
    with open(dvc_file) as file_obj:
        return yaml.safe_load(file_obj)["outs"][0]["md5"]


def train():
    params = read_params(PARAMS_PATH)
    ingestion_config = DataIngestionConfig()
    preprocessor_path = DataTransformationConfig().preprocessor_obj_file_path

    train_arr = np.load(TRAIN_ARRAY_PATH)
    test_arr = np.load(TEST_ARRAY_PATH)

    setup_mlflow()
    with mlflow.start_run(run_name="train_pipeline"):
        # lineage: which data and which parameters produced this model
        mlflow.set_tags({
            "data.source": ingestion_config.source_data_path,
            "data.dvc_md5": data_version(ingestion_config.source_data_path) or "not tracked",
        })
        mlflow.log_params({
            "n_train": len(train_arr),
            "n_test": len(test_arr),
            "test_size": params["data"]["test_size"],
            "random_state": params["data"]["random_state"],
            "cv_folds": params["train"]["cv_folds"],
            "target_column": "math_score",
            "raw_columns": list(pd.read_csv(ingestion_config.test_data_path, nrows=0).columns),
        })
        mlflow.log_artifact(PARAMS_PATH)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        best_model_name, cv_r2, test_r2 = ModelTrainer().initiate_model_trainer(
            train_arr, test_arr, preprocessor_path, ingestion_config.test_data_path, params["train"]
        )

    # read by "dvc metrics show" / "dvc metrics diff"
    with open(METRICS_PATH, "w") as file_obj:
        json.dump({"best_model": best_model_name, "cv_r2": round(cv_r2, 4), "test_r2": round(test_r2, 4)},
                  file_obj, indent=2)

    logging.info(f"Training completed: {best_model_name}, cv_r2={cv_r2}, test_r2={test_r2}")
    return test_r2


STAGES = {"ingest": ingest, "transform": transform, "train": train}

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
            raise ValueError(f"usage: python -m src.pipeline.stages [{' | '.join(STAGES)}]")
        print(STAGES[sys.argv[1]]())
    except Exception as e:
        raise CustomException(e, sys)

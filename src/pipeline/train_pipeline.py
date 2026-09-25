import sys

import mlflow

from src.exception import CustomException
from src.logger import logging
from src.mlflow_config import setup_mlflow
from src.components.data_ingection import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        setup_mlflow()

    def run(self):
        try:
            with mlflow.start_run(run_name="train_pipeline"):
                logging.info("Training pipeline started")

                train_path, test_path = DataIngestion().initiate_data_ingestion()

                train_arr, test_arr, preprocessor_path = DataTransformation().initiate_data_transformation(
                    train_path, test_path
                )

                r2_square = ModelTrainer().initiate_model_trainer(
                    train_arr, test_arr, preprocessor_path, test_path
                )

                logging.info(f"Training pipeline completed, r2 on test: {r2_square}")
                return r2_square

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print(TrainPipeline().run())

import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.mlflow_config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, CHAMPION_ALIAS
import os
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

class PredictPipeline:
    # shared by all instances: the model is loaded once per process, not on every request
    _model=None

    def __init__(self):
        pass

    @classmethod
    def load_model(cls):
        '''
        Returns a model that takes the raw features DataFrame (preprocessor + model).
        1st choice: champion from the MLflow Model Registry, fallback: artifacts/*.pkl
        '''
        if cls._model is not None:
            return cls._model

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri=f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
            cls._model=mlflow.sklearn.load_model(model_uri)
            logging.info(f"Loaded model from {model_uri}")
        except Exception as e:
            logging.info(f"Could not load model from MLflow registry ({e}), falling back to artifacts/*.pkl")
            model=load_object(file_path=os.path.join("artifacts","model.pkl"))
            preprocessor=load_object(file_path=os.path.join("artifacts","preprocessor.pkl"))
            cls._model=Pipeline(steps=[("preprocessor",preprocessor),("model",model)])

        return cls._model

    def predict(self,features):
        try:
            return self.load_model().predict(features)

        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

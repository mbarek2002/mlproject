import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.mlflow_config import configure_mlflow_uris, REGISTERED_MODEL_NAME, CHAMPION_ALIAS
import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

class PredictPipeline:
    # shared by all instances: the model is loaded once per process, not on every request
    _model=None
    # where the served model comes from, shown by /health
    _model_info={"source":None}

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
            configure_mlflow_uris()
            # resolve the alias first, so the exact version served is known
            version=MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME,CHAMPION_ALIAS).version
            model_uri=f"models:/{REGISTERED_MODEL_NAME}/{version}"
            cls._model=mlflow.sklearn.load_model(model_uri)
            cls._model_info={"source":"mlflow-registry","model":REGISTERED_MODEL_NAME,
                             "alias":CHAMPION_ALIAS,"version":str(version)}
            logging.info(f"Loaded model from {model_uri} (@{CHAMPION_ALIAS})")
        except Exception as e:
            logging.info(f"Could not load model from MLflow registry ({e}), falling back to artifacts/*.pkl")
            model=load_object(file_path=os.path.join("artifacts","model.pkl"))
            preprocessor=load_object(file_path=os.path.join("artifacts","preprocessor.pkl"))
            cls._model=Pipeline(steps=[("preprocessor",preprocessor),("model",model)])
            cls._model_info={"source":"pickle-fallback"}

        return cls._model

    @classmethod
    def reload_model(cls):
        '''
        Drops the cached model and loads the current champion again (e.g. after a new promotion).
        The old model keeps serving requests until the new one is loaded.
        '''
        cls._model=None
        cls.load_model()
        return cls._model_info

    @classmethod
    def model_info(cls):
        return dict(cls._model_info)

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

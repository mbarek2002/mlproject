import os
import sys
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.mlflow_config import REGISTERED_MODEL_NAME, CHAMPION_ALIAS, CHALLENGER_ALIAS

from src.utils import save_object,evaluate_models,load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array,preprocessor_path,test_path):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_r2_test", r2_square)

            self.register_model(best_model, best_model_name, r2_square, preprocessor_path, test_path)

            return r2_square





        except Exception as e:
            raise CustomException(e,sys)

    def register_model(self, best_model, best_model_name, r2_square, preprocessor_path, test_path):
        '''
        Logs preprocessor + best model as a single sklearn Pipeline in the Model Registry,
        and moves the "champion" alias to the new version only if it beats the current champion.
        '''
        try:
            preprocessor = load_object(file_path=preprocessor_path)
            full_model = Pipeline(steps=[("preprocessor", preprocessor), ("model", best_model)])

            # raw input as the Flask app sends it (scores as float)
            X_test_raw = pd.read_csv(test_path).drop(columns=["math_score"])
            X_test_raw[["reading_score", "writing_score"]] = X_test_raw[["reading_score", "writing_score"]].astype(float)

            signature = infer_signature(X_test_raw, full_model.predict(X_test_raw))

            model_info = mlflow.sklearn.log_model(
                full_model,
                artifact_path="model",
                signature=signature,
                input_example=X_test_raw.head(3),
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            client = MlflowClient()
            new_version = model_info.registered_model_version
            client.set_model_version_tag(REGISTERED_MODEL_NAME, new_version, "model_type", best_model_name)

            try:
                champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
                champion_r2 = client.get_run(champion.run_id).data.metrics["best_r2_test"]
            except Exception:
                champion_r2 = None  # no champion yet: first training

            if champion_r2 is None or r2_square > champion_r2:
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version)
                logging.info(f"Model version {new_version} promoted to @{CHAMPION_ALIAS} (r2={r2_square}, previous={champion_r2})")
            else:
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, new_version)
                logging.info(f"Model version {new_version} set as @{CHALLENGER_ALIAS} (r2={r2_square} <= champion {champion_r2})")

        except Exception as e:
            raise CustomException(e,sys)
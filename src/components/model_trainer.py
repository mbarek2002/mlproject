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


    def initiate_model_trainer(self,train_array,test_array,preprocessor_path,test_path,train_params):
        '''
        train_params: the "train" section of params.yaml (cv_folds, min_cv_r2, param_grids)
        Returns (best model name, cross-validation r2, test r2)
        '''
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
            # hyperparameter grids come from params.yaml
            params=train_params["param_grids"]
            if set(params)!=set(models):
                raise CustomException(
                    f"params.yaml param_grids {sorted(params)} do not match the models {sorted(models)}",sys)

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params,cv=train_params["cv_folds"])

            ## model_report = {model name: cross-validation r2}, the best one is chosen without looking at the test set
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score<train_params["min_cv_r2"]:
                raise CustomException("No best model found",sys)
            logging.info(f"Best model on cross-validation: {best_model_name} (cv_r2={best_model_score})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_cv_r2", best_model_score)
            mlflow.log_metric("best_r2_test", r2_square)

            self.register_model(best_model, best_model_name, r2_square, preprocessor_path, test_path)

            return best_model_name, best_model_score, r2_square





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
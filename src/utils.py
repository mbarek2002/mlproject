import os
import sys

import numpy as np 
import pandas as pd
import pickle
import mlflow
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para=param[model_name]

            # one nested run per model, under the train_pipeline parent run
            with mlflow.start_run(run_name=model_name, nested=True):
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                #model.fit(X_train, y_train)  # Train model

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)

                mlflow.set_tag("model_type", type(model).__name__)
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics({
                    "cv_r2": gs.best_score_,
                    "r2_train": train_model_score,
                    "r2_test": test_model_score,
                })

            # model selection uses the cross-validation score only: the test set is kept for the final evaluation
            report[model_name] = gs.best_score_

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
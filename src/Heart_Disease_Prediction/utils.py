import os 
import sys
from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.exception import CustomException
import pandas as pd

import pickle
import numpy as np



def read_data(file_path):

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data read successfully from {file_path}")
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):

    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}")
            params = param_grid.get(model_name, {})
            if params:
                from sklearn.model_selection import GridSearchCV
                grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            report[model_name] = accuracy
            logging.info(f"{model_name} Accuracy: {accuracy}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
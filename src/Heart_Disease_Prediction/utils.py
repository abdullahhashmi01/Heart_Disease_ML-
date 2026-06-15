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
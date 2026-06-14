import os 
import sys
from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.exception import CustomException
import pandas as pd


def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data read successfully from {file_path}")
        return df
    except Exception as e:
        raise CustomException(e, sys)
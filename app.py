from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.exception import CustomException 
import sys
import os

from src.Heart_Disease_Prediction.components.data_ingestion import DataIngestion
from src.Heart_Disease_Prediction.components.data_ingestion import DataIngestionConfig
from src.Heart_Disease_Prediction.components.data_transformation import DataTransformation
from src.Heart_Disease_Prediction.components.data_transformation import DataTransformationConfig
from src.Heart_Disease_Prediction.components.model_trainer import ModelTrainer
from src.Heart_Disease_Prediction.components.model_trainer import ModelTrainerConfig


if __name__ == "__main__":
    logging.info("Starting the heart disease prediction application.")

    try:

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        logging.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")

        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)


        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        
    except Exception as e:
        logging.error("An error occurred: {}".format(str(e)))
        raise CustomException(e, sys)

    
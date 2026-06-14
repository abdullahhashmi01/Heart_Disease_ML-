from src.Heart_Disease_Prediction.logger import logging
from src.Heart_Disease_Prediction.exception import CustomException 
import sys


if __name__ == "__main__":
    logging.info("Starting the heart disease prediction application.")

    try:
        a =1/0
    except Exception as e:
        logging.error("An error occurred: {}".format(str(e)))
        raise CustomException(e, sys)

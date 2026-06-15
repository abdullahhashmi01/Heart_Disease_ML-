import sys
from src.Heart_Disease_Prediction.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{0} line number [{1}] error message [{2}]".format(
    file_name, exc_tb.tb_lineno, str(error))
    
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message=error_message, 
            error_detail=error_detail
        )
    
    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        # Add your error formatting logic here
        return f"{error_message}\nError Detail: {error_detail}"
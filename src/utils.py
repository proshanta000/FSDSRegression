import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object (file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            # train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:  # Use specific exception types
            custom_exception = CustomException(e, sys)
            logging.info('Exception occured during model training')
            raise custom_exception  # Re-raise the custom exception
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Excption occur in load_object function utils')
        raise CustomException(e, sys)







    '''
        def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}  # Initialize the report dictionary OUTSIDE the loop
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)

                # Predict Testing data
                y_test_pred = model.predict(X_test)

                # Get R2 scores for test data
                test_model_score = r2_score(y_test, y_test_pred)

                report[model_name] = test_model_score
            except Exception as inner_e:
                logging.error(f"Error evaluating model {model_name}: {inner_e}")
                report[model_name] = None  # Or a suitable error value

        return report  # Return the report AFTER the loop has finished

    except Exception as e:
        logging.info('Outer exception occurred during model evaluation')
        raise CustomException(e, sys)
    '''
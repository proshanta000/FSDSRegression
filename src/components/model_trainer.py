#Basic inport 
import os
import sys
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTraningConfig:
    traning_model_file_path = os.path.join('artifacts', 'model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTraningConfig()

    def initiate_model_traning(self, train_array, test_array):
        try:
            logging.info('Splotting dependent and independent variable from train and test data')
            X_train, y_train, X_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            ##Traning multipel model
            models ={
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet' : ElasticNet()
            }
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('=' * 35)  # Optional separator for clarity
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary
            ''' best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            '''

            best_model_name, best_model_score = max(model_report.items(), key=lambda item: item[1])
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('=' * 35)  # Optional separator for clarity
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.traning_model_file_path,
                obj= best_model
            )

        except Exception as e:
            logging.info('Exception occured at Model Traning')
            raise CustomException(e, sys)
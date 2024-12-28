import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
  def __init__(self):
    pass

  def predict(self, features):
    try:
      preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
      model_path = os.path.join('artifacts', 'model.pkl')
      

      # Use a more informative logging level
      logging.debug("Loading preprocessor from: %s", preprocessor_path)
      preprocessor = load_object(preprocessor_path)

      # Use a more informative logging level
      logging.debug("Loading model from: %s", model_path)
      model = load_object(model_path)

      data_scaled = preprocessor.transform(features)
      pred = model.predict(data_scaled)

      return pred

    except FileNotFoundError as e:
      logging.error("Model or preprocessor file not found: %s", e)
      raise CustomException("Model or preprocessor file not found.", sys)
    except Exception as e:
      logging.error("Exception occurred in prediction: %s", e)
      raise CustomException(e, sys)

class CustomData:
  def __init__(self,
                carat: float,
                depth: float,
                table: float,
                x: float,
                y: float,
                z: float,
                cut: str,
                color: str,
                clarity: str):
    self.carat = carat
    self.depth = depth
    self.table = table
    self.x = x
    self.y = y
    self.z = z
    self.cut = cut
    self.color = color
    self.clarity = clarity

  def get_data_as_dataframe(self):
    try:
      custom_data_input_dict = {
          'carat': [self.carat],
          'depth': [self.depth],
          'table': [self.table],
          'x': [self.x],
          'y': [self.y],
          'z': [self.z],
          'cut': [self.cut],
          'color': [self.color],
          'clarity': [self.clarity]
      }
      df = pd.DataFrame(custom_data_input_dict)
      logging.info('Dataframe gathered')
      return df
    except Exception as e:
      logging.error('Exception occurred in data preparation: %s', e)
      raise CustomException(e, sys)
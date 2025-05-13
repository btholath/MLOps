from etl_pipeline.security.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

import os
import sys

from etl_pipeline.security.exception.exception import NetworkSecurityException
from etl_pipeline.security.logging.logger_config import logging
class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e,sys)
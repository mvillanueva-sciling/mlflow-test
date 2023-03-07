import os
from typing import Optional, Tuple, Union
import warnings
import sys
import time
import logging

import pandas as pd
from common.trainer import Trainer, train_wrapper
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from utils import plot_utils


client = mlflow.client.MlflowClient()
now = round(time.time())
logger = logging.getLogger(__name__)

        
class ElasticNetTrainer(Trainer):
    """
    Trainer class 
    """

    def __init__(self, experiment_name: Optional[str],
                 save_signature: Optional[bool],
                 data_path: str,
                 registered_model_version_stage : Optional[str],
                 output_path: Optional[Union[str, None]],
                 **kwargs):
        self.data_path = data_path
        super().__init__(experiment_name=experiment_name,
                         save_signature=save_signature,
                         registered_model_version_stage=registered_model_version_stage,
                         output_path=output_path,
                         **kwargs)
        

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # noqa
        """
        Load data and split it into training and testing sets
        """
        data = pd.read_csv(self.data_path)
        train, test = train_test_split(data, test_size=0.30, random_state=42)
        col_label = "quality"

        # The predicted column is "quality" which is a scalar from [3, 9]
        X_train = train.drop([col_label], axis=1)
        X_test = test.drop([col_label], axis=1)
        y_train = train[[col_label]]
        y_test = test[[col_label]]

        logger.debug(f">> X_test.type: {type(X_test)}")
        logger.debug(f">> X_test.dtypes: {X_test.dtypes}")

        return X_train, X_test, y_train, y_test
         
    @train_wrapper
    def train(self,
              registered_model_name: Optional[str],
              plot_file: Optional[str],
              alpha: Optional[Union[int, None]] = None,
              l1_ratio: Optional[Union[int, None]] = 32):
        """
        Train a model and log in mlflow
        """        
        # Create model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        # Fit and predict
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        # MLflow params
        logger.debug("Parameters:")
        logger.debug(f"  alpha: {alpha}")
        logger.debug(f"  l1_ratio: {l1_ratio}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # MLflow metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        logger.debug("Metrics:")
        logger.debug("  rmse: {rmse}")
        logger.debug("  mae: {mae}")
        logger.debug("  r2: {r2}")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        if plot_file:
            plot_utils.create_plot_file(y_test_set=self.y_test, y_predicted=predictions, plot_file=plot_file)
            mlflow.log_artifact(plot_file)

        return model, predictions, registered_model_name

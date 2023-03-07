import time

from pprint import pprint

import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.xgboost

from utils import fetch_logged_data

now = fmt_ts_seconds(round(time.time()))

class Trainer():
    def __init__(self, experiment_name, data_path, log_as_onnx, save_signature, run_origin=None, use_run_id_as_run_name=False):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        self.log_as_onnx = log_as_onnx
        self.save_signature = save_signature
        self.use_run_id_as_run_name = use_run_id_as_run_name
        self.X_train, self.X_test, self.y_train, self.y_test = self._build_data(data_path)

        if self.experiment_name:
            mlflow.set_experiment(experiment_name)
            exp = client.get_experiment_by_name(experiment_name)
            client.set_experiment_tag(exp.experiment_id, "version_mlflow", mlflow.__version__)
            client.set_experiment_tag(exp.experiment_id, "experiment_created", now)

    
def main():
    # prepare example dataset
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # enable auto logging
    # this includes xgboost.sklearn estimators
    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name='Experiment 1') as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        mlflow.set_tag("run_id", run_id)

        # Define parameters
        n_estimators = 20
        reg_lambda= 1
        gamma = 0

        # Create Model
        model = xgb.XGBRegressor(n_estimators=n_estimators, reg_lambda=reg_lambda, gamma=gamma, max_depth=3)

        # Fit and predict
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mlflow.log_metric('n_estimators', n_estimators)
        mlflow.log_metric('reg_lambda', reg_lambda)
        mlflow.log_metric('gamma', gamma)



        # Create signature
        signature = infer_signature(X_train, predictions) if self.save_signature else None
        print("Signature:",signature)

        regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        

    # show logged data
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)


if __name__ == "__main__":
    main()
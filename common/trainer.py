# Standard libraries
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# Third party libraries
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Common variables
client = mlflow.client.MlflowClient()
now = round(time.time())
logger = logging.getLogger(__name__)


def train_wrapper(func):
    """
    This wrapper function is used to wrap the training function.
    Define mlflow tags and metrics to track the training process.
    """
    def wrapper(self, *args, **kwargs):
        # Defien a run name for this session
        run_name = f"{now} {mlflow.__version__}"
        
        with mlflow.start_run(run_name=run_name) as run: 
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            logger.debug("MLflow run:")
            logger.debug("run_id: %s", run_id)
            logger.debug("experiment_id: %s", str(experiment_id))
            logger.debug("experiment_name: %s", client.get_experiment(experiment_id).name) # noqa

            # Set mlflow tags for this run
            mlflow.set_tag("mlflow.runName", run_id)
            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag("save_signature", self.save_signature)
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("registered_model_version_stage", self.registered_model_version_stage) # noqa
            mlflow.set_tag("dataset", "wine-quality")
            mlflow.set_tag("timestamp", now)
            mlflow.set_tag("version.mlflow", mlflow.__version__)

            model, predictions, registered_model_name = func(self, *args, **kwargs) # noqa

            # Create signature
            signature = infer_signature(self.X_train, predictions) if self.save_signature else None # noqa
            logger.debug("Signature: %s", signature)

            # MLflow log model
            mlflow.sklearn.log_model(model, "model", signature=signature)
            if registered_model_name:
                model_uri = f"runs:/{run.info.run_id}/sklearn-model"
                mv = mlflow.register_model(model_uri, registered_model_name)
                logger.debug("Name: %s", mv.name)
                logger.debug("Version: %s", mv.version)

            # Write run ID to file
            if (self.output_path):
                mlflow.set_tag("output_path", self.output_path)
                output_path = self.output_path.replace("dbfs:", "/dbfs")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(run_id)

        # Log run info
        run = client.get_run(run_id)
        client.set_tag(run_id, "run.info.start_time", run.info.start_time)
        client.set_tag(run_id, "run.info.end_time", run.info.end_time)
        client.set_tag(run_id, "run.info._start_time", run.info.start_time)
        client.set_tag(run_id, "run.info._end_time", run.info.end_time)
        client.set_tag(run_id, "run.info.status", run.info.status)
        
        return (experiment_id, run_id)
    
    return wrapper


class Trainer(ABC):
    """
    Base class for all trainers.
    """
    def __init__(self, experiment_name: Optional[str],
                 save_signature: Optional[bool],
                 registered_model_version_stage : Optional[str],  # noqa
                 output_path: Optional[str],
                 **kwargs):  # noqa
        """
        Constructor for the Trainer class
        """
        self.experiment_name = experiment_name
        self.save_signature = save_signature
        self.registered_model_version_stage = registered_model_version_stage
        self.output_path = output_path

        self.X_train,self.X_test, self.y_train, self.y_test = self.load_data(**kwargs)  # noqa

        if self.experiment_name:
            mlflow.set_experiment(experiment_name)
            exp = client.get_experiment_by_name(experiment_name)
            client.set_experiment_tag(exp.experiment_id,
                                      "version_mlflow",
                                      mlflow.__version__)
            client.set_experiment_tag(exp.experiment_id,
                                      "experiment_created",
                                      now)
    
    @abstractmethod
    def load_data(*args, **kwargs) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: # noqa
        """
        Load the data.
        """
        pass

    
    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

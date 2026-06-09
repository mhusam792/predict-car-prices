import mlflow
import mlflow.lightgbm
from pathlib import Path
from src.data.yml_validator import ValidateInputs


class MLflowTracker:

    def __init__(self, config: ValidateInputs):

        self.config = config

        self.experiment_name = config.model.experiment_name
        self.model_name = config.model.name

        mlflow.set_tracking_uri(
            getattr(config.model, "tracking_uri", "sqlite:///mlflow.db")
        )

        mlflow.set_experiment(self.experiment_name)

    def start_run(self):
        return mlflow.start_run()

    @staticmethod
    def log_params(params: dict):
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict):
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifact(path: str | Path):
        mlflow.log_artifact(str(path))

    @staticmethod
    def log_model(model):
        return mlflow.lightgbm.log_model(
            model,
            name="model",
        )

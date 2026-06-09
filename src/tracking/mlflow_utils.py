import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import mlflow
import mlflow.lightgbm
import mlflow.xgboost

from src.config.schema import AppConfig

logger = logging.getLogger(__name__)


def configure_mlflow(config: AppConfig) -> None:
    """Set tracking URI and active experiment. Call once before training."""
    mlflow.set_tracking_uri(config.model.tracking_uri)
    mlflow.set_experiment(config.model.experiment_name)
    logger.info(
        "MLflow configured — experiment='%s' uri='%s'",
        config.model.experiment_name,
        config.model.tracking_uri,
    )


@contextmanager
def start_run() -> Generator[mlflow.ActiveRun, None, None]:
    with mlflow.start_run() as run:
        yield run


def log_params(params: dict) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def log_model(model, model_name: str) -> None:
    """Log model with the correct MLflow flavour based on type name."""
    type_name = type(model).__name__.lower()

    if "lgbm" in type_name:
        mlflow.lightgbm.log_model(model, name="model")
        return

    if "xgb" in type_name:
        mlflow.xgboost.log_model(model, name="model")
        return

    mlflow.sklearn.log_model(model, name="model")


def active_run_id() -> str:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run. Call start_run() first.")
    return run.info.run_id

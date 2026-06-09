import logging
from typing import NamedTuple

import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config.schema import AppConfig
from src.data.loader import load_dataframe
from src.data.preprocessing import build_preprocessing_pipeline
from src.models.evaluation import evaluate_model
from src.models.factory import build_model
from src.tracking import mlflow_utils as mlflow_utils
from src.tracking.registry import promote_if_better, register_model
from src.utils.helpers import save_artifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class TrainTestSplit(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_data(df: pd.DataFrame, config: AppConfig) -> TrainTestSplit:
    """
    Drop rows with missing target, deduplicate, split into train/test.
    Returns a typed NamedTuple for safe positional or named access.
    """
    df = df.dropna(subset=[config.target]).drop_duplicates()

    X = df.drop(columns=[config.target])
    y = df[config.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.split.test_size,
        random_state=config.split.splitting_random_state,
    )
    return TrainTestSplit(X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------


def fit_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    """Fit and return transformed training features."""
    X_processed = pipeline.fit_transform(X_train, y_train)
    logger.info(
        "Processed train shape: %s | cols: %s",
        X_processed.shape,
        list(X_processed.columns),
    )
    return X_processed


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


def run_experiment(config: AppConfig, splits: TrainTestSplit) -> dict[str, float]:
    """
    Fit model, log to MLflow, register, and promote if better.

    Returns the evaluation metrics dict.
    All MLflow interaction is explicit and localised here.
    """
    X_train, X_test, y_train, y_test = splits
    model_name = config.model.name.lower()

    pipeline = build_preprocessing_pipeline(config)
    X_train_processed = fit_pipeline(pipeline, X_train, y_train)
    X_test_processed = pipeline.transform(X_test)

    model = build_model(config)

    with mlflow_utils.start_run() as active_run:
        model.fit(X_train_processed, y_train)

        mlflow_utils.log_params(config.model.params)

        metrics = evaluate_model(
            model, X_train_processed, X_test_processed, y_train, y_test
        )
        mlflow_utils.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        mlflow_utils.log_model(model, model_name=model_name)

        # -------------------------------------------------------------------
        # Register + promote
        # -------------------------------------------------------------------
        run_id = active_run.info.run_id
        new_version = register_model(run_id=run_id, model_name=model_name)

        metric_name = "test_r2"
        if metric_name not in metrics:
            raise ValueError(f"'{metric_name}' not found in metrics: {list(metrics)}")

        client = MlflowClient()
        decision = promote_if_better(
            client=client,
            model_name=model_name,
            new_version=new_version,
            new_metric=float(metrics[metric_name]),
            metric_name=metric_name,
        )
        logger.info("Registry decision: %s", decision)

    # Persist artifacts locally for inference
    save_artifact(model, "artifacts/model.joblib")
    save_artifact(pipeline, "artifacts/pipeline.joblib")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train(config: AppConfig) -> dict[str, float]:
    """
    Top-level training entry point.

    Orchestrates: load → prepare → configure mlflow → run experiment.
    Returns final metrics for programmatic consumption (e.g. CI checks).
    """
    mlflow_utils.configure_mlflow(config)

    df = load_dataframe(config.main_data_path)
    splits = prepare_data(df, config)

    metrics = run_experiment(config, splits)
    return metrics

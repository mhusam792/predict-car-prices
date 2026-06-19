import logging
from typing import NamedTuple
import os


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.build_model.config.schemas.app_schema import AppConfig
from src.build_model.data.loader import load_dataframe
from src.build_model.data.preprocessing import build_preprocessing_pipeline
from src.build_model.models.evaluation import evaluate_model
from src.build_model.models.factory import build_model

import mlflow
from mlflow.models import infer_signature

logger = logging.getLogger(__name__)


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


def run_experiment(
    config: AppConfig,
    splits: TrainTestSplit,
) -> dict[str, float]:

    X_train, X_test, y_train, y_test = splits

    pipeline = build_preprocessing_pipeline(config)

    model = build_model(config)

    # Log model Parameters
    full_pipeline = Pipeline(
        steps=[
            ("preprocessing", pipeline),
            ("model", model),
        ]
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("car_price_experiment")

    with mlflow.start_run(run_name="car_price_experiment"):

        # -----------------
        # Log parameters
        # -----------------
        mlflow.log_param("model_name", config.model.model_name)
        mlflow.log_param("model_type", config.model.model)
        mlflow.log_param("model_version", config.model.version)
        mlflow.log_params(params=config.model.params.model_dump())

        # -----------------
        # Train
        # -----------------
        full_pipeline.fit(X_train, y_train)

        # -----------------
        # Evaluate
        # -----------------
        metrics = evaluate_model(
            full_pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # -----------------
        # Log metrics
        # -----------------
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # -----------------
        # Log model (VERY IMPORTANT)
        # -----------------
        signature = infer_signature(X_train, full_pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            full_pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
        )
        # -----------------
        # Optional: log dataset size
        # -----------------
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        logger.info("Metrics: %s", metrics)


def train(config: AppConfig) -> dict[str, float]:
    """
    Top-level training entry point.

    Returns final metrics for programmatic consumption (e.g. CI checks).
    """

    df = load_dataframe(config.main_data_path)
    splits = prepare_data(df, config)

    metrics = run_experiment(config, splits)
    return metrics

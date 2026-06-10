import logging
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config.schema import AppConfig
from src.data.loader import load_dataframe
from src.data.preprocessing import build_preprocessing_pipeline
from src.models.evaluation import evaluate_model
from src.models.factory import build_model
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


def run_experiment(
    config: AppConfig,
    splits: TrainTestSplit,
) -> dict[str, float]:

    X_train, X_test, y_train, y_test = splits

    pipeline = build_preprocessing_pipeline(config)

    X_train_processed = fit_pipeline(
        pipeline,
        X_train,
        y_train,
    )

    X_test_processed = pipeline.transform(X_test)

    model = build_model(config)

    model.fit(X_train_processed, y_train)

    metrics = evaluate_model(
        model,
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
    )

    logger.info("Metrics: %s", metrics)

    save_artifact(model, "artifacts/model.joblib")
    save_artifact(pipeline, "artifacts/pipeline.joblib")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train(config: AppConfig) -> dict[str, float]:
    """
    Top-level training entry point.

    Returns final metrics for programmatic consumption (e.g. CI checks).
    """

    df = load_dataframe(config.main_data_path)
    splits = prepare_data(df, config)

    metrics = run_experiment(config, splits)
    return metrics

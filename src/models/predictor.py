from pathlib import Path
from typing import NamedTuple

import joblib
import pandas as pd


class InferenceArtifacts(NamedTuple):
    """Lightweight container for loaded model + pipeline (immutable)."""

    model: object
    pipeline: object


def load_artifacts(
    model_path: str | Path = "artifacts/model.joblib",
    pipeline_path: str | Path = "artifacts/pipeline.joblib",
) -> InferenceArtifacts:
    return InferenceArtifacts(
        model=joblib.load(model_path),
        pipeline=joblib.load(pipeline_path),
    )


def predict(artifacts: InferenceArtifacts, X: pd.DataFrame) -> pd.Series:
    X_processed = artifacts.pipeline.transform(X)
    return artifacts.model.predict(X_processed)

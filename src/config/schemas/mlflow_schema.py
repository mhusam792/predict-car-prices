# mlflow_schema.py
from pathlib import Path

from pydantic import BaseModel, Field


class TrackingConfig(BaseModel):
    uri: str = Field(default="sqlite:///datasets/mlflow/database/mlflow.db")


class ExperimentConfig(BaseModel):
    name: str = Field(default="car_price_prediction")


class ArtifactConfig(BaseModel):
    location: str = Field(default=Path("datasets/mlflow/artifacts"))


class RegistryConfig(BaseModel):
    model_name: str = Field(default="car_price_model")


class LoggingConfig(BaseModel):
    autolog: bool = True
    log_models: bool = True
    log_input_examples: bool = True
    log_model_signatures: bool = True


class MLflowConfig(BaseModel):
    tracking: TrackingConfig
    experiment: ExperimentConfig
    artifacts: ArtifactConfig
    registry: RegistryConfig
    logging: LoggingConfig

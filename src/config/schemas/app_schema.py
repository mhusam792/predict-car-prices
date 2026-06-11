# app_schema.py

from pydantic import BaseModel
from pathlib import Path

from src.config.schemas.data_schema import (
    SplitConfig,
    RareLabelConfig,
)
from src.config.schemas.model_schema import ModelConfig
from src.config.schemas.mlflow_schema import MLflowConfig


class AppConfig(BaseModel):
    main_data_path: Path
    target: str

    features: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    date_cols: list[str]

    split: SplitConfig
    rare_label: RareLabelConfig

    model: ModelConfig
    mlflow: MLflowConfig

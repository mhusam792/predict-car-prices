"""
Pydantic configuration schema.
Only classes that are justified: pure data validation contracts.
"""

from typing import Any
from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    splitting_random_state: int = Field(ge=0)


class RareLabelConfig(BaseModel):
    tol: float = Field(gt=0, lt=1)
    variables: list[str]


class ModelConfig(BaseModel):
    name: str
    experiment_name: str
    tracking_uri: str = "sqlite:///datasets/mlflow/mlflow.db"
    params: dict[str, Any]


class AppConfig(BaseModel):
    """Root config. Replaces ValidateInputs — more descriptive name."""

    main_data_path: str
    target: str

    features: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    date_cols: list[str]

    split: SplitConfig
    rare_label: RareLabelConfig
    model: ModelConfig

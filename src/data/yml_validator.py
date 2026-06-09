from typing import List, Any
from pydantic import BaseModel, Field


class Split(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    splitting_random_state: int = Field(ge=0)


class RareLabels(BaseModel):
    tol: float = Field(gt=0, lt=1)
    variables: List[str]


class ModelConfig(BaseModel):
    name: str
    experiment_name: str
    params: dict[str, Any]


class ValidateInputs(BaseModel):
    main_data_path: str
    target: str

    features: List[str]
    num_cols: List[str]
    cat_cols: List[str]
    date_cols: List[str]

    split: Split

    rare_label: RareLabels

    model: ModelConfig

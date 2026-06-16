from pydantic import BaseModel
from enum import StrEnum


class ModelName(StrEnum):
    LGBM = "lgbm"
    XGB = "xgb"
    RF = "random_forest"


class ModelParams(BaseModel):
    n_estimators: int
    learning_rate: float
    subsample: float
    colsample_bytree: float

class ModelConfig(BaseModel):
    model: ModelName
    type: str
    version: int
    model_name: str
    params: ModelParams

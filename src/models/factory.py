from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.data.yml_validator import ValidateInputs


def build_model(config: ValidateInputs):
    model_name = config.model.name.lower()

    if model_name == "lgbm":
        return LGBMRegressor(**config.model.params)

    if model_name == "xgb":
        return XGBRegressor(**config.model.params)

    raise ValueError(f"Unsupported model: {config.model.name}")

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.build_model.config.schemas.app_schema import AppConfig

_SUPPORTED_MODELS = {"lgbm", "xgb"}


def build_model(config: AppConfig) -> LGBMRegressor | XGBRegressor:
    """
    Instantiate the model specified in config.

    Raises ValueError for unsupported model names so callers get a clear
    message at configuration time, not buried in a generic AttributeError.
    """
    name = config.model.model.lower()

    if name == "lgbm":
        return LGBMRegressor(**config.model.params.model_dump())

    if name == "xgb":
        return XGBRegressor(**config.model.params.model_dump())

    raise ValueError(
        f"Unsupported model '{config.model.name}'. "
        f"Choose one of: {_SUPPORTED_MODELS}"
    )

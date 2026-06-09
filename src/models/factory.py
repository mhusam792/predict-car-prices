from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config.schema import AppConfig

_SUPPORTED_MODELS = {"lgbm", "xgb"}


def build_model(config: AppConfig) -> LGBMRegressor | XGBRegressor:
    """
    Instantiate the model specified in config.

    Raises ValueError for unsupported model names so callers get a clear
    message at configuration time, not buried in a generic AttributeError.
    """
    name = config.model.name.lower()

    if name == "lgbm":
        return LGBMRegressor(**config.model.params)

    if name == "xgb":
        return XGBRegressor(**config.model.params)

    raise ValueError(
        f"Unsupported model '{config.model.name}'. "
        f"Choose one of: {_SUPPORTED_MODELS}"
    )

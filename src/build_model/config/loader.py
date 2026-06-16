# src/build_model/config/loader.py
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.build_model.config.schemas.app_schema import (
    AppConfig,
)


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(
    data_path: str | Path = "src/build_model/config/configs/data.yml",
    model_path: str | Path = "src/build_model/config/configs/model.yml",
    mlflow_path: str | Path = "src/build_model/config/configs/mlflow.yml",
) -> AppConfig:

    try:
        data_cfg = _load_yaml(data_path)
        model_cfg = _load_yaml(model_path)
        mlflow_cfg = _load_yaml(mlflow_path)

        config_dict = {
            **data_cfg,
            "model": model_cfg,
            "mlflow": mlflow_cfg,
        }

        return AppConfig(**config_dict)

    except ValidationError as e:
        raise ValueError(f"Config validation error:\n{e}") from e

    except Exception as e:
        raise ValueError(f"Failed to load config: {e}") from e

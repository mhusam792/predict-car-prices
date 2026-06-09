from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("Empty dataset")

        logger.info(f"Loaded data shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception("Failed to load data")
        raise ValueError(f"Data loading error: {e}")

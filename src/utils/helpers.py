import logging
from pathlib import Path

import joblib
import pandas as pd


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def save_artifact(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logging.getLogger(__name__).info("Saved artifact → %s", path)


def load_model(model):
    return joblib.load(model)


def get_rare_label_summary(
    df: pd.DataFrame,
    threshold: float = 0.001,
) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    rows = []
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True, dropna=False)
        rare = freq[freq <= threshold]
        rows.append(
            {
                "column": col,
                "n_unique": df[col].nunique(),
                "n_rare": len(rare),
                "rare_labels": list(rare.index),
            }
        )
    return pd.DataFrame(rows)

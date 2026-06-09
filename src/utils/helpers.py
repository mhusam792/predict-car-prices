import pandas as pd

from pathlib import Path
import joblib


def get_rare_labels_cardinality(
    df: pd.DataFrame, threshold: float = 0.001
) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    results = []

    for col in cat_cols:
        freq = df[col].value_counts(normalize=True, dropna=False)
        rare = freq[freq <= threshold]

        results.append(
            {
                "column": col,
                "n_unique": df[col].nunique(),
                "n_rare": len(rare),
                "rare_labels": list(rare.index),
            }
        )

    return pd.DataFrame(results)


def save_object(obj, path: str | Path) -> None:
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(obj, path)

import pandas as pd
from datetime import datetime


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    CURRENT_YEAR = datetime.now().year
    df["age"] = CURRENT_YEAR - df["year"]

    df["mileage_per_year"] = df["mileage"] / (df["age"] + 1)
    df["engine_mileage_interaction"] = df["engineSize"] * df["mileage"]

    df.drop(columns=["year"], inplace=True)

    return df

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from feature_engine.encoding import RareLabelEncoder

from src.data.yml_validator import ValidateInputs
from src.features.feature_engineering import create_features

feature_engineer = FunctionTransformer(create_features)


# ----------------------------
# Numeric pipeline
# ----------------------------
num_pipeline = Pipeline(
    [("power", PowerTransformer(method="yeo-johnson")), ("scaler", StandardScaler())]
)


# ----------------------------
# Categorical pipeline
# ----------------------------
def build_cat_pipeline(config: ValidateInputs):
    return Pipeline(
        [
            (
                "rare",
                RareLabelEncoder(
                    tol=config.rare_label.tol,
                    variables=config.rare_label.variables,
                    missing_values="ignore",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
            ),
        ]
    )


# ----------------------------
# Full preprocessing BEFORE imputation
# ----------------------------
def build_preprocessor(config: ValidateInputs):
    cat_pipe = build_cat_pipeline(config)

    return ColumnTransformer(
        [
            ("num", num_pipeline, list(config.num_cols)),
            ("cat", cat_pipe, list(config.cat_cols)),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ----------------------------
# Imputer
# ----------------------------
def build_imputer():
    return IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        max_iter=1,
        random_state=42,
    )


# ----------------------------
# Full pipeline
# ----------------------------
def build_pipeline(config: ValidateInputs):
    preprocessor = build_preprocessor(config)
    imputer = build_imputer()

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("imputer", imputer),
            ("feature_eng", feature_engineer),
        ]
    ).set_output(transform="pandas")

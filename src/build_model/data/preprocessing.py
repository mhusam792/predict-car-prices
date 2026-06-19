import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)
from feature_engine.encoding import RareLabelEncoder

from src.build_model.config.schemas.app_schema import AppConfig
from src.build_model.features.engineering import create_features

# ---------------------------------------------------------------------------
# Leaf pipeline builders
# ---------------------------------------------------------------------------


def build_numeric_pipeline() -> Pipeline:
    """Yeo-Johnson power transform → standard scale."""
    return Pipeline(
        [
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline(config: AppConfig) -> Pipeline:
    """Rare-label collapse → ordinal encode."""
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
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Column transformer (pre-imputation)
# ---------------------------------------------------------------------------


def build_column_transformer(config: AppConfig) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", build_numeric_pipeline(), list(config.num_cols)),
            ("cat", build_categorical_pipeline(config), list(config.cat_cols)),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------


def build_imputer() -> IterativeImputer:
    return IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        max_iter=1,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Round floating catigories after imputing
# ---------------------------------------------------------------------------


def round_float_cat(config: AppConfig) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("round", FunctionTransformer(np.round), list(config.cat_cols))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# ---------------------------------------------------------------------------
# Full end-to-end preprocessing pipeline
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline(config: AppConfig) -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline:
        column_transformer → imputer → feature_engineering

    Call .fit_transform(X_train) then .transform(X_test).
    """
    return Pipeline(
        [
            ("preprocess", build_column_transformer(config)),
            ("imputer", build_imputer()),
            ("round_cat", round_float_cat(config)),
            ("feature_eng", FunctionTransformer(create_features)),
        ]
    ).set_output(transform="pandas")

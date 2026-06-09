import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Compute regression metrics on train and test splits.

    Returns a flat dict ready to pass directly to mlflow.log_metrics().
    """
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        "train_mae": mean_absolute_error(y_train, pred_train),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_r2": r2_score(y_test, pred_test),
    }

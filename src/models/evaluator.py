import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    @staticmethod
    def evaluate(model, X_train, X_test, y_train, y_test):
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        return {
            "train_mae": mean_absolute_error(y_train, pred_train),
            "test_mae": mean_absolute_error(y_test, pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
            "test_r2": r2_score(y_test, pred_test),
        }

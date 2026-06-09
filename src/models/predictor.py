from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    def __init__(
        self,
        model_path: str | Path = "artifacts/model.joblib",
        pipeline_path: str | Path = "artifacts/pipeline.joblib",
    ):
        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)

    def predict(self, X: pd.DataFrame):
        X_processed = self.pipeline.transform(X)
        return self.model.predict(X_processed)

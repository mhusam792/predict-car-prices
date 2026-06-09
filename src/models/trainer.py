import logging
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataframe
from src.data.preprocessor import build_pipeline
from src.data.yml_validator import ValidateInputs
from src.models.evaluator import Evaluator
from src.models.factory import build_model
from src.models.registry import ModelRegistry
from src.tracking.mlflow_tracker import MLflowTracker

from src.tracking.registry import ModelRegistry
from src.models.promotion import ModelPromotionService
import mlflow
import mlflow.lightgbm

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: ValidateInputs):
        self.config = config

    def load_data(self):
        return load_dataframe(self.config.main_data_path)

    def split(self, df):
        X = df.drop(columns=[self.config.target])
        y = df[self.config.target]

        return train_test_split(
            X,
            y,
            test_size=self.config.split.test_size,
            random_state=self.config.split.splitting_random_state,
        )

    def train(self):
        df = self.load_data()

        df = df.dropna(subset=[self.config.target])
        df = df.drop_duplicates()

        X_train, X_test, y_train, y_test = self.split(df)

        pipeline = build_pipeline(self.config)

        X_train_processed = pipeline.fit_transform(X_train, y_train)

        logger.info(f"Processed train shape: {X_train_processed.shape}")
        logger.info(f"Columns: {list(X_train_processed.columns)}")

        tracker = MLflowTracker(self.config)

        with tracker.start_run():

            model = build_model(self.config)

            model.fit(
                X_train_processed,
                y_train,
            )

            mlflow.lightgbm.log_model(
                model,
                artifact_path="model"
            )

            X_test_processed = pipeline.transform(X_test)

            metrics = Evaluator.evaluate(
                model,
                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
            )

            tracker.log_params(self.config.model.params)

            tracker.log_metrics(metrics)


            # ----------------------------
            # REGISTER MODEL
            # ----------------------------
            active_run = mlflow.active_run()
            if active_run is None:
                raise RuntimeError("No active MLflow run found")

            run_id = active_run.info.run_id
            model_name = self.config.model.name.lower()

            registry = ModelRegistry()

            model_version = registry.register_model(
                run_id=run_id,
                model_name=model_name,
            )

            version = (
                model_version.version
                if hasattr(model_version, "version")
                else str(model_version)
            )

            # ----------------------------
            # PROMOTION LOGIC
            # ----------------------------
            metric_name = "test_r2"

            if metric_name not in metrics:
                raise ValueError(f"{metric_name} not found in metrics")

            new_metric = float(metrics[metric_name])

            promotion_service = ModelPromotionService(model_name=model_name)

            decision = promotion_service.promote_if_better(
                new_version=version, new_metric=new_metric, metric_name=metric_name
            )

            logger.info(f"Model registry decision: {decision}")

        X_test_processed = pipeline.transform(X_test)

        metrics = Evaluator.evaluate(
            model, X_train_processed, X_test_processed, y_train, y_test
        )

        logger.info(metrics)
        return model, pipeline

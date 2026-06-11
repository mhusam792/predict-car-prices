from src.config.schemas.app_schema import AppConfig
import mlflow


def setup_mlflow(config: AppConfig) -> None:

    mlflow.set_tracking_uri(config.mlflow.tracking.uri)

    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment.name)

    if experiment is None:
        mlflow.create_experiment(
            name=config.mlflow.experiment.name,
            artifact_location=config.mlflow.artifacts.location,
        )

    mlflow.set_experiment(config.mlflow.experiment.name)

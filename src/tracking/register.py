import mlflow
from mlflow.tracking import MlflowClient

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def register_with_aliases(
    model,
    model_name: str,
    test_mae: float,
    artifact_path: str = "model",
    metric_name: str = "test_mae",
) -> None:
    client = MlflowClient()

    model_info = mlflow.sklearn.log_model(
        model,
        name=artifact_path,
        registered_model_name=model_name,
    )

    version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=model_name,
        alias=CHALLENGER_ALIAS,
        version=version,
    )

    try:
        champion_version = client.get_model_version_by_alias(
            model_name,
            CHAMPION_ALIAS,
        )

        champion_run = client.get_run(
            champion_version.run_id,
        )

        champion_mae = champion_run.data.metrics.get(
            metric_name,
        )

    except Exception:
        champion_mae = None

    if champion_mae is None or test_mae < champion_mae:
        client.set_registered_model_alias(
            name=model_name,
            alias=CHAMPION_ALIAS,
            version=version,
        )

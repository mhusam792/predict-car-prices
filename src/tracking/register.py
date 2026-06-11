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
):
    client = MlflowClient()

    run = mlflow.active_run()
    run_id = run.info.run_id if run else None

    # log model under current run (no new run)
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)

    model_uri = f"runs:/{run_id}/{artifact_path}"

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(
        name=model_name, alias="challenger", version=mv.version
    )

    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion")

        champion_run = client.get_run(champion_version.run_id)
        champion_mae = champion_run.data.metrics.get(metric_name)

    except Exception:
        champion_mae = None

    if champion_mae is None or test_mae < champion_mae:
        client.set_registered_model_alias(model_name, "champion", mv.version)

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_ALIAS = "champion"

mlflow.set_tracking_uri("sqlite:///datasets/mlflow/mlflow.db")

print("TRACKING URI =", mlflow.get_tracking_uri())
print(
    "DB PATH =",
    Path("datasets/mlflow/mlflow.db").resolve(),
)


def load_champion_model(model_name: str):

    client = MlflowClient()

    print("REGISTERED MODELS:", [m.name for m in client.search_registered_models()])

    model_uri = f"models:/{model_name}@{MODEL_ALIAS}"

    print("MODEL URI =", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)

    return model


def get_champion_metadata(model_name: str):

    client = MlflowClient()

    version = client.get_model_version_by_alias(
        model_name,
        MODEL_ALIAS,
    )

    return {
        "model_name": model_name,
        "version": version.version,
        "alias": MODEL_ALIAS,
    }

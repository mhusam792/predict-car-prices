import pandas as pd

from src.serving.model_loader import (
    load_champion_model,
    get_champion_metadata,
)


def predict(
    model_name: str,
    payload: dict,
):

    model = load_champion_model(model_name)

    df = pd.DataFrame([payload])

    prediction = model.predict(df)

    metadata = get_champion_metadata(model_name)

    return {
        "predicted_price": float(prediction[0]),
        "model_name": metadata["model_name"],
        "model_version": metadata["version"],
        "model_alias": metadata["alias"],
    }

import mlflow
import pandas as pd

from fastapi import FastAPI
from src.api.schemas import CarFeatures
from src.config.loader import load_config

config = load_config()
mlflow.set_tracking_uri(config.mlflow.tracking.uri)

MODEL_NAME = config.mlflow.registry.model_name + "_" + config.model.name

model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@champion")

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model_alias": "champion"}


@app.post("/predict")
def predict(features: CarFeatures):

    df = pd.DataFrame([features.model_dump()])

    prediction = model.predict(df)

    return {"predicted_price": float(prediction[0])}

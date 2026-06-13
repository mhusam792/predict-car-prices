import pandas as pd

from fastapi import FastAPI
from src.api.schemas import CarFeatures
from src.config.loader import load_config
from src.utils.helpers import load_model

config = load_config()
MODEL_PATH = "artifacts/models/car_price_model_lgbm_regression_1.joblib"


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "model_alias": "champion"}


@app.post("/predict")
def predict(features: CarFeatures):

    df = pd.DataFrame([features.model_dump()])

    model = load_model(MODEL_PATH)
    prediction = model.predict(df)

    return {"predicted_price": float(prediction[0])}

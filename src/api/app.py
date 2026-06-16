# src/api/app.py
import pandas as pd

from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.api.schemas import CarFeatures
from src.utils.helpers import load_model

# Loading model
MODEL_PATH = "artifacts/models/car_price_model_lgbm_regression_1.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: CarFeatures):

    df = pd.DataFrame([features.model_dump()])

    prediction = app.state.model.predict(df)

    return {"predicted_price": float(prediction[0])}

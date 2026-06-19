# src/api/app.py

import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
import mlflow.pyfunc

from src.api.schemas import CarFeatures


MODEL_NAME = "car_reg_price"  # اسم الـ registered model في MLflow


@asynccontextmanager
async def lifespan(app: FastAPI):

    # 🔥 Load latest production/champion model from MLflow Registry
    model_uri = f"models:/{MODEL_NAME}@champion"

    app.state.model = mlflow.pyfunc.load_model(model_uri)

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
from fastapi import FastAPI

from src.serving.schemas import (
    CarFeatures,
    PredictionResponse,
)

from src.serving.inference import predict

app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0",
)


MODEL_NAME = "lgbm"


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_price(
    request: CarFeatures,
):

    result = predict(
        model_name=MODEL_NAME,
        payload=request.model_dump(),
    )

    return PredictionResponse(**result)

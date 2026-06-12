import os
import requests
import gradio as gr

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/predict",
)


def predict_price(
    mileage,
    tax,
    mpg,
    engineSize,
    model_name,
    transmission,
    fuelType,
    make,
    year,
):

    payload = {
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize,
        "model": model_name or None,
        "transmission": transmission or None,
        "fuelType": fuelType or None,
        "Make": make or None,
        "year": int(year) if year not in (None, "") else None,
    }

    payload = {k: (None if v == "" else v) for k, v in payload.items()}

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        raise gr.Error("Please provide at least 5 fields to predict the car price.")
    return response.json()["predicted_price"]


ui = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Mileage"),
        gr.Number(label="Tax"),
        gr.Number(label="MPG"),
        gr.Number(label="Engine Size"),
        gr.Textbox(label="Model"),
        gr.Textbox(label="Transmission"),
        gr.Textbox(label="Fuel Type"),
        gr.Textbox(label="Make"),
        gr.Number(label="Year"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Car Price Prediction",
)

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)

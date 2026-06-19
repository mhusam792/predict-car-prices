# 🚗 Predict Car Prices

End-to-end Machine Learning project for predicting used car prices using modern MLOps practices.

## Overview

This project demonstrates the complete lifecycle of a machine learning system:

* Data versioning with DVC
* Experiment tracking with MLflow
* Model training using LightGBM and XGBoost
* Model serving through FastAPI
* Interactive user interface with Gradio
* Artifact storage using MinIO
* Containerized deployment using Docker Compose

---

## Project Architecture

```text
                    ┌───────────────┐
                    │  Raw Dataset  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Data Pipeline │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Model Training│
                    │ LightGBM      │
                    │ XGBoost       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    MLflow     │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
      PostgreSQL                       MinIO
   (Metadata Store)              (Artifact Store)

                            │
                            ▼
                    ┌───────────────┐
                    │ FastAPI       │
                    │ Prediction API│
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Gradio UI     │
                    └───────────────┘
```

---

## Tech Stack

### Machine Learning

* Python 3.12
* Scikit-Learn
* LightGBM
* XGBoost
* Feature Engine

### MLOps

* MLflow
* DVC
* Docker
* Docker Compose
* MinIO
* PostgreSQL

### Serving

* FastAPI
* Uvicorn
* Gradio

---

## Repository Structure

```text
.
├── datasets/
│   └── raw/
├── notebook/
├── src/
│   ├── api/
│   ├── build_model/
│   ├── ui/
│   └── utils/
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── Dockerfile.mlflow
├── .dvc/
├── docker-compose.yaml
├── pyproject.toml
└── Makefile
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/mhusam792/predict-car-prices.git
cd predict-car-prices
```

### Install Dependencies

```bash
uv sync
```

or

```bash
pip install -e .
```

---

## Run MLflow Infrastructure

```bash
docker compose up -d postgres minio mlflow
```

Services:

| Service       | URL                   |
| ------------- | --------------------- |
| MLflow        | http://localhost:5000 |
| MinIO API     | http://localhost:9000 |
| MinIO Console | http://localhost:9001 |

---

## Train Model

```bash
docker compose run --rm build_model
```

or

```bash
python -m src.train
```

---

## Run Prediction API

```bash
docker compose up api
```

API Documentation:

```text
http://localhost:8000/docs
```

---

## Run Gradio UI

```bash
docker compose up ui
```

UI:

```text
http://localhost:7860
```

---

## Example Prediction Request

```json
{
  "mileage": 2071,
  "tax": 145,
  "mpg": 33.6,
  "engineSize": 2.0,
  "model": "Caravelle",
  "transmission": "Automatic",
  "fuelType": "Diesel",
  "Make": "vw",
  "year": 2019
}
```

---

## Features

* End-to-end ML workflow
* Experiment tracking with MLflow
* Artifact management with MinIO
* Reproducible training pipeline
* Dockerized deployment
* FastAPI inference service
* Interactive Gradio dashboard
* DVC data versioning

---

## Future Improvements

* CI/CD with GitHub Actions
* Automated model retraining
* Monitoring and drift detection
* Kubernetes deployment
* Model registry promotion workflow

---

## Author

Mohamed Hussam

AI/ML Engineer

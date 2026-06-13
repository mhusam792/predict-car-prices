api: 
	uvicorn src.api.app:app --reload

ui: 
	python -m src.ui.app

train:
	python -m src.train

	
# # To build container for first time
# docker build \
#   -f docker/Dockerfile.base \
#   -t car-price-base:latest .

# docker compose up --build
# docker compose down -v
# docker exec -it car_prices_ui bash

# apt-get update && apt-get install -y \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# mlflow ui --backend-store-uri sqlite:///datasets/mlflow/database/mlflow.db


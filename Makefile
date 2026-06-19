api: 
	uvicorn src.api.app:app --reload

ui: 
	python -m src.ui.app

train:
	python -m src.build_model.train

	
# To build container for first time
build_base_image:
	docker build \
	-f docker/Dockerfile.base \
	-t car-price-base .

run_model_container:
	docker run -it --rm \
	--name build_model \
	-v artifacts:/app/artifacts \
	car-price-base


# Building api image and container
build_api_image:
	docker build \
	-f docker/Dockerfile.api \
	-t car-reg-api .

run_api_container:
	docker run -it --rm \
	--name car_reg_api_app \
	-p 8000:8000 \
	-v artifacts:/app/artifacts \
	--network car_price_net \
	car-reg-api


# Building ui image and container
build_ui_image:
	docker build \
	-f docker/Dockerfile.ui \
	-t car-reg-ui .

run_ui_container:
	docker run -it --rm \
	--name car_reg_ui \
	-p 7860:7860 \
	--network car_price_net \
	car-reg-ui

# docker compose up --build
# docker compose down -v
# docker exec -it car_prices_ui bash

# apt-get update && apt-get install -y \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# mlflow ui --backend-store-uri sqlite:///datasets/mlflow/database/mlflow.db


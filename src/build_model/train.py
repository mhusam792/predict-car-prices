from src.build_model.config.loader import load_config
from src.build_model.models.training import train
from src.utils.helpers import setup_logger


def main() -> None:
    setup_logger()
    config = load_config()
    metrics = train(config)
    print(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()

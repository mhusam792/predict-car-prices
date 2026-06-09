from src.config.core import load_config
from src.models.trainer import Trainer
from src.utils.logger import setup_logger


def main():
    setup_logger()

    config = load_config("src/configs/data.yml")

    trainer = Trainer(config)
    trainer.train()

    print("Training completed")


if __name__ == "__main__":
    main()

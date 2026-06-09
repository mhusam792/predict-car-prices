from pathlib import Path

from src.utils.helpers import save_object


class ModelRegistry:
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
    ):
        self.artifacts_dir = Path(artifacts_dir)

    def save(
        self,
        model,
        pipeline,
    ) -> None:

        save_object(
            model,
            self.artifacts_dir / "model.joblib",
        )

        save_object(
            pipeline,
            self.artifacts_dir / "pipeline.joblib",
        )

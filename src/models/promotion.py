from src.tracking.registry import ModelRegistry


class ModelPromotionService:
    def __init__(self, model_name: str):
        self.registry = ModelRegistry()
        self.model_name = model_name

    def promote_if_better(
        self,
        new_version: str,
        new_metric: float,
        metric_name: str = "test_r2",
    ) -> str:

        production_model = self.registry.get_production_model(self.model_name)

        # ----------------------------
        # First model in registry
        # ----------------------------
        if production_model is None:

            self.registry.transition_model_version(
                self.model_name,
                new_version,
                "Production",
            )

            return "first_model_promoted"

        # ----------------------------
        # Get production metric
        # ----------------------------
        production_run = self.registry.client.get_run(production_model.run_id)

        production_metric = production_run.data.metrics.get(
            metric_name,
            float("-inf"),
        )

        print("\n========== PROMOTION DEBUG ==========")
        print(f"new_version={new_version}")
        print(f"production_version={production_model.version}")
        print(f"production_metric={production_metric}")
        print(f"new_metric={new_metric}")
        print("=====================================\n")

        # ----------------------------
        # Promote if better
        # ----------------------------
        if new_metric > production_metric:

            self.registry.transition_model_version(
                self.model_name,
                production_model.version,
                "Archived",
            )

            self.registry.transition_model_version(
                self.model_name,
                new_version,
                "Production",
            )

            return "promoted_new_model"

        # ----------------------------
        # Otherwise staging
        # ----------------------------
        self.registry.transition_model_version(
            self.model_name,
            new_version,
            "Staging",
        )

        return "sent_to_staging"

from mlflow.tracking import MlflowClient
import mlflow


class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        model_name: str,
    ):
        return mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

    def transition_model_version(
        self,
        model_name: str,
        version: str,
        stage: str,
    ):
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

    def get_best_model(
        self,
        model_name: str,
        metric_name: str = "test_r2",
    ):

        versions = self.client.search_model_versions(f"name='{model_name}'")

        best_model = None
        best_score = float("-inf")

        for version in versions:

            run = self.client.get_run(version.run_id)

            metric_value = run.data.metrics.get(metric_name)

            if metric_value is None:
                continue

            if metric_value > best_score:
                best_score = metric_value
                best_model = version

        return best_model, best_score

    def compare_with_previous_runs(
        self,
        model_name: str,
        metric_name: str = "test_r2",
    ):

        versions = self.client.search_model_versions(f"name='{model_name}'")

        results = []

        for version in versions:

            run = self.client.get_run(version.run_id)

            results.append(
                {
                    "version": version.version,
                    "run_id": version.run_id,
                    metric_name: run.data.metrics.get(metric_name),
                    "stage": version.current_stage,
                }
            )

        return sorted(
            results,
            key=lambda x: (
                x[metric_name] if x[metric_name] is not None else float("-inf")
            ),
            reverse=True,
        )

    def get_production_model(self, model_name: str):

        versions = self.client.search_model_versions(f"name='{model_name}'")

        print("\n==== MODEL VERSIONS ====")

        for version in versions:

            print(f"version={version.version}, " f"stage={version.current_stage}")

            if version.current_stage == "Production":
                return version

        print("========================\n")

        return None

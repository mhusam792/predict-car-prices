import logging

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alias constants  (single source of truth)
# ---------------------------------------------------------------------------

CHAMPION = "champion"
CHALLENGER = "challenger"
ARCHIVED = "archived"

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def register_model(run_id: str, model_name: str) -> str:
    """
    Register the model logged in `run_id` under `model_name`.
    Returns the new version string.
    """
    model_uri = f"runs:/{run_id}/model"
    version_info = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(version_info.version)
    logger.info("Registered model '%s' version %s", model_name, version)
    return version


def set_alias(
    client: MlflowClient,
    model_name: str,
    version: str,
    alias: str,
) -> None:
    """Assign an alias to a model version."""
    client.set_registered_model_alias(model_name, alias, version)
    logger.info("'%s' v%s → @%s", model_name, version, alias)


def delete_alias(
    client: MlflowClient,
    model_name: str,
    alias: str,
) -> None:
    """Remove an alias (silently ignored if it doesn't exist)."""
    try:
        client.delete_registered_model_alias(model_name, alias)
    except MlflowException:
        pass


def get_champion_version(client: MlflowClient, model_name: str):
    """Return the ModelVersion tagged @champion, or None."""
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        return None


def get_version_metric(
    client: MlflowClient,
    version,
    metric_name: str,
) -> float:
    run = client.get_run(version.run_id)
    return run.data.metrics.get(metric_name, float("-inf"))


def list_versions_by_metric(
    client: MlflowClient,
    model_name: str,
    metric_name: str = "test_r2",
) -> list[dict]:
    """Return all versions sorted by metric descending (useful for audits)."""
    versions = client.search_model_versions(f"name='{model_name}'")
    rows = []
    for v in versions:
        run = client.get_run(v.run_id)
        rows.append(
            {
                "version": v.version,
                "run_id": v.run_id,
                metric_name: run.data.metrics.get(metric_name),
                "aliases": v.aliases,
            }
        )
    return sorted(
        rows,
        key=lambda x: x[metric_name] if x[metric_name] is not None else float("-inf"),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Promotion logic
# ---------------------------------------------------------------------------

PromotionDecision = (
    str  # "first_model_promoted" | "promoted_new_model" | "sent_to_challenger"
)


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    new_metric: float,
    metric_name: str = "test_r2",
) -> PromotionDecision:
    """
    Compare new_version against the current @champion model.

    Rules
    -----
    - No champion yet          → assign @champion directly.
    - new_metric > champion    → move old champion to @archived, promote new to @champion.
    - new_metric ≤ champion    → assign @challenger to new version.
    """
    champion = get_champion_version(client, model_name)

    if champion is None:
        set_alias(client, model_name, new_version, CHAMPION)
        logger.info("No prior champion — promoting v%s directly.", new_version)
        return "first_model_promoted"

    champion_metric = get_version_metric(client, champion, metric_name)

    logger.info(
        "Promotion check | new=v%s (%.4f) vs champion=v%s (%.4f)",
        new_version,
        new_metric,
        champion.version,
        champion_metric,
    )

    if new_metric > champion_metric:
        # Retire old champion → archived, promote new → champion
        delete_alias(client, model_name, CHAMPION)
        set_alias(client, model_name, champion.version, ARCHIVED)
        set_alias(client, model_name, new_version, CHAMPION)
        return "promoted_new_model"

    set_alias(client, model_name, new_version, CHALLENGER)
    return "sent_to_challenger"

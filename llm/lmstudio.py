"""Model discovery for LM Studio; independent of model families and publishers."""

from typing import Any


def _normalized(value: object) -> str:
    return str(value or "").strip().casefold()


def _match_score(configured: str, candidate: object) -> int:
    candidate = _normalized(candidate)
    if not candidate:
        return 0
    if configured == candidate:
        return 4
    if configured.rsplit("/", 1)[-1] == candidate.rsplit("/", 1)[-1]:
        return 3
    # An unqualified model may follow its loaded variant. Explicit @variants
    # never match a different quantization by stripping their suffix.
    if "@" not in configured:
        base = candidate.split("@", 1)[0]
        if configured == base:
            return 2
        if configured.rsplit("/", 1)[-1] == base.rsplit("/", 1)[-1]:
            return 1
    return 0


def select_model(
    configured: str, models: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Resolve a catalog key or custom instance ID without guessing on ties."""
    configured = _normalized(configured)
    if not configured:
        return None, None, "Set LOCAL_LLM_MODEL to an LM Studio model identifier"
    matches = []
    for model in models:
        instances = model.get("loaded_instances") or []
        ids = [model.get("key"), model.get("selected_variant")]
        quantization = (model.get("quantization") or {}).get("name")
        if quantization and "@" not in str(model.get("key", "")):
            ids.append(f"{model.get('key')}@{quantization}")
        catalog_score = max(_match_score(configured, value) for value in ids)
        for instance in instances or [None]:
            score = max(
                catalog_score,
                _match_score(configured, (instance or {}).get("id")),
            )
            if score:
                matches.append((score, bool(instance), model, instance))
    if not matches:
        return None, None, "Configured model is not installed in LM Studio; copy its exact API identifier"
    best_rank = max((score, loaded) for score, loaded, _, _ in matches)
    best = [item for item in matches if item[:2] == best_rank]
    # Multiple loaded instances of one catalog model are interchangeable only
    # if the user supplied its catalog key. Choose the first in that case.
    if len({id(item[2]) for item in best}) > 1:
        return None, None, "Model name is ambiguous; use the exact LM Studio key including its @variant or a custom instance ID"
    _, _, model, instance = best[0]
    if model.get("type", "llm") != "llm":
        return None, None, "Configured model is an embedding model, not a chat LLM"
    return model, instance, ""


def reasoning_effort(model: dict[str, Any]) -> str | None:
    """Use only a setting advertised by this model's current runtime."""
    options = ((model.get("capabilities") or {}).get("reasoning") or {}).get(
        "allowed_options", []
    )
    for option in ("off", "low", "medium", "high"):
        if option in options:
            return "none" if option == "off" else option
    return None

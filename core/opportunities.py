"""Pre-model opportunity screening using the same deterministic entry rules."""

from typing import Any

from app_config.settings import settings
from core.evidence import build_evidence_ids, permitted_entry_actions
from risk.manager import RiskManager


def screen_opportunities(analyses: dict[str, Any], capital_fit: dict[str, Any], history=None) -> dict[str, Any]:
    frames = [analyses.get(timeframe) for timeframe in ("M5", "M15", "H1", "H4")]
    actions = permitted_entry_actions(build_evidence_ids(analyses))
    viable = []
    rejected = {}
    modes = {}
    for action in actions:
        direction_fit = (capital_fit.get("directions") or {}).get(action)
        if direction_fit is not None and not direction_fit.get("capital_fit"):
            rejected[action] = "The technical plan for this direction does not fit the account"
            continue
        mode = RiskManager._resolve_strategy_mode(action, {}, *frames)
        alternatives = [mode]
        if history and RiskManager.qualify_failed_thesis_reversal(
            action, settings.failed_thesis_reversal_min_confidence, history, frames[0], frames[1]
        )[0]:
            alternatives.append("FAILED_THESIS_REVERSAL")
        for candidate_mode in alternatives:
            # This is an optimistic feasibility check, not model confidence.
            # The final gate reruns with the actual confidence and live quote.
            allowed, reason = RiskManager._check_entry_structure(
                action, *frames, strategy_mode=candidate_mode, decision_confidence=1.0,
            )
            if allowed:
                viable.append(action)
                modes[action] = candidate_mode
                break
        if action not in viable:
            rejected[action] = reason
    return {
        "actionable_entry_evidence": bool(actions),
        "permitted_entry_actions": list(actions),
        "viable_entry_actions": viable,
        "opportunity_rejections": rejected,
        "opportunity_modes": modes,
        "model_eligible": bool(capital_fit.get("capital_fit") and capital_fit.get("broker_open", True) and viable),
    }

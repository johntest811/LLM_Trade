"""Strict validation for model-proposed trading decisions."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

from app_config.settings import settings
from core.evidence import (
    has_directional_m5_trigger,
    has_directional_trigger,
    permitted_entry_actions,
    resolve_evidence_aliases,
    unknown_evidence_ids,
)

logger = logging.getLogger("TradingSystem.DecisionValidator")


class DecisionValidator:
    """Normalize model output without trusting it with position sizing."""

    @staticmethod
    def validate_decision(
        raw_output: Any,
        allowed_evidence_ids: Optional[Iterable[str]] = None,
        close_position_side: Optional[str] = None,
        close_trigger_timeframes: Iterable[str] = ("M5",),
        permitted_actions: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        if not raw_output:
            return False, None, "Empty LLM output response."
        if isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output.strip())
            except Exception as exc:
                return False, None, f"Failed to parse LLM output as JSON: {exc}"
        elif isinstance(raw_output, dict):
            parsed = dict(raw_output)
        else:
            return False, None, f"Unsupported output format: {type(raw_output)}"

        action = str(parsed.get("action", "")).upper().strip()
        if action not in {"BUY", "SELL", "HOLD", "CLOSE"}:
            return False, parsed, f"Invalid action '{action}'."
        parsed["action"] = action

        try:
            confidence = float(parsed.get("confidence", 0.0))
            if confidence > 1.0:
                confidence /= 100.0
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("outside 0..1")
            parsed["confidence"] = confidence
        except (TypeError, ValueError):
            return False, parsed, "Confidence must be between 0 and 1."

        parsed["reasoning"] = str(parsed.get("reasoning", ""))[:500]
        parsed["trade_management"] = str(parsed.get("trade_management", ""))[:500]
        evidence = parsed.get("evidence_ids", [])
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list) or any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in evidence
        ):
            return False, parsed, "evidence_ids must be a list of non-empty strings."
        evidence = list(dict.fromkeys(identifier.strip() for identifier in evidence))[:8]
        if action == "HOLD":
            # HOLD cannot change broker state, so discard model commentary that
            # was accidentally placed in the machine-verifiable evidence field.
            evidence = []
        elif allowed_evidence_ids is not None:
            original_evidence = tuple(evidence)
            evidence = list(
                resolve_evidence_aliases(
                    evidence,
                    allowed_evidence_ids,
                    default_timeframe="M5" if action in {"BUY", "SELL"} else None,
                )
            )
            if tuple(evidence) != original_evidence:
                logger.info(
                    "Canonicalized model evidence aliases: %s -> %s",
                    list(original_evidence),
                    evidence,
                )
        parsed["evidence_ids"] = evidence
        if allowed_evidence_ids is not None:
            unknown = unknown_evidence_ids(evidence, allowed_evidence_ids)
            if unknown:
                return (
                    False,
                    parsed,
                    "Decision cited unavailable evidence: " + ", ".join(unknown),
                )
            if action in {"BUY", "SELL"} and not has_directional_m5_trigger(
                evidence, action
            ):
                return (
                    False,
                    parsed,
                    f"{action} requires a supplied directional M5 BOS, CHoCH, "
                    "breakout, pullback-retest, or validated range evidence ID.",
                )
            if action in {"BUY", "SELL"}:
                bounded_actions = tuple(
                    dict.fromkeys(
                        str(value).upper()
                        for value in (
                            permitted_actions
                            if permitted_actions is not None
                            else permitted_entry_actions(allowed_evidence_ids)
                        )
                        if str(value).upper() in {"BUY", "SELL"}
                    )
                )
                if action not in bounded_actions:
                    label = ", ".join(bounded_actions) or "HOLD only"
                    return (
                        False,
                        parsed,
                        f"{action} is outside the deterministic entry contract; "
                        f"permitted action(s): {label}.",
                    )
            if action == "CLOSE" and close_position_side:
                opposing_action = {
                    "BUY": "SELL",
                    "SELL": "BUY",
                }.get(str(close_position_side).upper())
                normalized_close_timeframes = tuple(
                    dict.fromkeys(
                        str(value).upper() for value in close_trigger_timeframes
                    )
                ) or ("M5",)
                if opposing_action and not has_directional_trigger(
                    evidence,
                    opposing_action,
                    timeframes=normalized_close_timeframes,
                ):
                    return (
                        False,
                        parsed,
                        "CLOSE requires a verified opposing "
                        + "/".join(normalized_close_timeframes)
                        + " BOS, CHoCH, or breakout evidence ID.",
                    )
        # Position risk is user configuration, not a model decision. Ignore the
        # legacy model field and attach the deterministic value for audit logs.
        parsed["risk_percentage"] = settings.risk_percent

        if action == "HOLD":
            parsed.update(entry=None, stop_loss=None, take_profit=None, ticket_to_close=None)
            return True, parsed, ""

        if action == "CLOSE":
            try:
                ticket = int(parsed.get("ticket_to_close"))
                if ticket <= 0:
                    raise ValueError
                parsed["ticket_to_close"] = ticket
            except (TypeError, ValueError):
                return False, parsed, "CLOSE requires a valid ticket_to_close."
            parsed.update(entry=None, stop_loss=None, take_profit=None)
            return True, parsed, ""

        # Model-proposed prices and volume are intentionally discarded. The
        # deterministic planner derives current broker-valid levels after the
        # direction has been validated.
        parsed.update(entry=None, stop_loss=None, take_profit=None, ticket_to_close=None)
        parsed.pop("lot_size", None)
        return True, parsed, ""

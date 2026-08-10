"""Canonical, machine-verifiable identifiers for completed-candle evidence."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, Iterable, Mapping, Tuple


TIMEFRAMES: Tuple[str, ...] = ("M1", "M5", "M15", "H1", "H4")
_DIRECTIONAL_ALIAS = re.compile(
    r"^(?:(M1|M5|M15|H1|H4)_)?"
    r"(?:(BULLISH|BEARISH)_(BOS|CHOCH|BREAKOUT|RETEST)|"
    r"(BOS|CHOCH|BREAKOUT|RETEST)_(BULLISH|BEARISH))$"
)


def _direction(analysis: Mapping[str, Any]) -> str:
    structure = analysis.get("market_structure", {}) or {}
    value = (
        structure.get("trend_state_direction")
        or structure.get("trend")
        or "NEUTRAL"
    )
    normalized = str(value).upper()
    return normalized if normalized in {"BULLISH", "BEARISH"} else "NEUTRAL"


def _time_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNDATED"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%Y%m%dT%H%M")
    except ValueError:
        return "".join(character for character in text.upper() if character.isalnum())[:20] or "UNDATED"


def build_evidence_ids(analyses: Mapping[str, Mapping[str, Any]]) -> Tuple[str, ...]:
    """Return bounded facts the model may cite in a directional decision.

    The identifiers are generated from deterministic analysis output. A model
    cannot authorize an order by inventing a BOS, CHoCH, breakout, retest, or
    timeframe alignment that is absent from this catalog.
    """
    identifiers = []
    for timeframe in TIMEFRAMES:
        analysis = analyses.get(timeframe) or {}
        direction = _direction(analysis)
        if direction != "NEUTRAL":
            identifiers.append(f"{timeframe}_TREND_{direction}")

        structure = analysis.get("market_structure", {}) or {}
        range_setup = structure.get("range_reversion", {}) or {}
        if isinstance(range_setup, dict) and range_setup.get("eligible"):
            range_direction = str(range_setup.get("direction", "")).upper()
            evidence_id = str(range_setup.get("evidence_id", "")).strip()
            if range_direction in {"BUY", "SELL"} and evidence_id:
                identifiers.append(evidence_id)
        retest = structure.get("retest_continuation")
        if isinstance(retest, dict):
            retest_direction = str(retest.get("direction", "")).upper()
            if retest_direction in {"BULLISH", "BEARISH"}:
                identifiers.append(
                    f"{timeframe}_RETEST_{retest_direction}_"
                    f"{_time_token(retest.get('time') or analysis.get('timestamp'))}"
                )

        for event in structure.get("structure_events", []) or []:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "")).upper()
            event_direction = str(event.get("direction", "")).upper()
            if (
                event_type not in {"BOS", "CHOCH"}
                or event_direction not in {"BULLISH", "BEARISH"}
            ):
                continue
            identifiers.append(
                f"{timeframe}_{event_type}_{event_direction}_"
                f"{_time_token(event.get('time') or analysis.get('timestamp'))}"
            )

        breakout = str(structure.get("breakout_status", "")).upper()
        for breakout_direction in ("BULLISH", "BEARISH"):
            if breakout_direction in breakout and "BREAKOUT" in breakout:
                identifiers.append(f"{timeframe}_BREAKOUT_{breakout_direction}")

    return tuple(dict.fromkeys(identifiers))


def unknown_evidence_ids(
    supplied: Iterable[str], allowed: Iterable[str]
) -> Tuple[str, ...]:
    allowed_set = {str(identifier) for identifier in allowed}
    return tuple(
        identifier
        for identifier in dict.fromkeys(str(value) for value in supplied)
        if identifier not in allowed_set
    )


def resolve_evidence_aliases(
    supplied: Iterable[str],
    allowed: Iterable[str],
    *,
    default_timeframe: str | None = None,
) -> Tuple[str, ...]:
    """Resolve bounded directional aliases to evidence that actually exists.

    Local models occasionally copy a descriptive analysis tag such as
    ``BEARISH_BREAKOUT`` instead of the canonical
    ``M5_BREAKOUT_BEARISH`` identifier. This function repairs only that token
    ordering/timeframe mistake. It cannot create evidence: a single matching
    canonical identifier must already be present in ``allowed``.

    ``default_timeframe`` is intentionally supplied only by entry validation,
    where the deterministic entry contract requires a completed M5 trigger.
    Ambiguous or invented aliases remain unchanged and are rejected later by
    :func:`unknown_evidence_ids`.
    """
    allowed_ids = tuple(dict.fromkeys(str(identifier) for identifier in allowed))
    allowed_set = set(allowed_ids)
    normalized_default = str(default_timeframe or "").upper()
    if normalized_default not in TIMEFRAMES:
        normalized_default = ""

    resolved = []
    for raw_identifier in supplied:
        identifier = str(raw_identifier).strip()
        if identifier in allowed_set:
            resolved.append(identifier)
            continue

        normalized = re.sub(r"[^A-Z0-9]+", "_", identifier.upper()).strip("_")
        match = _DIRECTIONAL_ALIAS.fullmatch(normalized)
        if not match:
            resolved.append(identifier)
            continue

        timeframe = match.group(1) or normalized_default
        direction = match.group(2) or match.group(5)
        event_type = match.group(3) or match.group(4)
        if not timeframe:
            resolved.append(identifier)
            continue

        if event_type == "BREAKOUT":
            canonical_prefix = f"{timeframe}_BREAKOUT_{direction}"
            candidates = [
                value for value in allowed_ids if value == canonical_prefix
            ]
        else:
            canonical_prefix = f"{timeframe}_{event_type}_{direction}_"
            candidates = [
                value for value in allowed_ids if value.startswith(canonical_prefix)
            ]

        resolved.append(candidates[0] if len(candidates) == 1 else identifier)

    return tuple(dict.fromkeys(resolved))


def has_directional_trigger(
    evidence_ids: Iterable[str],
    action: str,
    *,
    timeframes: Iterable[str] = ("M5",),
) -> bool:
    direction = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(str(action).upper())
    if direction is None:
        return False
    prefixes = tuple(
        prefix
        for timeframe in dict.fromkeys(str(value).upper() for value in timeframes)
        for prefix in (
            f"{timeframe}_BOS_{direction}_",
            f"{timeframe}_CHOCH_{direction}_",
            f"{timeframe}_BREAKOUT_{direction}",
            f"{timeframe}_RETEST_{direction}_",
        )
    )
    return any(str(identifier).startswith(prefixes) for identifier in evidence_ids)


def has_directional_m5_trigger(evidence_ids: Iterable[str], action: str) -> bool:
    """Entry gate supporting trend triggers and entry-only range evidence."""
    direction = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(
        str(action).upper()
    )
    if direction is None:
        return False
    return has_directional_trigger(
        evidence_ids, action, timeframes=("M5",)
    ) or any(
        str(identifier).startswith(f"M5_RANGE_{direction}_")
        for identifier in evidence_ids
    )

"""Bounded deterministic exceptions for slightly declining entry momentum."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _structure(analysis: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    value = (analysis or {}).get("market_structure", {}) or {}
    return value if isinstance(value, Mapping) else {}


def _direction(analysis: Optional[Mapping[str, Any]]) -> str:
    structure = _structure(analysis)
    value = (
        structure.get("trend_state_direction")
        or structure.get("trend")
        or "NEUTRAL"
    )
    normalized = str(value).upper()
    return normalized if normalized in {"BULLISH", "BEARISH"} else "NEUTRAL"


def _has_fresh_directional_trigger(
    analysis: Optional[Mapping[str, Any]], direction: str
) -> bool:
    structure = _structure(analysis)
    events = structure.get("structure_events", []) or []
    has_event = any(
        isinstance(event, Mapping)
        and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
        and str(event.get("direction", "")).upper() == direction
        for event in events
    )
    breakout = str(structure.get("breakout_status", "")).upper()
    has_breakout = direction in breakout and "BREAKOUT" in breakout
    retest = structure.get("retest_continuation")
    has_retest = bool(
        isinstance(retest, Mapping)
        and str(retest.get("direction", "")).upper() == direction
    )
    return has_event or has_breakout or has_retest


def aligned_structure_allows_adx_decline(
    m5_analysis: Mapping[str, Any],
    m15_analysis: Optional[Mapping[str, Any]] = None,
    h1_analysis: Optional[Mapping[str, Any]] = None,
    *,
    action: Optional[str] = None,
    max_decline: float,
    min_m5_adx: float,
    min_m15_adx: float,
    require_confirmation_alignment: bool = True,
) -> bool:
    """Allow only a bounded ADX dip inside a fresh aligned structure.

    This is not a general weak-momentum bypass.  It requires a deterministic
    M5 trigger, a still-strong M5 ADX, and (for live entry approval) matching
    M5/M15/H1 direction plus sufficient M15 ADX.  Market discovery may use the
    provisional M5-only form because it has not loaded confirmation frames yet;
    the live risk gate always repeats the complete aligned check.
    """
    expected = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(
        str(action or "").upper()
    ) or _direction(m5_analysis)
    if expected not in {"BULLISH", "BEARISH"}:
        return False

    indicators = (m5_analysis or {}).get("indicators", {}) or {}
    adx = _number(indicators.get("adx_14"))
    delta = _number(indicators.get("adx_delta"))
    if (
        adx is None
        or delta is None
        or adx < min_m5_adx
        or delta >= 0.0
        or delta < -max(0.0, max_decline)
        or _direction(m5_analysis) != expected
        or not _has_fresh_directional_trigger(m5_analysis, expected)
    ):
        return False

    if not require_confirmation_alignment:
        return True
    if not m15_analysis or not h1_analysis:
        return False
    m15_adx = _number(
        ((m15_analysis or {}).get("indicators", {}) or {}).get("adx_14")
    )
    return bool(
        _direction(m15_analysis) == expected
        and _direction(h1_analysis) == expected
        and m15_adx is not None
        and m15_adx >= min_m15_adx
    )

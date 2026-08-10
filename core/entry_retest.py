"""Deterministic, one-candle pullback-resumption evidence."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from app_config.settings import settings


def _structure(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    value = analysis.get("market_structure", {}) or {}
    return value if isinstance(value, dict) else {}


def _direction(analysis: Mapping[str, Any]) -> str:
    structure = _structure(analysis)
    value = (
        structure.get("trend_state_direction")
        or structure.get("trend")
        or "NEUTRAL"
    )
    normalized = str(value).upper()
    return normalized if normalized in {"BULLISH", "BEARISH"} else "NEUTRAL"


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def annotate_retest_continuation(
    m5_analysis: Dict[str, Any],
    m15_analysis: Mapping[str, Any],
    h1_analysis: Mapping[str, Any],
    previous_trend_states: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Attach a fresh M5 retest event after an exact pullback transition.

    The event exists only on the first completed candle that changes from
    ``PULLBACK_IN_<direction>_TREND`` to ``CONFIRMED_<direction>``.  It is not
    inferred from model prose and cannot persist into a later candle.
    """
    structure = _structure(m5_analysis)
    structure.pop("retest_continuation", None)
    if not settings.retest_continuation_enabled:
        return None

    current_state = str(structure.get("trend_state", "NEUTRAL")).upper()
    previous_state = str(previous_trend_states.get("M5", "NEUTRAL")).upper()
    direction = _direction(m5_analysis)
    if direction not in {"BULLISH", "BEARISH"}:
        return None
    if (
        previous_state != f"PULLBACK_IN_{direction}_TREND"
        or current_state != f"CONFIRMED_{direction}"
    ):
        return None
    if any(
        _direction(analysis) != direction
        for analysis in (m5_analysis, m15_analysis, h1_analysis)
    ):
        return None

    m5_indicators = m5_analysis.get("indicators", {}) or {}
    m15_indicators = m15_analysis.get("indicators", {}) or {}
    m5_adx = _number(m5_indicators.get("adx_14"))
    m15_adx = _number(m15_indicators.get("adx_14"))
    if (
        m5_adx is None
        or m15_adx is None
        or m5_adx < settings.entry_min_adx
        or m15_adx < settings.confirmation_min_adx
    ):
        return None

    adx_delta = _number(m5_indicators.get("adx_delta"))
    if (
        settings.entry_require_adx_rising
        and adx_delta is not None
        and adx_delta < -settings.entry_adx_decline_tolerance
    ):
        return None

    multiplier = 1.0 if direction == "BULLISH" else -1.0
    directional_values = []
    for key in ("candle_body_atr_signed", "candle_return_atr"):
        value = _number(m5_indicators.get(key))
        if value is not None:
            directional_values.append(multiplier * value)
    if not directional_values:
        return None
    resumption_atr = max(directional_values)
    if resumption_atr < settings.retest_min_resumption_atr:
        return None

    candle_range_atr = _number(m5_indicators.get("candle_range_atr"))
    if (
        settings.entry_max_candle_range_atr > 0
        and candle_range_atr is not None
        and candle_range_atr > settings.entry_max_candle_range_atr
        and resumption_atr > settings.entry_max_candle_range_atr
    ):
        return None

    event = {
        "direction": direction,
        "time": m5_analysis.get("timestamp"),
        "previous_state": previous_state,
        "current_state": current_state,
        "resumption_atr": round(resumption_atr, 4),
        "m5_adx": round(m5_adx, 2),
        "m15_adx": round(m15_adx, 2),
    }
    structure["retest_continuation"] = event
    return event

"""Fresh, completed-candle pullback-resumption evidence shared by all lanes."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from app_config.settings import settings
from core.entry_momentum import aligned_structure_allows_adx_decline


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
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _price_pullback(
    analysis: Mapping[str, Any], bars: Optional[pd.DataFrame],
    h4: Optional[Mapping[str, Any]], direction: str,
) -> Optional[Dict[str, Any]]:
    """Require a counter-move followed by a close through its anchored boundary.

    No synthetic BOS, intrabar signal, stale retest, or trend-label-only entry
    is created. The caller supplies only completed M5 bars; timestamps and
    gaps are checked again here so replay and discovery use identical facts.
    """
    if (
        not settings.price_pullback_enabled or bars is None or len(bars) < 3
        or bars.attrs.get("is_stale", False) or not h4
        or _direction(h4) != direction
        or str(_structure(analysis).get("trend_state", "")).upper()
        != f"CONFIRMED_{direction}"
    ):
        return None
    fields = ("open", "high", "low", "close")
    if not set((*fields, "time")).issubset(bars.columns):
        return None
    recent = bars.tail(settings.price_pullback_lookback_bars + 1)
    try:
        times = pd.to_datetime(recent["time"], utc=True, errors="coerce")
        stamp = pd.to_datetime(analysis.get("timestamp"), utc=True, errors="coerce")
        if (
            times.isna().any() or pd.isna(stamp) or times.iloc[-1] != stamp
            or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all()
        ):
            return None
        rows = [{key: float(row[key]) for key in fields} for row in recent.to_dict("records")]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(
        not all(math.isfinite(value) and value > 0 for value in row.values())
        or row["low"] > min(row["open"], row["close"])
        or row["high"] < max(row["open"], row["close"])
        for row in rows
    ):
        return None
    indicators = analysis.get("indicators", {}) or {}
    atr = _number(indicators.get("atr_14"))
    ema9 = _number(indicators.get("ema_9"))
    ema21 = _number(indicators.get("ema_21"))
    rsi = _number(indicators.get("rsi_14"))
    if atr is None or atr <= 0 or any(value is None for value in (ema9, ema21, rsi)):
        return None
    sign = 1 if direction == "BULLISH" else -1
    current, previous = rows[-1], rows[-2]
    if (
        sign * (ema9 - ema21) <= 0 or sign * (rsi - 50) < 0
        or sign * (current["close"] - ema9) < 0
        or any(
            isinstance(event, Mapping)
            and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
            and str(event.get("direction", "")).upper() not in {direction, ""}
            for event in _structure(analysis).get("structure_events", []) or []
        )
    ):
        return None
    # A pullback can take up to three bars to resume. Anchor the boundary to
    # the most recent counter-trend close and emit only its FIRST close-break;
    # moving the boundary every bar would recycle an old signal indefinitely.
    first = max(1, len(rows) - 1 - settings.price_pullback_max_resumption_bars)
    counter_index = next((
        index for index in range(len(rows)-2, first-1, -1)
        if sign * (rows[index]["close"] - rows[index-1]["close"]) < 0
    ), None)
    if counter_index is None:
        return None
    counter = rows[counter_index]
    level = counter["high"] if sign > 0 else counter["low"]
    buffer = settings.price_pullback_break_buffer_atr * atr
    if any(sign * (row["close"] - level) >= buffer for row in rows[counter_index+1:-1]):
        return None
    extreme = (max(row["high"] for row in rows[:counter_index+1]) if sign > 0
               else min(row["low"] for row in rows[:counter_index+1]))
    depth = sign * (extreme - counter["close"]) / atr
    body = sign * (current["close"] - current["open"]) / atr
    advance = sign * (current["close"] - previous["close"]) / atr
    breakout = sign * (current["close"] - level) / atr
    if not (
        settings.price_pullback_min_depth_atr <= depth <= settings.price_pullback_max_depth_atr
        and body >= settings.retest_min_resumption_atr
        and breakout >= settings.price_pullback_break_buffer_atr
        and max(body, advance) <= settings.price_pullback_max_resumption_atr
        and sign * (current["close"] - counter["close"]) / atr <= settings.price_pullback_max_resumption_atr
        and abs(current["open"] - previous["close"]) / atr <= 0.25
        and (current["high"] - current["low"]) / atr <= settings.entry_max_candle_range_atr
    ):
        return None
    return {
        "kind": "PRICE_PULLBACK_RESUMPTION",
        "pullback_bar": times.iloc[counter_index].isoformat(),
        "bars_since_pullback": len(rows) - 1 - counter_index,
        "break_level": level,
        "pullback_depth_atr": round(depth, 4),
        "resumption_atr": round(max(body, advance), 4),
    }


def annotate_retest_continuation(
    m5_analysis: Dict[str, Any],
    m15_analysis: Mapping[str, Any],
    h1_analysis: Mapping[str, Any],
    previous_trend_states: Mapping[str, Any],
    *,
    completed_bars: Optional[pd.DataFrame] = None,
    h4_analysis: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Attach a label-transition or price-confirmed pullback resumption."""
    structure = _structure(m5_analysis)
    structure.pop("retest_continuation", None)
    if not settings.retest_continuation_enabled:
        return None
    if completed_bars is not None:
        try:
            times = pd.to_datetime(completed_bars["time"].tail(2), utc=True, errors="coerce")
            stamp = pd.to_datetime(m5_analysis.get("timestamp"), utc=True, errors="coerce")
            if (
                len(times) < 2 or times.isna().any() or pd.isna(stamp)
                or times.iloc[-1] != stamp
                or times.iloc[-1] - times.iloc[-2] != pd.Timedelta(minutes=5)
            ):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    current_state = str(structure.get("trend_state", "NEUTRAL")).upper()
    previous_state = str(previous_trend_states.get("M5", "NEUTRAL")).upper()
    direction = _direction(m5_analysis)
    if direction not in {"BULLISH", "BEARISH"}:
        return None
    transition = (
        previous_state == f"PULLBACK_IN_{direction}_TREND"
        and current_state == f"CONFIRMED_{direction}"
    )
    price_event = _price_pullback(m5_analysis, completed_bars, h4_analysis, direction)
    if not transition and price_event is None:
        return None
    if (
        completed_bars is not None and completed_bars.attrs.get("is_stale", False)
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
        "kind": "STATE_PULLBACK_RESUMPTION",
        "direction": direction,
        "time": m5_analysis.get("timestamp"),
        "previous_state": previous_state,
        "current_state": current_state,
        "resumption_atr": round(resumption_atr, 4),
        "m5_adx": round(m5_adx, 2),
        "m15_adx": round(m15_adx, 2),
        **(price_event or {}),
    }
    structure["retest_continuation"] = event
    # Use the same bounded exception as discovery and final risk validation.
    # The deterministic candidate must exist before checking its evidence.
    adx_delta = _number(m5_indicators.get("adx_delta"))
    if settings.entry_require_adx_rising and (
        adx_delta is None or (
            adx_delta < -settings.entry_adx_decline_tolerance
            and not aligned_structure_allows_adx_decline(
                m5_analysis, m15_analysis, h1_analysis,
                max_decline=settings.entry_aligned_adx_decline_tolerance,
                min_m5_adx=settings.breakout_min_adx,
                min_m15_adx=settings.confirmation_min_adx,
            )
        )
    ):
        structure.pop("retest_continuation", None)
        return None
    return event

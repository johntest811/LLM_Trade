"""Deterministic trend-transition classification shared by UI and prompts."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def classify_trend_state(
    *,
    slow_trend: str,
    indicators: Dict[str, Any],
    structure_events: Iterable[Dict[str, Any]] | None = None,
    breakout_status: str = "None",
) -> Dict[str, Any]:
    """Classify confirmed regime, pullback, or early reversal without an LLM.

    The slow EMA50/EMA200 regime remains visible, while fast EMA9/EMA21,
    RSI, MACD, ADX, CHoCH and breakouts determine whether price is currently
    confirming that regime, pulling back inside it, or attempting a reversal.
    Only completed-candle inputs should be supplied by callers.
    """
    regime = str(slow_trend or "NEUTRAL").upper()
    if regime not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        regime = "NEUTRAL"

    ema9 = _finite(indicators.get("ema_9"))
    ema21 = _finite(indicators.get("ema_21"))
    rsi = _finite(indicators.get("rsi_14"))
    adx = _finite(indicators.get("adx_14")) or 0.0
    macd = _finite((indicators.get("macd") or {}).get("diff"))

    fast_bullish = bool(
        ema9 is not None
        and ema21 is not None
        and ema9 > ema21
        and rsi is not None
        and rsi >= 52.0
        and macd is not None
        and macd >= 0.0
    )
    fast_bearish = bool(
        ema9 is not None
        and ema21 is not None
        and ema9 < ema21
        and rsi is not None
        and rsi <= 48.0
        and macd is not None
        and macd <= 0.0
    )
    fast_direction = (
        "BULLISH" if fast_bullish else "BEARISH" if fast_bearish else "NEUTRAL"
    )

    events = [event for event in (structure_events or []) if isinstance(event, dict)]
    bullish_choch = any(
        str(event.get("type", "")).upper() == "CHOCH"
        and str(event.get("direction", "")).upper() == "BULLISH"
        for event in events
    )
    bearish_choch = any(
        str(event.get("type", "")).upper() == "CHOCH"
        and str(event.get("direction", "")).upper() == "BEARISH"
        for event in events
    )
    bullish_bos = any(
        str(event.get("type", "")).upper() == "BOS"
        and str(event.get("direction", "")).upper() == "BULLISH"
        for event in events
    )
    bearish_bos = any(
        str(event.get("type", "")).upper() == "BOS"
        and str(event.get("direction", "")).upper() == "BEARISH"
        for event in events
    )
    breakout = str(breakout_status or "None").upper()
    bullish_breakout = "BULLISH BREAKOUT" in breakout
    bearish_breakout = "BEARISH BREAKOUT" in breakout
    strong_momentum = adx >= 20.0

    evidence = []
    if fast_direction != "NEUTRAL":
        evidence.append(f"FAST_{fast_direction}")
    if bullish_choch:
        evidence.append("BULLISH_CHOCH")
    if bearish_choch:
        evidence.append("BEARISH_CHOCH")
    if bullish_bos:
        evidence.append("BULLISH_BOS")
    if bearish_bos:
        evidence.append("BEARISH_BOS")
    if bullish_breakout:
        evidence.append("BULLISH_BREAKOUT")
    if bearish_breakout:
        evidence.append("BEARISH_BREAKOUT")
    if strong_momentum:
        evidence.append("ADX_CONFIRMED")

    bullish_reversal = bullish_choch or (
        fast_bullish and bullish_breakout and strong_momentum
    )
    bearish_reversal = bearish_choch or (
        fast_bearish and bearish_breakout and strong_momentum
    )

    if regime == "BEARISH" and bullish_reversal:
        state, direction = "EARLY_BULLISH_REVERSAL", "BULLISH"
    elif regime == "BULLISH" and bearish_reversal:
        state, direction = "EARLY_BEARISH_REVERSAL", "BEARISH"
    elif regime == "BULLISH" and fast_bearish:
        state, direction = "PULLBACK_IN_BULLISH_TREND", "BULLISH"
    elif regime == "BEARISH" and fast_bullish:
        state, direction = "PULLBACK_IN_BEARISH_TREND", "BEARISH"
    elif regime == "BULLISH":
        state, direction = "CONFIRMED_BULLISH", "BULLISH"
    elif regime == "BEARISH":
        state, direction = "CONFIRMED_BEARISH", "BEARISH"
    elif bullish_reversal or (fast_bullish and strong_momentum):
        state, direction = "EARLY_BULLISH_REVERSAL", "BULLISH"
    elif bearish_reversal or (fast_bearish and strong_momentum):
        state, direction = "EARLY_BEARISH_REVERSAL", "BEARISH"
    else:
        state, direction = "NEUTRAL", "NEUTRAL"

    return {
        "state": state,
        "direction": direction,
        "regime": regime,
        "fast_direction": fast_direction,
        "evidence": evidence,
    }

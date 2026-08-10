"""Deterministic completed-candle qualification for range-reversion entries.

This module does not choose position size or submit orders.  It identifies a
very specific low-ADX setup so the trend-only ADX gate does not discard every
sideways market before the decision provider can review it.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Dict, Mapping, Optional

from app_config.settings import settings


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _direction(analysis: Optional[Mapping[str, Any]]) -> str:
    structure = (analysis or {}).get("market_structure", {}) or {}
    value = structure.get("trend_state_direction") or structure.get("trend")
    normalized = str(value or "NEUTRAL").upper()
    return normalized if normalized in {"BULLISH", "BEARISH"} else "NEUTRAL"


def _adx(analysis: Optional[Mapping[str, Any]]) -> float:
    return _finite((analysis or {}).get("indicators", {}).get("adx_14"), 0.0)


def _time_token(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%Y%m%dT%H%M")
    except ValueError:
        return "".join(ch for ch in text.upper() if ch.isalnum())[:20] or "UNDATED"


def classify_range_reversion(
    m5_analysis: Mapping[str, Any],
    m15_analysis: Optional[Mapping[str, Any]] = None,
    h1_analysis: Optional[Mapping[str, Any]] = None,
    h4_analysis: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a bounded range setup derived entirely from supplied analysis.

    ``candidate`` means the completed M5 candle is worth ranking/model review.
    ``eligible`` additionally requires M15/H1/H4 confirmation and is the only
    state accepted by live risk validation.
    """
    result: Dict[str, Any] = {
        "candidate": False,
        "eligible": False,
        "direction": "",
        "evidence_id": "",
        "target_price": 0.0,
        "invalidation_price": 0.0,
        "score": 0.0,
        "reason": "Range-reversion setup is not present.",
    }
    if not settings.range_reversion_enabled:
        result["reason"] = "Range-reversion strategy is disabled."
        return result

    indicators = (m5_analysis or {}).get("indicators", {}) or {}
    bands = indicators.get("bollinger_bands", {}) or {}
    stochastic = indicators.get("stochastic", {}) or {}
    values = {
        "price": _finite(indicators.get("current_price")),
        "atr": _finite(indicators.get("atr_14")),
        "adx": _finite(indicators.get("adx_14"), 0.0),
        "rsi": _finite(indicators.get("rsi_14")),
        "stoch_k": _finite(stochastic.get("k")),
        "stoch_d": _finite(stochastic.get("d")),
        "body": _finite(indicators.get("candle_body_atr_signed")),
        "lower": _finite(bands.get("lower")),
        "middle": _finite(bands.get("middle")),
        "upper": _finite(bands.get("upper")),
    }
    required = tuple(values.values())
    if not all(math.isfinite(value) for value in required) or values["atr"] <= 0:
        result["reason"] = "Range-reversion indicators are incomplete."
        return result
    if not values["lower"] < values["middle"] < values["upper"]:
        result["reason"] = "Bollinger range is invalid or collapsed."
        return result
    if values["adx"] >= settings.entry_min_adx:
        result["reason"] = "M5 is a trend candidate, not a low-ADX range setup."
        return result

    tolerance = values["atr"] * settings.range_band_tolerance_atr
    buy = (
        values["price"] <= values["lower"] + tolerance
        and values["price"] >= values["lower"] - values["atr"] * settings.range_max_band_overshoot_atr
        and values["rsi"] <= settings.range_buy_max_rsi
        and values["stoch_k"] <= settings.range_buy_max_stoch
        and values["stoch_d"] <= settings.range_buy_max_stoch + 5.0
        and values["body"] >= settings.range_min_reversal_body_atr
    )
    sell = (
        values["price"] >= values["upper"] - tolerance
        and values["price"] <= values["upper"] + values["atr"] * settings.range_max_band_overshoot_atr
        and values["rsi"] >= settings.range_sell_min_rsi
        and values["stoch_k"] >= settings.range_sell_min_stoch
        and values["stoch_d"] >= settings.range_sell_min_stoch - 5.0
        and values["body"] <= -settings.range_min_reversal_body_atr
    )
    if buy == sell:
        result["reason"] = (
            "M5 has no unambiguous inward reversal at a Bollinger extreme "
            "with matching RSI and stochastic evidence."
        )
        return result

    action = "BUY" if buy else "SELL"
    expected = "BULLISH" if buy else "BEARISH"
    target_distance = (
        values["middle"] - values["price"]
        if buy
        else values["price"] - values["middle"]
    )
    if target_distance < values["atr"] * settings.range_min_target_distance_atr:
        result["reason"] = "Range midpoint is too close to provide a useful objective."
        return result

    structure = (m5_analysis or {}).get("market_structure", {}) or {}
    breakout = str(structure.get("breakout_status", "") or "").upper()
    if "BREAKOUT" in breakout:
        result["reason"] = "An active M5 breakout invalidates range-reversion treatment."
        return result
    opposing = "BEARISH" if buy else "BULLISH"
    if any(
        isinstance(event, dict)
        and str(event.get("direction", "")).upper() == opposing
        and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
        for event in structure.get("structure_events", []) or []
    ):
        result["reason"] = "Fresh M5 structure opposes the proposed range reversal."
        return result

    evidence_id = (
        f"M5_RANGE_{expected}_{_time_token(m5_analysis.get('timestamp'))}"
    )
    result.update(
        candidate=True,
        direction=action,
        evidence_id=evidence_id,
        target_price=round(values["middle"], 8),
        invalidation_price=round(
            values["lower"] - values["atr"] * settings.range_invalidation_atr
            if buy
            else values["upper"] + values["atr"] * settings.range_invalidation_atr,
            8,
        ),
        score=round(
            min(
                100.0,
                70.0
                + abs(values["body"]) * 15.0
                + target_distance / values["atr"] * 5.0,
            ),
            1,
        ),
        reason=(
            f"M5 {action} range candidate: ADX {values['adx']:.1f}, "
            f"RSI {values['rsi']:.1f}, stochastic {values['stoch_k']:.1f}, "
            "and an inward completed candle at the outer band."
        ),
    )

    if not all((m15_analysis, h1_analysis, h4_analysis)):
        result["reason"] += " Higher-timeframe range confirmation is pending."
        return result
    if _adx(m15_analysis) > settings.range_confirmation_max_adx or _adx(
        h1_analysis
    ) > settings.range_confirmation_max_adx:
        result["reason"] = (
            "M15 or H1 has active trend strength; the M5 extreme is treated "
            "as a pullback, not a range reversal."
        )
        return result
    if (
        _adx(h4_analysis) >= settings.range_h4_opposition_min_adx
        and _direction(h4_analysis) == opposing
    ):
        result["reason"] = "Strong H4 direction opposes the range reversal."
        return result
    if (
        _direction(m15_analysis) == opposing
        and _direction(h1_analysis) == opposing
    ):
        result["reason"] = "M15 and H1 both oppose the range reversal."
        return result

    result["eligible"] = True
    result["reason"] += " M15/H1/H4 range confirmation passed."
    return result


def annotate_range_reversion(
    m5_analysis: Dict[str, Any],
    m15_analysis: Optional[Mapping[str, Any]] = None,
    h1_analysis: Optional[Mapping[str, Any]] = None,
    h4_analysis: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    setup = classify_range_reversion(
        m5_analysis, m15_analysis, h1_analysis, h4_analysis
    )
    structure = m5_analysis.setdefault("market_structure", {})
    structure["range_reversion"] = setup
    return setup


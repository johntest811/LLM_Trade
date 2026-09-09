"""Pre-model opportunity screening using the same deterministic entry rules."""

from typing import Any
import math

import pandas as pd

from app_config.settings import settings
from core.evidence import build_evidence_ids, permitted_entry_actions
from risk.manager import RiskManager


REVERSAL_WATCH_VERSION = "m5-countertrend-watch-v1"


def classify_reversal_watch(analyses: dict[str, Any], bars: pd.DataFrame) -> dict[str, Any]:
    """Observe a first local reversal without manufacturing live entry evidence.

    Fixed, scale-free research hypothesis: two directional closes, the first
    buffered close through the preceding four-bar boundary, fast EMA/RSI
    agreement, and at least two opposing higher timeframes. ADX is recorded,
    not used as a direction veto. Promotion requires separate validation.
    """
    result = {"candidate": False, "live_eligible": False,
              "strategy_version": REVERSAL_WATCH_VERSION, "direction": "",
              "reason": "No qualified countertrend research setup."}
    if not settings.reversal_watch_enabled:
        return {**result, "reason": "Reversal research watch is disabled."}
    m5 = analyses.get("M5") or {}
    try:
        if bars is None or len(bars) < 6 or bars.attrs.get("is_stale", False):
            return result
        recent = bars.tail(6)
        times = pd.to_datetime(recent["time"], utc=True, errors="raise")
        stamp = pd.to_datetime(m5["timestamp"], utc=True, errors="raise")
        if (times.isna().any() or pd.isna(stamp) or times.iloc[-1] != stamp
                or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all()):
            return result
        closes_at = stamp + pd.Timedelta(minutes=5)
        rows = [{key: float(row[key]) for key in ("open", "high", "low", "close")}
                for row in recent.to_dict("records")]
        if any(not all(math.isfinite(v) and v > 0 for v in row.values())
               or row["low"] > min(row["open"], row["close"])
               or row["high"] < max(row["open"], row["close"]) for row in rows):
            return result
        indicators = m5["indicators"]
        atr, ema9, ema21, rsi = (float(indicators[key]) for key in
                                ("atr_14", "ema_9", "ema_21", "rsi_14"))
        if not all(math.isfinite(v) for v in (atr, ema9, ema21, rsi)) or atr <= 0:
            return result
        sign = 1 if ema9 > ema21 and 52 <= rsi <= 70 else (
            -1 if ema9 < ema21 and 30 <= rsi <= 48 else 0)
        if not sign:
            return result
        opposite = "BEARISH" if sign > 0 else "BULLISH"
        opposing = []
        for tf, minutes in (("M15", 15), ("H1", 60), ("H4", 240)):
            item = analyses[tf]
            closed = pd.to_datetime(item["timestamp"], utc=True, errors="raise") + pd.Timedelta(minutes=minutes)
            if pd.isna(closed) or not pd.Timedelta(0) <= closes_at - closed < pd.Timedelta(minutes=minutes):
                return result  # Never use forming, future, or stale confirmations.
            structure = item.get("market_structure") or {}
            if (structure.get("trend_state_direction") or structure.get("trend")) == opposite:
                opposing.append(tf)
        if len(opposing) < 2:
            return result
        current, previous = rows[-1], rows[-2]
        anchor = max(row["high"] for row in rows[:-2]) if sign > 0 else min(row["low"] for row in rows[:-2])
        body = sign * (current["close"] - current["open"]) / atr
        if not (
            sign * (previous["close"] - rows[-3]["close"]) > 0
            and sign * (current["close"] - previous["close"]) > 0
            and sign * (previous["close"] - anchor) < 0.05 * atr
            and sign * (current["close"] - anchor) >= 0.05 * atr
            and 0.10 <= body <= 1.0
            and (current["high"] - current["low"]) / atr <= 1.5
            and abs(current["open"] - previous["close"]) / atr <= 0.25
            and sign * (current["close"] - ema9) >= 0
        ):
            return result
        stop = min(row["low"] for row in rows[-2:]) - 0.15 * atr if sign > 0 else max(row["high"] for row in rows[-2:]) + 0.15 * atr
        risk = sign * (current["close"] - stop)
        target = current["close"] + sign * 2.0 * risk
        if not 0.25 <= risk / atr <= 2.5 or stop <= 0 or target <= 0:
            return result
        action = "BUY" if sign > 0 else "SELL"
        return {**result, "candidate": True, "direction": action,
                "candle_time": stamp.isoformat(), "signal_time_utc": closes_at.isoformat(),
                "reference_price": current["close"], "stop_loss": stop,
                "take_profit": target,
                "atr": atr, "opposing_timeframes": opposing,
                "reason": f"{action} local reversal against {', '.join(opposing)}; RESEARCH ONLY, not an approved live entry."}
    except (KeyError, TypeError, ValueError, OverflowError):
        return result


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
        "research_watch": (analyses.get("M5") or {}).get("market_structure", {}).get("reversal_watch", {}),
        "live_reversal": (analyses.get("M5") or {}).get("market_structure", {}).get("live_reversal", {}),
        "missing_entry_reasons": {
            action: "No completed-M5 structure, breakout, verified retest, eligible range or qualified live reversal trigger."
            for action in ("BUY", "SELL") if action not in actions
        },
        "model_eligible": bool(capital_fit.get("capital_fit") and capital_fit.get("broker_open", True) and viable),
    }

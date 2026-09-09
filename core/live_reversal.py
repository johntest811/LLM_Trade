"""Opt-in, bounded live qualification of the separate reversal research watch.

This is an experimental entry contract, not evidence of profitable expectancy.
No broker calls, model output, wall-clock fitting, or order submission live here.
"""

import math
from datetime import datetime, timedelta, timezone

from app_config.settings import settings

LIVE_REVERSAL_VERSION = "m5-m15-local-reversal-v1"
LIVE_REVERSAL_MIN_CONFIDENCE = 0.85
LIVE_REVERSAL_MAX_RISK_PERCENT = 2.0
LIVE_REVERSAL_MAX_AGE_SECONDS = 120


def _stamp(value, *, analysis_utc=False):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        if not analysis_utc:
            raise ValueError("Timezone is required")
        # MarketAnalysisEngine's legacy timestamp format omits the suffix;
        # MT5DataReader has already converted those candle opens to UTC.
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def qualify_live_reversal(analyses, *, config=None, now=None):
    """Re-derive authority from deterministic facts, never an eligible flag.

    ``now`` is supplied at final risk validation to expire delayed model replies.
    Discovery uses bar-relative timestamps so identical closed bars give the
    same evidence catalog. The research-only payload stays untouched.
    """
    config = settings if config is None else config
    result = {"eligible": False, "strategy_version": LIVE_REVERSAL_VERSION,
              "direction": "", "reason": "Live reversal entries are disabled."}
    if not getattr(config, "live_reversal_enabled", False) or not config.reversal_watch_enabled:
        return result
    result["reason"] = "Requires a fresh research trigger, M5/M15 confirmation and rising momentum."
    try:
        m5 = analyses["M5"]
        watch = m5["market_structure"]["reversal_watch"]
        action = watch["direction"]
        if (watch.get("candidate") is not True or watch.get("live_eligible") is not False
                or watch.get("strategy_version") != "m5-countertrend-watch-v1"
                or action not in {"BUY", "SELL"}):
            return result
        sign = 1 if action == "BUY" else -1
        expected, opposite = ("BULLISH", "BEARISH") if sign == 1 else ("BEARISH", "BULLISH")
        stamp = _stamp(m5["timestamp"], analysis_utc=True)
        closed = stamp + timedelta(minutes=5)
        if (stamp.minute % 5 or stamp.second or stamp.microsecond
                or _stamp(watch["candle_time"]) != stamp
                or _stamp(watch["signal_time_utc"]) != closed):
            return result
        if now is not None and not 0 <= (_stamp(now) - closed).total_seconds() <= LIVE_REVERSAL_MAX_AGE_SECONDS:
            return {**result, "reason": "Live reversal signal is future-dated or more than 120 seconds old."}
        for tf, minutes in (("M5", 5), ("M15", 15), ("H1", 60), ("H4", 240)):
            analysis = analyses[tf]
            bar = _stamp(analysis["timestamp"], analysis_utc=True)
            age = (closed - bar - timedelta(minutes=minutes)).total_seconds()
            structure = analysis["market_structure"]
            direction = structure.get("trend_state_direction") or structure.get("trend")
            # H4 candle phase follows the broker server, including DST. It
            # need not align to UTC midnight; freshness/closedness is what matters.
            if (bar.second or bar.microsecond or bar.minute % min(minutes, 60)
                    or not 0 <= age < minutes * 60
                    or direction != (expected if tf in {"M5", "M15"} else opposite)):
                return result
        indicators = m5["indicators"]
        atr, price, ema9, ema21, rsi, adx, delta = (
            float(indicators[key]) for key in
            ("atr_14", "current_price", "ema_9", "ema_21", "rsi_14", "adx_14", "adx_delta"))
        m15_adx = float(analyses["M15"]["indicators"]["adx_14"])
        reference, stop, target, watch_atr = (float(watch[key]) for key in
                                             ("reference_price", "stop_loss", "take_profit", "atr"))
        if not all(math.isfinite(value) for value in
                   (atr, price, ema9, ema21, rsi, adx, delta, m15_adx, reference, stop, target, watch_atr)):
            return result
        risk = sign * (reference - stop)
        if (min(atr, price, reference, stop, target, ema9, ema21) <= 0
                or not math.isclose(atr, watch_atr, rel_tol=1e-9)
                or not math.isclose(price, reference, rel_tol=1e-9)
                or sign * (ema9 - ema21) <= 0 or sign * (price - ema9) < 0
                or not ((52 <= rsi <= 70) if sign == 1 else (30 <= rsi <= 48))
                or adx < max(25.0, config.entry_min_adx)
                or m15_adx < config.confirmation_min_adx or delta < 0
                or not 0.25 <= risk / atr <= 2.5
                or not math.isclose(sign * (target - reference), 2.0 * risk, rel_tol=1e-9)):
            return result
        return {**result, "eligible": True, "direction": action,
                "candle_time": stamp.isoformat(), "signal_time_utc": closed.isoformat(),
                "evidence_id": f"M5_LOCAL_REVERSAL_{expected}_{stamp:%Y%m%dT%H%M}",
                "invalidation_price": stop, "target_price": target,
                "min_confidence": LIVE_REVERSAL_MIN_CONFIDENCE,
                "max_risk_percent": LIVE_REVERSAL_MAX_RISK_PERCENT,
                "reason": "Experimental live candidate: M5/M15 confirmed against H1/H4; LLM and final risk approval still required."}
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return result


def live_reversal_plan(analysis, action, *, config=None):
    """Read the engine-owned plan annotation; final risk rechecks all frames."""
    config = settings if config is None else config
    setup = ((analysis or {}).get("market_structure") or {}).get("live_reversal") or {}
    if (getattr(config, "live_reversal_enabled", False)
            and isinstance(setup, dict) and setup.get("eligible") is True
            and setup.get("strategy_version") == LIVE_REVERSAL_VERSION
            and setup.get("direction") == action):
        try:
            if _stamp(setup["candle_time"]) == _stamp(analysis["timestamp"], analysis_utc=True):
                return setup
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
            pass
    return {}


def live_reversal_risk_percent(config):
    return min(float(config.risk_percent), LIVE_REVERSAL_MAX_RISK_PERCENT)

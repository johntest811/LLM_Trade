"""Scale-independent confirmation for fresh, unretested continuation entries."""

import math


def continuation_entry_check(action, analysis, *, min_clearance_atr=0.15):
    """Do not treat two names for one break as independent momentum evidence.

    Called only for ordinary trend continuation. Independently verified retests
    and separately qualified reversal/range modes have their own timing gates.
    This is a conservative rule, not a calibrated probability of success.
    """
    structure = (analysis or {}).get("market_structure") or {}
    expected = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(action)
    if expected is None:
        return False, "Invalid continuation direction."
    retest = structure.get("retest_continuation") or {}
    if isinstance(retest, dict) and retest.get("direction") == expected:
        return True, "Verified retest uses the existing retest validation."
    try:
        sign = 1 if action == "BUY" else -1
        indicators = analysis["indicators"]
        price, atr, fast, slow, rsi = (float(indicators[key]) for key in
                                      ("current_price", "atr_14", "ema_9", "ema_21", "rsi_14"))
        macd = float(indicators["macd"]["diff"])
        if (not all(math.isfinite(v) for v in (price, atr, fast, slow, rsi, macd))
                or min(price, atr, fast, slow) <= 0):
            return False, "Valid finite continuation momentum and ATR are required."
        if sign * macd <= 0 or sign * (fast - slow) <= 0 or not (50 <= rsi <= 68 if sign > 0 else 32 <= rsi <= 50):
            return False, "Unretested continuation requires EMA, RSI and MACD to confirm together; wait for confirmation or a verified retest."
        levels = []
        for event in structure.get("structure_events") or []:
            if isinstance(event, dict) and event.get("type") == "BOS" and event.get("direction") == expected:
                level = float(event["level"])
                if not math.isfinite(level) or level <= 0:
                    return False, "Invalid broken-structure level."
                levels.append(level)
        if not levels:
            level = float(structure["resistance" if sign > 0 else "support"])
            if not math.isfinite(level) or level <= 0:
                return False, "Valid broken-structure level is required."
            levels.append(level)
        # The most recently reported break must be clear on the completed
        # candle, not merely on a later ask quote or by a fraction of a tick.
        clearance = sign * (price - levels[-1]) / atr
        if clearance + 1e-9 < min_clearance_atr:
            return False, f"Completed close clears structure by {clearance:.2f} ATR; at least {min_clearance_atr:.2f} ATR or a verified retest is required."
        return True, "Continuation momentum and completed-close clearance confirmed."
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, "Completed-candle continuation evidence is incomplete."

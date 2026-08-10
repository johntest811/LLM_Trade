"""Deterministic counterfactual evaluation for rejected entry signals.

This module is deliberately isolated from execution.  It evaluates a stored
entry/stop/target against completed broker M1 bars and returns gross R outcome
telemetry that can be audited before any rule is promoted to live trading.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

import math
import pandas as pd


@dataclass(frozen=True)
class ShadowResolution:
    status: str
    resolved_at_utc: str
    exit_price: float
    outcome_r: Optional[float]
    mfe_r: float
    mae_r: float


def _utc(value: Any) -> Optional[datetime]:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def evaluate_shadow_candidate(
    candidate: Dict[str, Any],
    bars: Iterable[Dict[str, Any]] | pd.DataFrame,
    *,
    now_utc: Optional[datetime] = None,
) -> Optional[ShadowResolution]:
    """Resolve one counterfactual using chronological completed M1 OHLC bars.

    If a single bar touches both stop and target the path is unknowable from
    OHLC data, so it is marked AMBIGUOUS and excluded from expectancy.
    """
    action = str(candidate.get("action", "")).upper()
    if action not in {"BUY", "SELL"}:
        return None
    try:
        entry = float(candidate["entry"])
        stop = float(candidate["stop_loss"])
        target = float(candidate["take_profit"])
        horizon_minutes = float(candidate.get("horizon_minutes", 60.0) or 60.0)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    risk = abs(entry - stop)
    if not all(math.isfinite(value) for value in (entry, stop, target, risk)) or risk <= 0:
        return None
    created = _utc(candidate.get("created_at_utc"))
    if created is None:
        return None
    deadline = created + timedelta(minutes=max(1.0, horizon_minutes))
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    records = bars.to_dict("records") if isinstance(bars, pd.DataFrame) else list(bars)
    usable = []
    for row in records:
        stamp = _utc(row.get("time"))
        if stamp is None or stamp < created or stamp > deadline:
            continue
        try:
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if all(math.isfinite(value) for value in (high, low, close)):
            usable.append((stamp, high, low, close))
    usable.sort(key=lambda row: row[0])

    mfe_r = 0.0
    mae_r = 0.0
    for stamp, high, low, close in usable:
        if action == "BUY":
            favorable = (high - entry) / risk
            adverse = (entry - low) / risk
            target_hit = high >= target
            stop_hit = low <= stop
        else:
            favorable = (entry - low) / risk
            adverse = (high - entry) / risk
            target_hit = low <= target
            stop_hit = high >= stop
        mfe_r = max(mfe_r, favorable)
        mae_r = max(mae_r, adverse)
        if target_hit and stop_hit:
            return ShadowResolution(
                "AMBIGUOUS", stamp.isoformat(), close, None, mfe_r, mae_r
            )
        if target_hit:
            reward_r = abs(target - entry) / risk
            return ShadowResolution(
                "TP", stamp.isoformat(), target, reward_r, mfe_r, mae_r
            )
        if stop_hit:
            return ShadowResolution(
                "SL", stamp.isoformat(), stop, -1.0, mfe_r, mae_r
            )

    if now < deadline or not usable:
        return None
    stamp, _, _, close = usable[-1]
    signed_move = (close - entry) if action == "BUY" else (entry - close)
    outcome_r = signed_move / risk
    if outcome_r > 0.02:
        status = "TIMEOUT_WIN"
    elif outcome_r < -0.02:
        status = "TIMEOUT_LOSS"
    else:
        status = "TIMEOUT_FLAT"
    return ShadowResolution(
        status, stamp.isoformat(), close, outcome_r, mfe_r, mae_r
    )

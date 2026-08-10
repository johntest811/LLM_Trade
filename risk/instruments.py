"""Broker-instrument helpers shared by risk and execution.

FX pips are not a meaningful cost unit for every CFD.  Crypto spreads are
therefore measured in basis points while FX/metals retain conventional pips.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from app_config.settings import settings


_CRYPTO_CODES = {
    "ADA", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "EOS", "ETH",
    "LINK", "LTC", "MATIC", "NEAR", "SOL", "TRX", "UNI", "XLM", "XRP",
}

_TRADE_MODE_DISABLED = 0
_TRADE_MODE_LONG_ONLY = 1
_TRADE_MODE_SHORT_ONLY = 2
_TRADE_MODE_CLOSE_ONLY = 3
_TRADE_MODE_FULL = 4


def is_crypto_symbol(symbol: str, info: Any = None) -> bool:
    path = str(getattr(info, "path", "") or "").lower()
    if "crypto" in path:
        return True
    upper = str(symbol).upper()
    return any(code in upper for code in _CRYPTO_CODES)


def validate_symbol_trade_mode(info: Any, action: str) -> Tuple[bool, str]:
    """Check whether the broker currently permits a new order direction."""
    side = str(action or "").strip().upper()
    if side not in {"BUY", "SELL"}:
        return False, "Order direction must be BUY or SELL"
    raw_mode = getattr(info, "trade_mode", _TRADE_MODE_FULL)
    try:
        mode = int(_TRADE_MODE_FULL if raw_mode is None else raw_mode)
    except (TypeError, ValueError, OverflowError):
        return False, "Broker returned an invalid symbol trade mode"
    if mode == _TRADE_MODE_DISABLED:
        return False, "Broker has disabled trading for this market"
    if mode == _TRADE_MODE_CLOSE_ONLY:
        return False, "Broker market is close-only; new positions are unavailable"
    if mode == _TRADE_MODE_LONG_ONLY and side != "BUY":
        return False, "Broker market currently permits BUY orders only"
    if mode == _TRADE_MODE_SHORT_ONLY and side != "SELL":
        return False, "Broker market currently permits SELL orders only"
    if mode not in {
        _TRADE_MODE_LONG_ONLY,
        _TRADE_MODE_SHORT_ONLY,
        _TRADE_MODE_FULL,
    }:
        return False, f"Broker returned unsupported symbol trade mode {mode}"
    return True, ""


def pip_size(info: Any) -> float:
    """Return one conventional pip for an MT5 symbol descriptor."""
    point = float(getattr(info, "point", 0.0) or 0.0)
    digits = int(getattr(info, "digits", 0) or 0)
    return point * 10.0 if digits in (3, 5) else point


def spread_metrics(symbol: str, info: Any, tick: Any) -> Dict[str, float | str]:
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    spread_abs = max(0.0, ask - bid)
    midpoint = (ask + bid) / 2.0 if ask > 0 and bid > 0 else 0.0
    bps = spread_abs / midpoint * 10_000.0 if midpoint else 0.0
    pip = pip_size(info)
    pips = spread_abs / pip if pip else 0.0
    crypto = is_crypto_symbol(symbol, info)
    return {
        "absolute": spread_abs,
        "pips": pips,
        "bps": bps,
        "value": bps if crypto else pips,
        "unit": "bps" if crypto else "pips",
        "asset_class": "CRYPTO" if crypto else "FX/CFD",
    }


def downside_risk_usd(calculated_profit: Any) -> float:
    """Convert an MT5 stop P/L estimate into remaining downside only.

    A stop which has already been moved beyond break-even has positive
    ``order_calc_profit`` and therefore contributes no remaining loss risk.
    """
    if calculated_profit is None:
        raise ValueError("MT5 profit estimate is unavailable")
    try:
        profit = float(calculated_profit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("MT5 profit estimate is invalid") from exc
    if not math.isfinite(profit):
        raise ValueError("MT5 profit estimate is not finite")
    return max(0.0, -profit)


def validate_spread(
    symbol: str,
    info: Any,
    tick: Any,
    *,
    entry: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> Tuple[bool, str, Dict[str, float | str]]:
    """Validate spread in an asset-appropriate unit and relative to the stop."""
    metrics = spread_metrics(symbol, info, tick)
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= 0.0:
        return False, "Bid and ask quotes must be finite positive prices", metrics
    if ask < bid:
        return False, "Crossed quote: ask is below bid", metrics
    if is_crypto_symbol(symbol, info):
        if float(metrics["bps"]) > settings.max_crypto_spread_bps:
            return (
                False,
                f"Crypto spread {float(metrics['bps']):.1f} bps exceeds "
                f"{settings.max_crypto_spread_bps:.1f} bps",
                metrics,
            )
    elif float(metrics["pips"]) > settings.max_spread_pips:
        return (
            False,
            f"Spread {float(metrics['pips']):.1f} pips exceeds "
            f"{settings.max_spread_pips:.1f} pips",
            metrics,
        )

    if entry is not None and stop_loss is not None:
        stop_distance = abs(float(entry) - float(stop_loss))
        if stop_distance <= 0:
            return False, "Stop distance is zero", metrics
        ratio = float(metrics["absolute"]) / stop_distance * 100.0
        metrics["spread_to_stop_pct"] = ratio
        ratio_tolerance = max(1e-9, abs(settings.max_spread_to_stop_pct) * 1e-9)
        if ratio > settings.max_spread_to_stop_pct + ratio_tolerance:
            return (
                False,
                f"Spread consumes {ratio:.1f}% of stop distance; maximum is "
                f"{settings.max_spread_to_stop_pct:.1f}%",
                metrics,
            )
    return True, "", metrics

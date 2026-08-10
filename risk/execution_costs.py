"""Account-currency execution reserves shared by planning and execution.

The live BID/ASK entry already includes spread.  This module adds the two
costs that a stop-distance calculation otherwise misses: adverse fill within
the order's configured MT5 deviation and explicitly configured commissions or
fees.  All monetary values are in the MT5 account currency as returned by
``order_calc_profit`` (USD for the current account).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

from mt5.safe_api import mt5

from app_config.settings import settings
from risk.instruments import is_crypto_symbol


@dataclass(frozen=True)
class ExecutionRiskEstimate:
    stop_risk_usd: float
    slippage_reserve_usd: float
    configured_cost_usd: float
    total_risk_usd: float
    worst_entry: float
    deviation_points: int


def _configured_per_lot_cost(symbol: str, info: Any) -> float:
    if is_crypto_symbol(symbol, info):
        value = float(getattr(settings, "crypto_round_turn_cost_usd_per_lot", 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid crypto execution-cost configuration")
        return value

    path = str(getattr(info, "path", "") or "").lower()
    calc_mode = getattr(info, "trade_calc_mode", None)
    forex_modes = {
        getattr(mt5, "SYMBOL_CALC_MODE_FOREX", object()),
        getattr(mt5, "SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE", object()),
    }
    if "forex" in path or calc_mode in forex_modes:
        value = float(getattr(settings, "fx_round_turn_cost_usd_per_lot", 0.0))
    else:
        value = float(getattr(settings, "cfd_round_turn_cost_usd_per_lot", 0.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Invalid execution-cost configuration")
    return value


def configured_execution_cost_usd(symbol: str, info: Any, lot: float) -> float:
    """Return the configured round-turn fee reserve for this volume."""
    fixed = float(getattr(settings, "fixed_execution_cost_usd", 0.0))
    lot = float(lot)
    if not math.isfinite(fixed) or fixed < 0 or not math.isfinite(lot) or lot < 0:
        raise ValueError("Invalid fixed execution-cost or volume input")
    total = fixed + _configured_per_lot_cost(symbol, info) * lot
    if not math.isfinite(total):
        raise ValueError("Execution-cost estimate is non-finite")
    return total


def estimate_execution_risk(
    symbol: str,
    action: str,
    lot: float,
    entry: float,
    stop_loss: float,
    info: Any,
    *,
    deviation_points: Optional[int] = None,
) -> Optional[ExecutionRiskEstimate]:
    """Estimate worst fill-to-stop loss plus configured round-turn costs.

    ``None`` is returned when MT5 cannot express the loss in account currency;
    callers must fail closed in that case.
    """
    action = str(action).upper()
    if action not in {"BUY", "SELL"}:
        return None
    lot = float(lot)
    entry = float(entry)
    stop_loss = float(stop_loss)
    if (
        not all(math.isfinite(value) for value in (lot, entry, stop_loss))
        or lot <= 0
        or entry <= 0
        or stop_loss <= 0
    ):
        return None

    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    base_profit = mt5.order_calc_profit(order_type, symbol, lot, entry, stop_loss)
    if base_profit is None or not math.isfinite(float(base_profit)):
        return None
    stop_risk = max(0.0, -float(base_profit))
    if stop_risk <= 0:
        return None

    try:
        deviation = max(
            0,
            int(
                getattr(settings, "max_order_deviation_points", 20)
                if deviation_points is None
                else deviation_points
            ),
        )
    except (TypeError, ValueError, OverflowError):
        return None
    point = float(getattr(info, "point", 0.0) or 0.0)
    if not math.isfinite(point) or point < 0:
        return None
    adverse_move = deviation * point
    worst_entry = entry + adverse_move if action == "BUY" else entry - adverse_move
    if not math.isfinite(worst_entry) or worst_entry <= 0:
        return None
    worst_profit = mt5.order_calc_profit(order_type, symbol, lot, worst_entry, stop_loss)
    if worst_profit is None or not math.isfinite(float(worst_profit)):
        return None
    worst_stop_risk = max(0.0, -float(worst_profit))
    if worst_stop_risk <= 0:
        return None

    slippage = max(0.0, worst_stop_risk - stop_risk)
    try:
        configured_cost = configured_execution_cost_usd(symbol, info, lot)
    except (TypeError, ValueError, OverflowError):
        return None
    total_risk = worst_stop_risk + configured_cost
    if not math.isfinite(total_risk):
        return None
    return ExecutionRiskEstimate(
        stop_risk_usd=stop_risk,
        slippage_reserve_usd=slippage,
        configured_cost_usd=configured_cost,
        total_risk_usd=total_risk,
        worst_entry=worst_entry,
        deviation_points=deviation,
    )

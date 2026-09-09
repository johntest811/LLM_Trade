"""
execution/executor.py — Production MT5 Trade Execution Engine

Supports: BUY, SELL, Modify SL/TP, Partial Close, Full Close,
          Break-even, Trailing Stop.

Validates: symbol, spread, lot size, margin before every action.
Retries:   transient MT5 error codes automatically (up to MAX_RETRIES).
Dedup:     tracks in-flight requests to prevent duplicate orders.
Assets:    Forex, Crypto, Metals, Indices — filling mode resolved per symbol.
"""
import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from mt5.safe_api import mt5, serialized_mt5

from app_config.settings import settings
from mt5.timebase import broker_tick_age_seconds
from risk.execution_costs import (
    configured_execution_cost_usd,
    estimate_execution_risk,
)
from risk.instruments import (
    pip_size,
    validate_spread,
    validate_symbol_trade_mode,
)
logger = logging.getLogger("TradingSystem.Executor")

# MT5 retcodes that are transient and safe to retry
_RETRYABLE_CODES: Set[int] = {
    mt5.TRADE_RETCODE_REQUOTE,
    mt5.TRADE_RETCODE_PRICE_CHANGED,
    mt5.TRADE_RETCODE_PRICE_OFF,
}

MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds between retries
MAGIC_NUMBER = settings.strategy_magic
ORDER_COMMENT = settings.order_comment


@dataclass
class ExecutionResult:
    success: bool
    ticket: Optional[int]
    price: Optional[float]
    volume: Optional[float]
    error: Optional[str]
    retcode: Optional[int] = None
    partial: bool = False
    verified: bool = False
    state_changed: bool = False
    requested_volume: Optional[float] = None
    remaining_volume: Optional[float] = None


class MT5OrderExecutor:
    """
    Async execution engine for all MT5 trade operations.

    All blocking mt5.* calls are dispatched to a thread-pool via
    asyncio.to_thread() so they never stall the event loop.
    """

    def __init__(self, connection_manager) -> None:
        self._conn = connection_manager
        self._inflight: Set[str] = set()   # dedup lock keys
        self._workers: Set[asyncio.Task] = set()
        self._accepting_work = True

    def resume(self) -> None:
        self._accepting_work = True

    def begin_shutdown(self) -> None:
        """Reject new broker mutations while existing native calls drain."""
        self._accepting_work = False

    async def wait_until_idle(self, timeout: float = 15.0) -> bool:
        pending = list(self._workers)
        if not pending:
            return True
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout)
            return True
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for %s in-flight MT5 worker(s).", len(pending))
            return False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def open_trade(
        self,
        symbol: str,
        action: str,                      # "BUY" | "SELL"
        lot_size: float,
        sl_price: Optional[float] = None, # absolute price levels
        tp_price: Optional[float] = None,
        comment: str = ORDER_COMMENT,
        expected_account: Optional[Dict[str, Any]] = None,
        max_risk_usd: Optional[float] = None,
        operator_override: bool = False,
    ) -> ExecutionResult:
        """Open a new BUY or SELL market order with absolute SL/TP prices."""
        if not self._accepting_work:
            return ExecutionResult(False, None, None, None, "Executor is shutting down")
        action = action.upper()
        if action not in ("BUY", "SELL"):
            return ExecutionResult(False, None, None, None, f"Invalid action '{action}'")

        # Dedup guard — same symbol+direction cannot fire twice simultaneously
        lock_key = f"{symbol}:{action}"
        if lock_key in self._inflight:
            return ExecutionResult(False, None, None, None,
                                   f"Duplicate request in-flight: {lock_key}")
        self._inflight.add(lock_key)

        try:
            return await self._with_retry(self._submit_open, symbol, action,
                                          lot_size, sl_price, tp_price, comment,
                                          expected_account, max_risk_usd,
                                          operator_override)
        finally:
            self._inflight.discard(lock_key)

    async def close_position(
        self,
        ticket: int,
        comment: str = f"{ORDER_COMMENT}-Close",
        expected_account: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Fully close an open position by ticket number."""
        if not self._accepting_work:
            return ExecutionResult(False, ticket, None, None, "Executor is shutting down")
        lock_key = f"CLOSE:{ticket}"
        if lock_key in self._inflight:
            return ExecutionResult(False, None, None, None,
                                   f"Close already in-flight for ticket {ticket}")
        self._inflight.add(lock_key)
        try:
            return await self._with_retry(self._submit_close, ticket, comment, expected_account)
        finally:
            self._inflight.discard(lock_key)

    async def partial_close(
        self,
        ticket: int,
        close_volume: float,
        comment: str = f"{ORDER_COMMENT}-Partial",
        expected_account: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Close a partial volume of an existing position."""
        if not self._accepting_work:
            return ExecutionResult(False, ticket, None, None, "Executor is shutting down")
        lock_key = f"CLOSE:{ticket}"
        if lock_key in self._inflight:
            return ExecutionResult(False, ticket, None, None,
                                   f"Close already in-flight for ticket {ticket}")
        self._inflight.add(lock_key)
        try:
            return await self._with_retry(
                self._submit_partial_close, ticket, close_volume, comment, expected_account
            )
        finally:
            self._inflight.discard(lock_key)

    async def modify_sl_tp(
        self,
        ticket: int,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        expected_account: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Modify the Stop Loss and/or Take Profit of an open position."""
        if not self._accepting_work:
            return ExecutionResult(False, ticket, None, None, "Executor is shutting down")
        lock_key = f"MODIFY:{ticket}"
        if lock_key in self._inflight:
            return ExecutionResult(False, ticket, None, None,
                                   f"Modification already in-flight for ticket {ticket}")
        self._inflight.add(lock_key)
        try:
            return await self._with_retry(
                self._submit_modify, ticket, sl_price, tp_price, expected_account
            )
        finally:
            self._inflight.discard(lock_key)

    async def move_to_breakeven(
        self,
        ticket: int,
        buffer_pips: float = 1.0,
        expected_account: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Move Stop Loss to entry (break-even) + buffer pips.
        Only moves SL if trade is currently in profit beyond the buffer.
        """
        def _calc():
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return None, None, f"Ticket {ticket} not found"
            pos = positions[0]
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info is None:
                return None, None, "Cannot query symbol info"

            buffer = buffer_pips * pip_size(symbol_info)
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return None, None, "Cannot fetch live quote"
            current_price = (
                tick.bid
                if pos.type == mt5.POSITION_TYPE_BUY
                else tick.ask
            )
            if pos.type == mt5.POSITION_TYPE_BUY:
                be_sl = round(pos.price_open + buffer, symbol_info.digits)
                if current_price < be_sl:
                    return None, None, "Price not yet above break-even level"
                if pos.sl > 0 and be_sl <= pos.sl:
                    return None, None, "Break-even would worsen the current stop"
            else:
                be_sl = round(pos.price_open - buffer, symbol_info.digits)
                if current_price > be_sl:
                    return None, None, "Price not yet below break-even level"
                if pos.sl > 0 and be_sl >= pos.sl:
                    return None, None, "Break-even would worsen the current stop"

            return ticket, be_sl, None

        ticket_val, be_sl, err = await asyncio.to_thread(_calc)
        if err:
            logger.warning(f"Break-even skipped for ticket {ticket}: {err}")
            return ExecutionResult(False, ticket, None, None, err)

        result = await self.modify_sl_tp(
            ticket_val, sl_price=be_sl, expected_account=expected_account
        )
        if result.success:
            logger.info(f"Break-even set for ticket {ticket} at SL={be_sl}")
        return result

    async def lock_minimum_net_profit(
        self,
        ticket: int,
        floor_usd: float,
        expected_account: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Move SL to a broker price intended to retain a small net profit.

        The target price is derived with MT5 ``order_calc_profit`` in account
        currency. Configured round-turn costs are added before solving for the
        price, so this is more accurate across JPY pairs and CFDs than a fixed
        pip buffer. The method can only improve an existing stop.
        """
        try:
            requested_floor = float(floor_usd)
        except (TypeError, ValueError, OverflowError):
            requested_floor = float("nan")
        if not math.isfinite(requested_floor) or requested_floor < 0:
            return ExecutionResult(False, ticket, None, None, "Invalid net-profit floor")

        def _calc():
            positions = mt5.positions_get(ticket=ticket)
            if positions is None:
                return None, None, f"Position query failed: {mt5.last_error()}"
            if not positions:
                return None, None, f"Ticket {ticket} not found"
            pos = positions[0]
            info = mt5.symbol_info(pos.symbol)
            tick = mt5.symbol_info_tick(pos.symbol)
            if info is None or tick is None:
                return None, None, "Cannot fetch symbol metadata and live quote"

            one_pip = pip_size(info)
            if one_pip <= 0:
                return None, None, "Broker pip size is invalid"
            try:
                execution_cost = configured_execution_cost_usd(
                    pos.symbol, info, float(pos.volume)
                )
            except (TypeError, ValueError, OverflowError) as exc:
                return None, None, f"Execution-cost estimate failed: {exc}"

            order_type = (
                mt5.ORDER_TYPE_BUY
                if pos.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_SELL
            )
            probe_price = (
                float(pos.price_open) + one_pip
                if pos.type == mt5.POSITION_TYPE_BUY
                else float(pos.price_open) - one_pip
            )
            profit_per_pip = mt5.order_calc_profit(
                order_type,
                pos.symbol,
                float(pos.volume),
                float(pos.price_open),
                probe_price,
            )
            if profit_per_pip is None or not math.isfinite(float(profit_per_pip)):
                return None, None, "MT5 could not calculate profit-lock price"
            profit_per_pip = float(profit_per_pip)
            if profit_per_pip <= 0:
                return None, None, "Broker profit-per-pip result is invalid"

            swap = float(getattr(pos, "swap", 0.0) or 0.0)
            if not math.isfinite(swap) or not math.isfinite(execution_cost):
                return None, None, "Invalid profit-lock costs or swap"
            target_gross = execution_cost + requested_floor - swap
            distance = target_gross / profit_per_pip * one_pip
            lock_sl = (
                float(pos.price_open) + distance
                if pos.type == mt5.POSITION_TYPE_BUY
                else float(pos.price_open) - distance
            )
            direction = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            rounding = "up" if direction > 0 else "down"
            step = float(getattr(info, "trade_tick_size", 0.0) or info.point)
            if not math.isfinite(step) or step <= 0:
                return None, None, "Invalid profit-lock tick size"
            # Round toward greater protection, then verify with the broker's
            # actual P/L calculator. A one-pip linear approximation alone can
            # understate a floor on coarse ticks or price-dependent conversion.
            lock_sl = self._normalize_price(lock_sl, info, mode=rounding)
            for _ in range(3):
                projected = mt5.order_calc_profit(
                    order_type, pos.symbol, float(pos.volume),
                    float(pos.price_open), lock_sl,
                )
                if projected is None or not math.isfinite(float(projected)):
                    return None, None, "Cannot verify net profit at profit-lock SL"
                deficit = target_gross - float(projected)
                if deficit <= 1e-8:
                    break
                correction = max(step, deficit / profit_per_pip * one_pip)
                lock_sl = self._normalize_price(
                    lock_sl + direction * correction, info, mode=rounding
                )
            else:
                return None, None, "Broker-calculated SL profit is below requested floor"

            if pos.type == mt5.POSITION_TYPE_BUY:
                if float(pos.sl or 0.0) > 0 and lock_sl <= float(pos.sl):
                    return None, None, "Profit lock would worsen the current stop"
                current_exit = float(tick.bid)
                if lock_sl >= current_exit:
                    return None, None, "Price has not advanced far enough for profit lock"
            else:
                if float(pos.sl or 0.0) > 0 and lock_sl >= float(pos.sl):
                    return None, None, "Profit lock would worsen the current stop"
                current_exit = float(tick.ask)
                if lock_sl <= current_exit:
                    return None, None, "Price has not advanced far enough for profit lock"

            minimum = max(
                float(getattr(info, "trade_stops_level", 0.0) or 0.0),
                float(getattr(info, "trade_freeze_level", 0.0) or 0.0),
            ) * float(info.point)
            if minimum > 0 and abs(current_exit - lock_sl) < minimum:
                return None, None, (
                    f"Profit-lock SL is inside broker stop/freeze distance "
                    f"({minimum:g})"
                )
            return int(pos.ticket), lock_sl, None

        ticket_value, lock_sl, error = await asyncio.to_thread(_calc)
        if error:
            logger.debug("Profit lock skipped for ticket %s: %s", ticket, error)
            return ExecutionResult(False, ticket, None, None, error)

        result = await self.modify_sl_tp(
            ticket_value,
            sl_price=lock_sl,
            expected_account=expected_account,
        )
        if result.success:
            logger.info(
                "Net-profit floor set for ticket %s at SL=%s (floor=$%.2f)",
                ticket,
                lock_sl,
                requested_floor,
            )
        return result

    async def apply_trailing_stop(
        self,
        ticket: int,
        trail_pips: float,
        expected_account: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Moves SL to trail_pips behind current price if the new SL is
        better than the existing one (locks in more profit).
        """
        def _calc():
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return None, None, None, f"Ticket {ticket} not found"
            pos = positions[0]
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info is None:
                return None, None, None, "Cannot query symbol info"

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return None, None, None, "Cannot fetch tick"

            trail = trail_pips * pip_size(symbol_info)
            digits = symbol_info.digits

            if pos.type == mt5.POSITION_TYPE_BUY:
                new_sl = round(tick.bid - trail, digits)
                # Only move SL up, never down
                if new_sl <= pos.sl:
                    return None, None, None, "Trail SL not better than current"
            else:
                new_sl = round(tick.ask + trail, digits)
                # Only move SL down, never up
                if pos.sl > 0 and new_sl >= pos.sl:
                    return None, None, None, "Trail SL not better than current"

            return ticket, new_sl, pos.tp or None, None

        t, new_sl, tp, err = await asyncio.to_thread(_calc)
        if err:
            logger.debug(f"Trailing stop skip for ticket {ticket}: {err}")
            return ExecutionResult(False, ticket, None, None, err)

        result = await self.modify_sl_tp(
            t, sl_price=new_sl, tp_price=tp, expected_account=expected_account
        )
        if result.success:
            logger.info(f"Trailing stop updated for ticket {ticket}: SL={new_sl}")
        return result

    # ------------------------------------------------------------------ #
    # Validation helpers                                                   #
    # ------------------------------------------------------------------ #

    def _validate_symbol(self, symbol: str) -> Optional[str]:
        """Ensure symbol exists and is selectable. Returns error str or None."""
        if not mt5.symbol_select(symbol, True):
            return f"Cannot select symbol '{symbol}': {mt5.last_error()}"
        info = mt5.symbol_info(symbol)
        if info is None:
            return f"Symbol '{symbol}' not found in market watch"
        if not info.visible:
            return f"Symbol '{symbol}' is hidden in market watch"
        return None

    def _validate_account_identity(
        self, expected_account: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Fail closed unless the active terminal/account is verified."""
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        if account is None or terminal is None:
            return "Cannot verify active MT5 account and terminal"
        if not bool(getattr(terminal, "connected", False)):
            return "MT5 terminal is disconnected"
        if not settings.dry_run:
            if not bool(getattr(account, "trade_allowed", False)):
                return "Trading is disabled for the active account"
            if not bool(getattr(account, "trade_expert", False)):
                return "Expert trading is disabled for the active account"
            if not bool(getattr(terminal, "trade_allowed", False)):
                return "Algo Trading is disabled in MT5"
            if bool(getattr(terminal, "tradeapi_disabled", True)):
                return "MT5 external Python trading is disabled"
        if expected_account:
            checks = {
                "login": int(getattr(account, "login", 0) or 0),
                "server": str(getattr(account, "server", "") or ""),
                "company": str(getattr(account, "company", "") or ""),
                "trade_mode": int(getattr(account, "trade_mode", -1)),
            }
            for key, actual in checks.items():
                if key in expected_account and expected_account[key] is not None:
                    expected = expected_account[key]
                    if str(actual) != str(expected):
                        return f"Active account changed ({key} mismatch)"
        return None

    def _validate_spread(
        self,
        symbol: str,
        *,
        action: Optional[str] = None,
        stop_loss: Optional[float] = None,
    ) -> Optional[str]:
        """Reject excessive cost using FX pips or crypto basis points."""
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return "Cannot fetch live spread"
        tick_age = broker_tick_age_seconds(
            float(getattr(tick, "time", 0.0) or 0.0),
            symbol=symbol,
        )
        if not math.isfinite(tick_age) or tick_age > settings.max_tick_age_seconds:
            return f"Live quote is stale ({tick_age:.1f}s old)"
        entry = None
        if action in {"BUY", "SELL"}:
            entry = tick.ask if action == "BUY" else tick.bid
        ok, reason, _ = validate_spread(
            symbol, info, tick, entry=entry, stop_loss=stop_loss
        )
        return None if ok else reason

    def _validate_stops(
        self,
        symbol: str,
        action: str,
        price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
    ) -> Optional[str]:
        if settings.require_sl_tp and (not sl_price or not tp_price):
            return "Both stop loss and take profit are required"
        if not sl_price or not tp_price:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return "Cannot query symbol stop constraints"
        sl, tp = float(sl_price), float(tp_price)
        if action == "BUY" and not (sl < price < tp):
            return "BUY requires SL < live ask < TP"
        if action == "SELL" and not (tp < price < sl):
            return "SELL requires TP < live bid < SL"
        minimum = max(float(info.trade_stops_level), 0.0) * info.point
        if minimum and (abs(price - sl) < minimum or abs(tp - price) < minimum):
            return f"SL/TP is inside broker minimum stop distance ({minimum:g})"
        return None

    def _validate_lot(self, symbol: str, lot: float) -> Optional[str]:
        """Check lot against broker min/max/step constraints."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        if lot < info.volume_min:
            return f"Lot {lot} below broker minimum {info.volume_min}"
        if lot > info.volume_max:
            return f"Lot {lot} above broker maximum {info.volume_max}"
        # Floating modulo rejects valid values such as 0.03 % 0.01. Compare
        # against the nearest integer number of broker steps instead.
        step = float(info.volume_step or 0.0)
        if step > 0 and abs((float(lot) / step) - round(float(lot) / step)) > 1e-7:
            return f"Lot {lot} is not a valid step multiple of {step}"
        return None

    def _validate_margin(self, symbol: str, action: str, lot: float) -> Optional[str]:
        """Check free margin covers required margin for this trade."""
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return "Cannot fetch live quote for margin calculation"
        price = tick.ask if action == "BUY" else tick.bid
        margin_required = mt5.order_calc_margin(order_type, symbol, lot, price)
        if margin_required is None:
            return "Broker margin calculation failed"
        account = mt5.account_info()
        if account is None:
            return "Cannot query account margin"
        free_margin = account.margin_free
        if free_margin < margin_required:
            return (f"Insufficient free margin: have ${free_margin:.2f}, "
                    f"need ${margin_required:.2f} for {lot} lot {symbol}")
        return None

    def _get_filling_mode(self, symbol_info: Any) -> int:
        """Resolve the correct filling policy for the symbol's broker."""
        fm = symbol_info.filling_mode
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    # ------------------------------------------------------------------ #
    # Core submission workers (run in thread pool)                        #
    # ------------------------------------------------------------------ #

    @serialized_mt5
    def _submit_open(
        self, symbol: str, action: str, lot: float,
        sl_price: Optional[float], tp_price: Optional[float], comment: str,
        expected_account: Optional[Dict[str, Any]],
        max_risk_usd: Optional[float],
        operator_override: bool = False,
    ) -> ExecutionResult:
        """Blocking: validates and submits a new market order."""
        err = self._validate_account_identity(expected_account)
        if err:
            return ExecutionResult(False, None, None, None, f"[Account] {err}")

        # --- Symbol ---
        err = self._validate_symbol(symbol)
        if err:
            return ExecutionResult(False, None, None, None, f"[Symbol] {err}")
        symbol_info = mt5.symbol_info(symbol)
        trade_allowed, trade_reason = validate_symbol_trade_mode(
            symbol_info, action
        )
        if not trade_allowed:
            return ExecutionResult(
                False, None, None, None, f"[Market] {trade_reason}"
            )

        # --- Quote / spread ---
        # Human direction confirmation cannot bypass final quote/cost limits.
        err = self._validate_spread(symbol, action=action, stop_loss=sl_price)
        if err:
            return ExecutionResult(False, None, None, None, f"[Spread] {err}")

        # --- Lot ---
        err = self._validate_lot(symbol, lot)
        if err:
            return ExecutionResult(False, None, None, None, f"[Lot] {err}")

        # --- Margin ---
        err = self._validate_margin(symbol, action, lot)
        if err:
            return ExecutionResult(False, None, None, None, f"[Margin] {err}")

        symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if symbol_info is None or tick is None:
            return ExecutionResult(
                False, None, None, None, "[Quote] Cannot refresh symbol metadata and live quote"
            )

        if action == "BUY":
            price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL

        # Normalize first, then perform the final risk/R:R checks against the
        # exact levels that will be sent to Pepperstone. Otherwise a CFD tick
        # grid can silently make the submitted order worse than the validated
        # plan.
        sl = (
            self._normalize_price(
                float(sl_price),
                symbol_info,
                mode="down" if action == "BUY" else "up",
            )
            if sl_price
            else 0.0
        )
        tp = (
            self._normalize_price(
                float(tp_price),
                symbol_info,
                mode="down" if action == "BUY" else "up",
            )
            if tp_price
            else 0.0
        )

        err = self._validate_stops(
            symbol,
            action,
            price,
            sl if sl else None,
            tp if tp else None,
        )
        if err:
            return ExecutionResult(False, None, None, None, f"[Stops] {err}")

        if max_risk_usd is not None and sl:
            risk_budget = float(max_risk_usd)
            if not math.isfinite(risk_budget) or risk_budget <= 0:
                return ExecutionResult(
                    False, None, None, None, "[Risk] Final risk budget is invalid"
                )
            risk_estimate = estimate_execution_risk(
                symbol,
                action,
                lot,
                price,
                sl,
                symbol_info,
            )
            if risk_estimate is None:
                return ExecutionResult(False, None, None, None,
                                       "[Risk] Broker could not recalculate execution-adjusted risk")
            if risk_estimate.total_risk_usd > risk_budget + 1e-9:
                return ExecutionResult(
                    False, None, None, None,
                    f"[Risk] Worst allowed fill risks ${risk_estimate.total_risk_usd:.2f} "
                    f"(${risk_estimate.stop_risk_usd:.2f} stop + "
                    f"${risk_estimate.slippage_reserve_usd:.2f} slippage + "
                    f"${risk_estimate.configured_cost_usd:.2f} configured costs), above "
                    f"${risk_budget:.2f} budget",
                )
            if tp:
                final_reward = mt5.order_calc_profit(
                    order_type,
                    symbol,
                    lot,
                    float(risk_estimate.worst_entry),
                    tp,
                )
                if final_reward is None or not math.isfinite(float(final_reward)):
                    return ExecutionResult(
                        False,
                        None,
                        None,
                        None,
                        "[Risk] Broker could not recalculate final target reward",
                    )
                final_net_reward = max(
                    0.0,
                    float(final_reward) - risk_estimate.configured_cost_usd,
                )
                final_rr = final_net_reward / risk_estimate.total_risk_usd
                if final_rr + 1e-9 < settings.min_risk_reward_ratio:
                    return ExecutionResult(
                        False,
                        None,
                        None,
                        None,
                        f"[Risk] Final execution-adjusted R:R is {final_rr:.2f}; "
                        f"minimum is {settings.min_risk_reward_ratio:.2f}",
                    )

        # Dry run is a simulator, independent of the broker account type.
        if settings.dry_run:
            logger.info("[DRY RUN] %s %s %s SL=%s TP=%s", action, lot, symbol, sl, tp)
            return ExecutionResult(
                True,
                999999,
                price,
                lot,
                "Dry-run execution",
                verified=True,
                state_changed=False,
                requested_volume=lot,
                remaining_volume=lot,
            )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": max(0, int(settings.max_order_deviation_points)),
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol_info),
        }

        logger.info(
            f"EXECUTE {action} | {symbol} | lot={lot} | "
            f"price={price} | SL={sl} | TP={tp}"
        )
        check = mt5.order_check(request)
        if check is None or getattr(check, "retcode", -1) != 0:
            comment_text = getattr(check, "comment", mt5.last_error()) if check else mt5.last_error()
            return ExecutionResult(False, None, None, None, f"Order check failed: {comment_text}")
        before_positions = mt5.positions_get(symbol=symbol)
        if before_positions is None:
            return ExecutionResult(
                False, None, None, None,
                f"Position snapshot failed before order send: {mt5.last_error()}",
            )
        # Account selection is global to the terminal. Re-assert identity at
        # the last possible point before sending a live mutation.
        err = self._validate_account_identity(expected_account)
        if err:
            return ExecutionResult(False, None, None, None, f"[Account] {err}")
        result = mt5.order_send(request)
        parsed = self._parse_result(result, action)
        return self._reconcile_open_result(
            result,
            parsed,
            symbol=symbol,
            action=action,
            requested_volume=lot,
            before_positions=before_positions,
            symbol_info=symbol_info,
        )

    @serialized_mt5
    def _submit_close(
        self, ticket: int, comment: str, expected_account: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Blocking: closes a full position by ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if positions is None:
            return ExecutionResult(False, ticket, None, None,
                                   f"Position query failed: {mt5.last_error()}")
        if not positions:
            return ExecutionResult(False, ticket, None, None,
                                   f"Ticket {ticket} not found — already closed?")

        pos = positions[0]
        return self._send_close_request(pos, pos.volume, comment, expected_account)

    @serialized_mt5
    def _submit_partial_close(
        self, ticket: int, close_volume: float, comment: str,
        expected_account: Optional[Dict[str, Any]],
    ) -> ExecutionResult:
        """Blocking: closes a partial volume of a position."""
        positions = mt5.positions_get(ticket=ticket)
        if positions is None:
            return ExecutionResult(False, ticket, None, None,
                                   f"Position query failed: {mt5.last_error()}")
        if not positions:
            return ExecutionResult(False, ticket, None, None,
                                   f"Ticket {ticket} not found")
        pos = positions[0]

        # Validate partial volume
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info:
            close_volume = min(float(close_volume), float(pos.volume))
            if close_volume < symbol_info.volume_min:
                return ExecutionResult(
                    False, ticket, None, None,
                    f"Requested partial close is below broker minimum {symbol_info.volume_min}",
                )
            step = float(symbol_info.volume_step or symbol_info.volume_min)
            close_volume = round(math.floor((close_volume + 1e-12) / step) * step, 8)
            remaining = round(float(pos.volume) - close_volume, 8)
            if 0 < remaining < symbol_info.volume_min:
                return ExecutionResult(
                    False, ticket, None, None,
                    "Partial close would leave a position below broker minimum volume",
                )

        return self._send_close_request(pos, close_volume, comment, expected_account)

    @serialized_mt5
    def _submit_modify(
        self,
        ticket: int,
        sl_price: Optional[float],
        tp_price: Optional[float],
        expected_account: Optional[Dict[str, Any]],
    ) -> ExecutionResult:
        """Blocking: modifies SL and/or TP of an existing position."""
        positions = mt5.positions_get(ticket=ticket)
        if positions is None:
            return ExecutionResult(False, ticket, None, None,
                                   f"Position query failed: {mt5.last_error()}")
        if not positions:
            return ExecutionResult(False, ticket, None, None,
                                   f"Ticket {ticket} not found for modification")

        pos = positions[0]
        if settings.dry_run:
            return ExecutionResult(
                False, ticket, None, None,
                "Dry-run mode will not modify a broker position",
            )
        symbol_info = mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            return ExecutionResult(
                False, ticket, None, None, "Cannot query symbol metadata"
            )
        digits = symbol_info.digits
        action = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"

        if sl_price is not None:
            requested_sl = float(sl_price)
            # A stop beyond entry is a profit floor: round inward so broker
            # tick normalization cannot reduce the intended retained profit.
            # Initial loss stops still round outward so risk is not understated.
            if action == "BUY":
                sl_mode = "up" if requested_sl > float(pos.price_open) else "down"
            else:
                sl_mode = "down" if requested_sl < float(pos.price_open) else "up"
            new_sl = self._normalize_price(
                requested_sl,
                symbol_info,
                mode=sl_mode,
            )
        else:
            new_sl = pos.sl
        new_tp = (
            self._normalize_price(float(tp_price), symbol_info)
            if tp_price is not None
            else pos.tp
        )
        current_sl = round(float(pos.sl or 0.0), digits)
        current_tp = round(float(pos.tp or 0.0), digits)
        new_sl = round(float(new_sl or 0.0), digits)
        new_tp = round(float(new_tp or 0.0), digits)

        # MT5 returns "No changes" when protection code repeats a level the
        # broker already has.  That is an idempotent success, not an execution
        # failure, and it must not enter the retry/error path.
        if new_sl == current_sl and new_tp == current_tp:
            err = self._validate_account_identity(expected_account)
            if err:
                return ExecutionResult(False, pos.ticket, None, None, f"[Account] {err}")
            logger.debug(
                "MODIFY no-op for ticket=%s; requested SL/TP already applied.",
                ticket,
            )
            return ExecutionResult(
                True,
                pos.ticket,
                float(getattr(pos, "price_current", 0.0) or 0.0),
                float(getattr(pos, "volume", 0.0) or 0.0),
                None,
                verified=True,
                state_changed=False,
            )

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return ExecutionResult(False, ticket, None, None, "Cannot fetch live quote")
        price = tick.bid if action == "BUY" else tick.ask
        err = self._validate_stops(pos.symbol, action, price, new_sl, new_tp)
        if err:
            return ExecutionResult(False, ticket, None, None, f"[Stops] {err}")

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": new_tp,
            "magic": MAGIC_NUMBER,
            "comment": f"{ORDER_COMMENT}-Modify",
        }

        logger.info(f"MODIFY ticket={ticket} | SL={new_sl} | TP={new_tp}")
        check = mt5.order_check(request)
        if check is None or getattr(check, "retcode", -1) != 0:
            comment_text = getattr(check, "comment", mt5.last_error()) if check else mt5.last_error()
            return ExecutionResult(False, pos.ticket, None, None, f"Modify check failed: {comment_text}")
        err = self._validate_account_identity(expected_account)
        if err:
            return ExecutionResult(False, pos.ticket, None, None, f"[Account] {err}")
        result = mt5.order_send(request)
        parsed = self._parse_result(result, "MODIFY")
        return self._reconcile_modify_result(
            result,
            parsed,
            ticket=int(ticket),
            # The request always carries both fields. Verify both so a broker
            # cannot accept a one-field change while silently clearing the
            # unchanged protection level.
            requested_sl=new_sl,
            requested_tp=new_tp,
            symbol_info=symbol_info,
        )

    def _send_close_request(
        self,
        pos: Any,
        volume: float,
        comment: str,
        expected_account: Optional[Dict[str, Any]],
    ) -> ExecutionResult:
        """Shared helper for full and partial close requests."""
        symbol = pos.symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return ExecutionResult(False, pos.ticket, None, None,
                                   f"Cannot get symbol info for {symbol}")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return ExecutionResult(False, pos.ticket, None, None, "Cannot fetch tick")

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        if settings.dry_run:
            return ExecutionResult(
                False, pos.ticket, price, volume,
                "Dry-run mode will not close a broker position",
            )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": max(0, int(settings.max_order_deviation_points)),
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol_info),
        }

        logger.info(
            f"CLOSE ticket={pos.ticket} | {symbol} | vol={volume} | price={price}"
        )
        check = mt5.order_check(request)
        if check is None or getattr(check, "retcode", -1) != 0:
            comment_text = getattr(check, "comment", mt5.last_error()) if check else mt5.last_error()
            return ExecutionResult(False, pos.ticket, None, volume, f"Close check failed: {comment_text}")
        err = self._validate_account_identity(expected_account)
        if err:
            return ExecutionResult(False, pos.ticket, None, volume, f"[Account] {err}")
        result = mt5.order_send(request)
        parsed = self._parse_result(result, "CLOSE")
        return self._reconcile_close_result(
            result,
            parsed,
            pos=pos,
            requested_volume=float(volume),
            symbol_info=symbol_info,
        )

    @staticmethod
    def _completed_retcode(result: Any) -> bool:
        if result is None:
            return False
        return int(getattr(result, "retcode", -1)) in {
            mt5.TRADE_RETCODE_DONE,
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", -999999),
        }

    @staticmethod
    def _volume_tolerance(symbol_info: Any) -> float:
        step = max(0.0, float(getattr(symbol_info, "volume_step", 0.0) or 0.0))
        return max(1e-8, step * 0.25)

    @staticmethod
    def _normalize_price(
        price: float, symbol_info: Any, *, mode: str = "nearest"
    ) -> float:
        """Round a price to the broker's executable tick grid."""
        digits = int(getattr(symbol_info, "digits", 5) or 5)
        tick_size = float(
            getattr(symbol_info, "trade_tick_size", 0.0)
            or getattr(symbol_info, "point", 0.0)
            or 0.0
        )
        value = float(price)
        if tick_size > 0 and math.isfinite(tick_size):
            units = value / tick_size
            if mode == "down":
                units = math.floor(units + 1e-10)
            elif mode == "up":
                units = math.ceil(units - 1e-10)
            else:
                units = round(units)
            value = units * tick_size
        return round(value, digits)

    def _reconcile_modify_result(
        self,
        raw_result: Any,
        parsed: ExecutionResult,
        *,
        ticket: int,
        requested_sl: Optional[float],
        requested_tp: Optional[float],
        symbol_info: Any,
    ) -> ExecutionResult:
        """Verify Pepperstone retained the requested SL/TP after acceptance."""
        if not self._completed_retcode(raw_result):
            return parsed

        tick_size = float(
            getattr(symbol_info, "trade_tick_size", 0.0)
            or getattr(symbol_info, "point", 0.0)
            or 1e-8
        )
        tolerance = max(1e-8, tick_size * 0.51)
        query_failed = False
        position = None
        for attempt in range(3):
            rows = mt5.positions_get(ticket=ticket)
            if rows is None:
                query_failed = True
            elif not rows:
                return ExecutionResult(
                    False,
                    ticket,
                    parsed.price,
                    parsed.volume,
                    "Modification was accepted but the position closed before "
                    "its broker protection could be verified",
                    retcode=parsed.retcode,
                    verified=True,
                    state_changed=True,
                )
            else:
                query_failed = False
                candidate = rows[0]
                sl_ok = (
                    requested_sl is None
                    or abs(float(candidate.sl or 0.0) - requested_sl) <= tolerance
                )
                tp_ok = (
                    requested_tp is None
                    or abs(float(candidate.tp or 0.0) - requested_tp) <= tolerance
                )
                if sl_ok and tp_ok:
                    position = candidate
                    break
            if attempt < 2:
                time.sleep(0.05)

        if query_failed:
            error = (
                "Modification was accepted but broker position state could not "
                f"be re-read: {mt5.last_error()}"
            )
        elif position is None:
            error = (
                "Modification was accepted but Pepperstone did not retain the "
                "requested SL/TP"
            )
        else:
            return ExecutionResult(
                True,
                ticket,
                float(getattr(position, "price_current", 0.0) or 0.0),
                float(getattr(position, "volume", 0.0) or 0.0),
                None,
                retcode=parsed.retcode,
                verified=True,
                state_changed=True,
            )
        logger.error("%s (ticket=%s)", error, ticket)
        return ExecutionResult(
            False,
            ticket,
            parsed.price,
            parsed.volume,
            error,
            retcode=parsed.retcode,
            verified=False,
            state_changed=True,
        )

    def _reconcile_open_result(
        self,
        raw_result: Any,
        parsed: ExecutionResult,
        *,
        symbol: str,
        action: str,
        requested_volume: float,
        before_positions: Any,
        symbol_info: Any,
    ) -> ExecutionResult:
        """Bind an accepted deal to the actual MT5 position and volume."""
        if not self._completed_retcode(raw_result):
            return parsed

        before_volume = {
            int(position.ticket): float(position.volume)
            for position in (before_positions or ())
        }
        expected_type = (
            mt5.POSITION_TYPE_BUY if action == "BUY" else mt5.POSITION_TYPE_SELL
        )
        candidates = []
        query_failed = False
        for attempt in range(3):
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                query_failed = True
            else:
                query_failed = False
                candidates = [
                    position
                    for position in positions
                    if int(getattr(position, "type", -1)) == expected_type
                    and int(getattr(position, "magic", 0) or 0) == MAGIC_NUMBER
                    and float(getattr(position, "volume", 0.0) or 0.0)
                    > before_volume.get(int(getattr(position, "ticket", 0)), 0.0) + 1e-9
                ]
                if candidates:
                    break
            if attempt < 2:
                time.sleep(0.05)

        if query_failed:
            return ExecutionResult(
                False,
                parsed.ticket,
                parsed.price,
                parsed.volume,
                "Order was accepted but the resulting position could not be reconciled; "
                f"manual broker-state review is required ({mt5.last_error()})",
                retcode=parsed.retcode,
                partial=parsed.partial,
                verified=False,
                state_changed=True,
                requested_volume=requested_volume,
            )
        if not candidates:
            return ExecutionResult(
                False,
                parsed.ticket,
                parsed.price,
                parsed.volume,
                "Order was accepted but no matching strategy position was found; "
                "broker history reconciliation is required",
                retcode=parsed.retcode,
                partial=parsed.partial,
                verified=True,
                state_changed=True,
                requested_volume=requested_volume,
            )

        result_ids = {
            int(value)
            for value in (
                getattr(raw_result, "order", 0),
                getattr(raw_result, "deal", 0),
            )
            if int(value or 0) > 0
        }
        preferred = [position for position in candidates if int(position.ticket) in result_ids]
        if len(preferred) == 1:
            position = preferred[0]
        elif len(candidates) == 1:
            position = candidates[0]
        else:
            return ExecutionResult(
                False,
                parsed.ticket,
                parsed.price,
                parsed.volume,
                "Order was accepted but matched multiple new strategy positions; "
                "automatic ticket binding is unsafe",
                retcode=parsed.retcode,
                partial=parsed.partial,
                verified=True,
                state_changed=True,
                requested_volume=requested_volume,
            )

        prior = before_volume.get(int(position.ticket), 0.0)
        actual_volume = max(0.0, float(position.volume) - prior)
        tolerance = self._volume_tolerance(symbol_info)
        if actual_volume <= tolerance:
            return ExecutionResult(
                False,
                int(position.ticket),
                float(getattr(position, "price_open", parsed.price or 0.0)),
                actual_volume,
                "Accepted order did not produce a positive reconciled position volume",
                retcode=parsed.retcode,
                verified=True,
                state_changed=True,
                requested_volume=requested_volume,
                remaining_volume=float(position.volume),
            )
        if actual_volume > float(requested_volume) + tolerance:
            return ExecutionResult(
                False,
                int(position.ticket),
                float(getattr(position, "price_open", parsed.price or 0.0)),
                actual_volume,
                "Reconciled position volume exceeds the requested order volume",
                retcode=parsed.retcode,
                verified=True,
                state_changed=True,
                requested_volume=requested_volume,
                remaining_volume=float(position.volume),
            )

        partial = actual_volume + tolerance < float(requested_volume)
        if partial:
            logger.warning(
                "%s partially filled: requested=%s actual=%s position=%s",
                action, requested_volume, actual_volume, position.ticket,
            )
        return ExecutionResult(
            True,
            int(position.ticket),
            float(getattr(position, "price_open", parsed.price or 0.0)),
            actual_volume,
            None,
            retcode=parsed.retcode,
            partial=partial,
            verified=True,
            state_changed=True,
            requested_volume=requested_volume,
            remaining_volume=float(position.volume),
        )

    def _reconcile_close_result(
        self,
        raw_result: Any,
        parsed: ExecutionResult,
        *,
        pos: Any,
        requested_volume: float,
        symbol_info: Any,
    ) -> ExecutionResult:
        """Verify that a close removed exactly the requested position volume."""
        if not self._completed_retcode(raw_result):
            return parsed

        before_volume = float(pos.volume)
        expected_remaining = max(0.0, before_volume - float(requested_volume))
        tolerance = self._volume_tolerance(symbol_info)
        remaining: Optional[float] = None
        query_failed = False
        complete = False
        for attempt in range(3):
            positions = mt5.positions_get(ticket=int(pos.ticket))
            if positions is None:
                query_failed = True
            else:
                query_failed = False
                remaining = float(positions[0].volume) if positions else 0.0
                complete = abs(remaining - expected_remaining) <= tolerance
                if complete:
                    break
            if attempt < 2:
                time.sleep(0.05)

        if query_failed or remaining is None:
            return ExecutionResult(
                False,
                int(pos.ticket),
                parsed.price,
                parsed.volume,
                "Close was accepted but the remaining position could not be reconciled; "
                f"manual broker-state review is required ({mt5.last_error()})",
                retcode=parsed.retcode,
                partial=parsed.partial,
                verified=False,
                state_changed=True,
                requested_volume=requested_volume,
            )

        actual_closed = max(0.0, before_volume - remaining)
        if complete:
            return ExecutionResult(
                True,
                int(pos.ticket),
                parsed.price,
                actual_closed,
                None,
                retcode=parsed.retcode,
                partial=False,
                verified=True,
                state_changed=actual_closed > tolerance,
                requested_volume=requested_volume,
                remaining_volume=remaining,
            )

        partial = actual_closed > tolerance or (
            int(getattr(raw_result, "retcode", -1))
            == getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", -999999)
        )
        error = (
            f"Close did not complete the requested volume: requested {requested_volume:g}, "
            f"closed {actual_closed:g}, remaining {remaining:g}"
        )
        logger.error("%s (ticket=%s)", error, pos.ticket)
        return ExecutionResult(
            False,
            int(pos.ticket),
            parsed.price,
            actual_closed,
            error,
            retcode=parsed.retcode,
            partial=partial,
            verified=True,
            state_changed=actual_closed > tolerance,
            requested_volume=requested_volume,
            remaining_volume=remaining,
        )

    # ------------------------------------------------------------------ #
    # Retry wrapper                                                        #
    # ------------------------------------------------------------------ #

    async def _with_retry(self, func, *args) -> ExecutionResult:
        """
        Runs a blocking executor function in a thread with automatic retries
        for transient MT5 error codes.
        """
        if not await self._conn.is_connected():
            logger.warning("MT5 offline — attempting reconnect before execution...")
            if not await self._conn.initialize():
                return ExecutionResult(False, None, None, None,
                                       "MT5 connection unavailable")

        last_result = ExecutionResult(False, None, None, None, "No attempts made")
        for attempt in range(1, MAX_RETRIES + 1):
            start = time.monotonic()
            worker = asyncio.create_task(asyncio.to_thread(func, *args))
            self._workers.add(worker)
            worker.add_done_callback(self._workers.discard)
            # Shield keeps the native call registered if its caller is
            # cancelled. Engine shutdown waits for these workers before
            # calling mt5.shutdown().
            last_result = await asyncio.shield(worker)
            elapsed = time.monotonic() - start

            logger.debug(
                f"Execution attempt {attempt}/{MAX_RETRIES} | "
                f"success={last_result.success} | {elapsed:.3f}s"
            )

            if last_result.success:
                return last_result

            # Only quote-refresh failures are safe to retry. Timeout and
            # connection outcomes are ambiguous and may already have changed
            # broker state, so they are deliberately not in this set.
            if last_result.retcode in _RETRYABLE_CODES:
                logger.warning(
                    f"Transient error (code {last_result.retcode}) on attempt {attempt}. "
                    f"Retrying in {RETRY_DELAY}s..."
                )
                await asyncio.sleep(RETRY_DELAY * attempt)
                continue

            # Non-retryable failure — bail immediately
            break

        logger.error(
            "Execution failed after %d attempt(s): %s", attempt, last_result.error
        )
        return last_result

    # ------------------------------------------------------------------ #
    # Result parser                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_result(result: Any, action: str) -> ExecutionResult:
        """Converts raw mt5.order_send() result to ExecutionResult."""
        if result is None:
            return ExecutionResult(
                False, None, None, None,
                f"{action} order returned None — MT5 may be disconnected",
                retcode=None,
            )

        retcode = getattr(result, "retcode", None)
        order = getattr(result, "order", 0) or 0
        deal = getattr(result, "deal", 0) or 0
        price = getattr(result, "price", None)
        volume = getattr(result, "volume", None)
        comment = str(getattr(result, "comment", "No broker comment") or "No broker comment")

        if retcode == mt5.TRADE_RETCODE_DONE:
            ticket = order or deal or None
            logger.info(
                f"{action} SUCCESS | ticket={ticket} | "
                f"price={price} | vol={volume}"
            )
            return ExecutionResult(
                True,
                ticket=ticket,
                price=price,
                volume=volume,
                error=None,
                retcode=retcode,
                verified=False,
                state_changed=True,
                requested_volume=volume,
            )

        if retcode == getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", -999999):
            ticket = order or deal or None
            logger.warning("%s received a partial fill; broker reconciliation required.", action)
            return ExecutionResult(
                False,
                ticket=ticket,
                price=price,
                volume=volume,
                error="Broker returned a partial fill; requested state is not yet verified",
                retcode=retcode,
                partial=True,
                verified=False,
                state_changed=True,
                requested_volume=volume,
            )

        err_msg = f"retcode={retcode} | {comment}"
        logger.error(f"{action} FAILED | {err_msg}")
        return ExecutionResult(
            False, order or None, None, None,
            err_msg, retcode=retcode,
        )

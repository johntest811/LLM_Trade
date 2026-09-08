"""Reconstruct complete positions from MT5 deal history."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Any, Dict, Iterable, List, Set

from mt5.safe_api import mt5

from mt5.timebase import (
    infer_positive_server_offset_seconds,
    normalized_broker_epoch,
)


logger = logging.getLogger("TradingSystem.Reconciliation")


REASON_NAMES = {
    0: "CLIENT",
    1: "MOBILE",
    2: "WEB",
    3: "EXPERT",
    4: "STOP_LOSS",
    5: "TAKE_PROFIT",
    6: "STOP_OUT",
    7: "SIGNAL",
}


def aggregate_closed_positions(
    deals: Iterable[Any],
    strategy_magic: int,
    active_position_ids: Set[int] | None = None,
    server_offset_seconds: int = 0,
) -> List[Dict[str, Any]]:
    """Pure aggregation function, kept separate for deterministic testing."""
    active_position_ids = active_position_ids or set()
    grouped: Dict[int, List[Any]] = defaultdict(list)
    for deal in deals:
        position_id = int(getattr(deal, "position_id", 0) or 0)
        if position_id:
            grouped[position_id].append(deal)

    rows: List[Dict[str, Any]] = []
    for position_id, position_deals in grouped.items():
        position_deals.sort(key=lambda item: (item.time, getattr(item, "time_msc", 0)))
        entries = [deal for deal in position_deals if int(deal.entry) in (0, 2)]
        exits = [deal for deal in position_deals if int(deal.entry) in (1, 3)]
        if not entries or not exits or position_id in active_position_ids:
            continue
        if not any(int(getattr(deal, "magic", 0) or 0) == strategy_magic for deal in entries):
            continue

        entry_volume = sum(float(deal.volume) for deal in entries)
        exit_volume = sum(float(deal.volume) for deal in exits)
        if exit_volume + 1e-9 < entry_volume:
            continue
        open_price = sum(float(d.price) * float(d.volume) for d in entries) / max(entry_volume, 1e-12)
        close_price = sum(float(d.price) * float(d.volume) for d in exits) / max(exit_volume, 1e-12)
        gross = sum(float(getattr(d, "profit", 0.0) or 0.0) for d in position_deals)
        commission = sum(float(getattr(d, "commission", 0.0) or 0.0) for d in position_deals)
        swap = sum(float(getattr(d, "swap", 0.0) or 0.0) for d in position_deals)
        fee = sum(float(getattr(d, "fee", 0.0) or 0.0) for d in position_deals)
        net_profit = round(gross + commission + swap + fee, 8)
        entry = entries[0]
        final_exit = exits[-1]
        broker_reason = REASON_NAMES.get(
            int(getattr(final_exit, "reason", -1)), "OTHER"
        )
        # MT5 correctly reports every activated SL as DEAL_REASON_SL, including
        # a break-even, trailing, or profit-floor stop that has already crossed
        # the entry price. Preserve the broker event while giving profitable
        # exits an unambiguous strategy-facing label.
        close_reason = (
            "PROTECTIVE_STOP"
            if broker_reason == "STOP_LOSS" and net_profit > 0
            else broker_reason
        )
        rows.append({
            "position_id": position_id,
            "open_time": datetime.fromtimestamp(
                normalized_broker_epoch(entry.time, server_offset_seconds),
                timezone.utc,
            ).isoformat(),
            "close_time": datetime.fromtimestamp(
                normalized_broker_epoch(final_exit.time, server_offset_seconds),
                timezone.utc,
            ).isoformat(),
            "symbol": str(entry.symbol),
            "direction": "BUY" if int(entry.type) == 0 else "SELL",
            "volume": round(entry_volume, 8),
            "open_price": open_price,
            "close_price": close_price,
            "gross_profit": round(gross, 8),
            "commission": round(commission, 8),
            "swap": round(swap, 8),
            "fee": round(fee, 8),
            "net_profit": net_profit,
            "magic": strategy_magic,
            "close_reason": close_reason,
        })
    return sorted(rows, key=lambda row: row["close_time"], reverse=True)


class BrokerHistoryReconciler:
    def __init__(self, strategy_magic: int, symbols: Iterable[str] | None = None) -> None:
        self.strategy_magic = strategy_magic
        self.symbols = [str(symbol).strip() for symbol in (symbols or []) if str(symbol).strip()]
        self.server_offset_seconds = 0

    def _detect_server_offset(self, now_epoch: float) -> int:
        """Detect the broker wall-clock offset from a current configured quote."""
        for symbol in self.symbols:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            offset = infer_positive_server_offset_seconds(
                getattr(tick, "time", 0.0), now_epoch=now_epoch
            )
            if offset:
                if offset != self.server_offset_seconds:
                    logger.info(
                        "Detected broker history clock offset UTC+%d.",
                        offset // 3600,
                    )
                self.server_offset_seconds = offset
                return offset
        return self.server_offset_seconds

    def fetch(self, days: int = 90) -> List[Dict[str, Any]]:
        now_epoch = time.time()
        offset = self._detect_server_offset(now_epoch)
        # This Pepperstone terminal exposes broker wall-clock values as epoch
        # seconds. Query in that same clock domain, then normalize every stored
        # deal back to UTC. Otherwise the most recent UTC+2/+3 deals disappear
        # from reconciliation until the operating-system clock catches up.
        broker_now = datetime.fromtimestamp(now_epoch + offset, timezone.utc)
        deals = mt5.history_deals_get(
            broker_now - timedelta(days=days), broker_now
        )
        if deals is None:
            raise RuntimeError(f"MT5 history_deals_get failed: {mt5.last_error()}")
        positions = mt5.positions_get()
        if positions is None:
            raise RuntimeError(f"MT5 positions_get failed during reconciliation: {mt5.last_error()}")
        active_ids = {int(p.ticket) for p in positions}
        return aggregate_closed_positions(
            deals,
            self.strategy_magic,
            active_ids,
            server_offset_seconds=offset,
        )

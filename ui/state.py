"""
ui/state.py
Thread-safe singleton store for all live dashboard state.

The trading engine writes updates here; the WebSocket broadcaster
reads from here every second and pushes JSON to connected browsers.
"""
import asyncio
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional


@dataclass
class LivePosition:
    ticket: int
    symbol: str
    direction: str      # BUY | SELL
    lot: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit_usd: float
    profit_pips: float
    peak_pips: float
    duration_min: float
    peak_profit_usd: float = 0.0
    trough_pips: float = 0.0
    trough_profit_usd: float = 0.0
    estimated_net_profit_usd: float = 0.0
    profit_lock_armed: bool = False
    magic: int = 0
    comment: str = ""
    risk_to_sl_usd: float = 0.0
    risk_pct_balance: float = 0.0
    reward_to_tp_usd: float = 0.0
    planned_rr: float = 0.0
    bot_owned: bool = False


@dataclass
class ClosedTrade:
    ticket: int
    symbol: str
    direction: str
    lot: float
    open_price: float
    close_price: float
    profit_usd: float
    profit_pips: float
    rr_achieved: float
    close_time: str
    close_reason: str = ""
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    mfe_usd: float = 0.0
    mae_usd: float = 0.0


@dataclass
class LLMMetrics:
    last_action: str = "—"
    last_symbol: str = "—"
    confidence: float = 0.0
    reasoning: str = "—"
    trade_management: str = "—"
    inference_time_s: float = 0.0
    prompt_tokens: int = 0
    provider: str = "local"
    model: str = ""
    trace_id: str = ""
    last_updated: str = "—"


@dataclass
class SymbolDecision:
    symbol: str
    stage: str = "WAITING"
    action: str = "—"
    confidence: float = 0.0
    reasoning: str = "Waiting for a completed M5 candle."
    trade_management: str = ""
    inference_time_s: float = 0.0
    candle_time: str = ""
    updated_at: str = ""
    gate_reason: str = ""
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    planned_rr: float = 0.0
    manual_override_available: bool = False


@dataclass
class ExecutionReadiness:
    ready: bool = False
    severity: str = "BLOCKED"
    code: str = "ENGINE_STOPPED"
    reason: str = "Decision engine is stopped."
    checks: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: str = ""


@dataclass
class SystemMetrics:
    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_pct: float = 0.0
    gpu_util_pct: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_mem_pct: float = 0.0
    gpu_temp_c: float = 0.0


@dataclass
class ShadowMetrics:
    enabled: bool = False
    pending: int = 0
    resolved: int = 0
    wins: int = 0
    losses: int = 0
    ambiguous: int = 0
    win_rate_pct: float = 0.0
    expectancy_r: float = 0.0
    updated_at: str = ""


@dataclass
class AccountMetrics:
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level_pct: float = 0.0
    currency: str = "USD"
    daily_pl: float = 0.0
    open_pl: float = 0.0
    total_closed_pl: float = 0.0
    win_rate_pct: float = 0.0
    avg_rr: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    broker: str = "—"
    server: str = "—"
    account_suffix: str = "—"
    account_mode: str = "UNKNOWN"
    account_trade_allowed: bool = False
    expert_trading_allowed: bool = False
    terminal_connected: bool = False
    terminal_trade_allowed: bool = False
    tradeapi_disabled: bool = True
    leverage: int = 0
    margin_usage_pct: float = 0.0
    portfolio_risk_usd: float = 0.0
    portfolio_risk_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy_usd: float = 0.0
    max_closed_drawdown_usd: float = 0.0
    strategy_evidence: str = "UNVALIDATED"


@dataclass
class AutomationMetrics:
    entries_armed: bool = False
    autonomous_enabled: bool = False
    autonomous_status: str = "DISABLED"
    autonomous_account_suffix: str = "—"
    last_health_check: str = "—"
    engine_heartbeat_utc: str = ""
    broker_poll_heartbeat_utc: str = ""
    decision_provider_inference_ready: bool = False
    safety_status: str = "LOCKED"
    safety_reason: str = "New entries are disarmed after startup"
    dry_run: bool = True
    strategy_magic: int = 0
    config_fingerprint: str = ""
    provider: str = "local"
    model: str = "—"
    llm_online: bool = False
    last_reconciled: str = "—"
    scan_timeframe: str = "M5 close"
    confirmation_timeframes: str = "M15 / H1 / H4"
    scan_status: str = "NOT STARTED"
    last_scan: str = "—"
    active_symbols: List[str] = field(default_factory=list)
    history_healthy: bool = False
    history_status: str = "NOT RECONCILED"


@dataclass
class SymbolPrice:
    symbol: str
    bid: float
    ask: float
    spread_pips: float
    trend: str = "—"
    adx: float = 0.0
    spread_value: float = 0.0
    spread_unit: str = "pips"
    asset_class: str = "FX/CFD"
    updated_at: str = ""


class DashboardState:
    """
    Global singleton state container.
    All mutating methods hold a threading.Lock for safety between the
    asyncio event loop and any thread-pool workers.
    """
    _lock = threading.RLock()

    def __init__(self):
        self.engine_running: bool = False
        self.engine_start_time: Optional[datetime] = None
        self.account = AccountMetrics()
        self.llm = LLMMetrics()
        self.system = SystemMetrics()
        self.shadow = ShadowMetrics()
        self.automation = AutomationMetrics()
        self.positions: List[LivePosition] = []
        self.closed_trades: List[ClosedTrade] = []
        self.prices: Dict[str, SymbolPrice] = {}
        self.symbol_decisions: Dict[str, SymbolDecision] = {}
        self.market_fits: Dict[str, Dict[str, Any]] = {}
        self.readiness = ExecutionReadiness()
        self.logs: List[str] = []          # last 200 log lines
        self._daily_start_balance: float = 0.0
        self._realized_today: float = 0.0
        self._trade_day: date = date.today()
        self.tick_stream: List[Dict[str, Any]] = []
        self._payload_sequence: int = 0

    # ----------------------------------------------------------------
    def update_account(self, info: Dict[str, Any]) -> None:
        with self._lock:
            a = self.account
            new_suffix = str(info.get("login", ""))[-4:] or "—"
            if a.account_suffix != "—" and new_suffix != a.account_suffix:
                self.closed_trades = []
                self._realized_today = 0.0
                self._daily_start_balance = 0.0
            a.account_suffix = new_suffix
            a.balance = info.get("balance", 0.0)
            a.equity = info.get("equity", 0.0)
            a.margin = info.get("margin", 0.0)
            a.free_margin = info.get("margin_free", 0.0)
            a.margin_level_pct = info.get("margin_level", 0.0)
            a.currency = info.get("currency", "USD")
            a.broker = info.get("company", "—")
            a.server = info.get("server", "—")
            a.account_mode = info.get("trade_mode_name", "UNKNOWN")
            a.account_trade_allowed = bool(info.get("account_trade_allowed", False))
            a.expert_trading_allowed = bool(info.get("expert_trading_allowed", False))
            a.terminal_connected = bool(info.get("terminal_connected", False))
            a.terminal_trade_allowed = bool(info.get("terminal_trade_allowed", False))
            a.tradeapi_disabled = bool(info.get("tradeapi_disabled", True))
            a.leverage = int(info.get("leverage", 0) or 0)
            a.margin_usage_pct = round(a.margin / a.equity * 100.0, 1) if a.equity else 0.0
            # Daily P/L — reset at day boundary
            today = date.today()
            if today != self._trade_day:
                self._daily_start_balance = a.balance
                self._trade_day = today
            if self._daily_start_balance == 0.0:
                self._daily_start_balance = a.balance
            a.daily_pl = round(self._realized_today + info.get("profit", 0.0), 2)

    def update_positions(self, raw_positions: List[Dict[str, Any]]) -> None:
        with self._lock:
            live: List[LivePosition] = []
            total_open_pl = 0.0
            for p in raw_positions:
                pos_type = p.get("type", 0)
                direction = "BUY" if pos_type == 0 else "SELL"
                profit = p.get("profit", 0.0)
                total_open_pl += profit
                live.append(LivePosition(
                    ticket=p.get("ticket", 0),
                    symbol=p.get("symbol", ""),
                    direction=direction,
                    lot=p.get("volume", 0.0),
                    open_price=p.get("price_open", 0.0),
                    current_price=p.get("price_current", 0.0),
                    sl=p.get("sl", 0.0),
                    tp=p.get("tp", 0.0),
                    profit_usd=round(profit, 2),
                    profit_pips=round(p.get("profit_pips", 0.0), 1),
                    peak_pips=round(p.get("peak_profit_pips", 0.0), 1),
                    peak_profit_usd=round(p.get("peak_profit_usd", 0.0), 2),
                    trough_pips=round(p.get("trough_profit_pips", 0.0), 1),
                    trough_profit_usd=round(
                        p.get("trough_profit_usd", 0.0), 2
                    ),
                    estimated_net_profit_usd=round(
                        p.get("estimated_net_profit_usd", profit), 2
                    ),
                    profit_lock_armed=bool(p.get("profit_lock_armed", False)),
                    duration_min=round(p.get("duration_min", 0.0), 0),
                    magic=int(p.get("magic", 0) or 0),
                    comment=str(p.get("comment", "")),
                    risk_to_sl_usd=round(p.get("risk_to_sl_usd", 0.0), 2),
                    risk_pct_balance=round(p.get("risk_pct_balance", 0.0), 2),
                    reward_to_tp_usd=round(p.get("reward_to_tp_usd", 0.0), 2),
                    planned_rr=round(p.get("planned_rr", 0.0), 2),
                    bot_owned=bool(p.get("bot_owned", False)),
                ))
            self.positions = live
            self.account.open_pl = round(total_open_pl, 2)
            self.account.daily_pl = round(self._realized_today + total_open_pl, 2)
            self.account.portfolio_risk_usd = round(sum(p.risk_to_sl_usd for p in live), 2)
            self.account.portfolio_risk_pct = round(
                self.account.portfolio_risk_usd / self.account.balance * 100.0, 2
            ) if self.account.balance else 0.0

    def add_closed_trade(self, trade: Dict[str, Any]) -> None:
        with self._lock:
            ct = ClosedTrade(
                ticket=trade.get("ticket", 0),
                symbol=trade.get("symbol", ""),
                direction=trade.get("direction", ""),
                lot=trade.get("lot", 0.0),
                open_price=trade.get("open_price", 0.0),
                close_price=trade.get("close_price", 0.0),
                profit_usd=round(trade.get("profit_usd", 0.0), 2),
                profit_pips=round(trade.get("profit_pips", 0.0), 1),
                rr_achieved=round(trade.get("rr_achieved", 0.0), 2),
                close_time=trade.get("close_time", ""),
                close_reason=trade.get("close_reason", ""),
                mfe_pips=round(trade.get("mfe_pips", 0.0), 1),
                mae_pips=round(trade.get("mae_pips", 0.0), 1),
                mfe_usd=round(trade.get("mfe_usd", 0.0), 2),
                mae_usd=round(trade.get("mae_usd", 0.0), 2),
            )
            self.closed_trades.insert(0, ct)
            self.closed_trades = self.closed_trades[:100]   # keep last 100
            self._recalc_stats()

    def load_trade_history(self, history: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.closed_trades = []
            for t in history[:100]:
                self.closed_trades.append(ClosedTrade(
                    ticket=t.get("ticket", 0),
                    symbol=t.get("symbol", ""),
                    direction=t.get("direction", t.get("action", "")),
                    lot=t.get("lot_size", t.get("lot", 0.0)),
                    open_price=t.get("open_price", t.get("price", 0.0)),
                    close_price=t.get("close_price", 0.0),
                    profit_usd=round(t.get("profit", 0.0), 2),
                    profit_pips=round(t.get("profit_pips", 0.0), 1),
                    rr_achieved=round(t.get("rr_achieved", 0.0), 2),
                    close_time=t.get("time", t.get("close_time", "")),
                    close_reason=t.get("close_reason", ""),
                    mfe_pips=round(t.get("mfe_pips", 0.0), 1),
                    mae_pips=round(t.get("mae_pips", 0.0), 1),
                    mfe_usd=round(t.get("mfe_usd", 0.0), 2),
                    mae_usd=round(t.get("mae_usd", 0.0), 2),
                ))
            self._recalc_stats()

    def load_broker_history(self, history: List[Dict[str, Any]]) -> None:
        """Load broker-confirmed, cost-inclusive closed positions."""
        with self._lock:
            today_utc = datetime.now(timezone.utc).date()
            self._realized_today = 0.0
            for trade in history:
                try:
                    closed = datetime.fromisoformat(str(trade.get("close_time", "")).replace("Z", "+00:00"))
                    if closed.tzinfo is None:
                        closed = closed.replace(tzinfo=timezone.utc)
                    if closed.astimezone(timezone.utc).date() == today_utc:
                        self._realized_today += float(trade.get("net_profit", 0.0))
                except (TypeError, ValueError):
                    continue
            self.closed_trades = [
                ClosedTrade(
                    ticket=int(t.get("position_id", 0)),
                    symbol=t.get("symbol", ""),
                    direction=t.get("direction", ""),
                    lot=float(t.get("volume", 0.0)),
                    open_price=float(t.get("open_price", 0.0)),
                    close_price=float(t.get("close_price", 0.0)),
                    profit_usd=round(float(t.get("net_profit", 0.0)), 2),
                    profit_pips=round(float(t.get("profit_pips", 0.0)), 1),
                    rr_achieved=round(float(t.get("rr_achieved", 0.0)), 2),
                    close_time=t.get("close_time", ""),
                    close_reason=str(t.get("close_reason", "")),
                    mfe_pips=round(float(t.get("mfe_pips", 0.0)), 1),
                    mae_pips=round(float(t.get("mae_pips", 0.0)), 1),
                    mfe_usd=round(float(t.get("mfe_usd", 0.0)), 2),
                    mae_usd=round(float(t.get("mae_usd", 0.0)), 2),
                )
                for t in history[:200]
            ]
            self._recalc_stats()
            self.account.daily_pl = round(self._realized_today + self.account.open_pl, 2)

    def update_automation(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self.automation, key):
                    setattr(self.automation, key, value)

    def _recalc_stats(self) -> None:
        """Recalculate win rate, avg RR, and total closed P/L from history."""
        a = self.account
        # Broker-confirmed break-even exits are still strategy observations and
        # must count toward sample size/expectancy (but not as losses).
        closed = list(self.closed_trades)
        wins = [t for t in closed if t.profit_usd > 0]
        losses = [t for t in closed if t.profit_usd < 0]
        a.total_trades = len(closed)
        a.winning_trades = len(wins)
        a.losing_trades = len(losses)
        a.win_rate_pct = round((len(wins) / len(closed)) * 100, 1) if closed else 0.0
        a.avg_rr = round(sum(t.rr_achieved for t in closed) / len(closed), 2) if closed else 0.0
        a.total_closed_pl = round(sum(t.profit_usd for t in closed), 2)
        gross_wins = sum(t.profit_usd for t in closed if t.profit_usd > 0)
        gross_losses = abs(sum(t.profit_usd for t in losses))
        a.profit_factor = round(gross_wins / gross_losses, 2) if gross_losses else (99.0 if gross_wins else 0.0)
        a.expectancy_usd = round(a.total_closed_pl / len(closed), 3) if closed else 0.0
        running = peak = max_drawdown = 0.0
        for trade in reversed(closed):
            running += trade.profit_usd
            peak = max(peak, running)
            max_drawdown = max(max_drawdown, peak - running)
        a.max_closed_drawdown_usd = round(max_drawdown, 2)
        if len(closed) < 30:
            a.strategy_evidence = f"UNVALIDATED · {len(closed)}/30 CLOSED TRADES"
        elif a.profit_factor > 1.0 and a.expectancy_usd > 0:
            a.strategy_evidence = "POSITIVE IN-SAMPLE · NEEDS WALK-FORWARD REVIEW"
        else:
            a.strategy_evidence = "NEGATIVE/INCONCLUSIVE EVIDENCE"

    def update_prices(self, symbol: str, bid: float, ask: float,
                      point: float, trend: str = "—", adx: float = 0.0) -> None:
        if any(c in symbol.upper() for c in ["ETH", "LTC", "BTC"]):
            pip_multi = 1.0
        elif "XRP" in symbol.upper():
            pip_multi = 100.0
        elif any(j in symbol.upper() for j in ["JPY", "XAU", "GOLD", "XAG"]):
            pip_multi = 100.0
        else:
            pip_multi = 10000.0
        spread_pips = round((ask - bid) / point * (point * pip_multi), 2) if point > 0 else 0.0
        with self._lock:
            upper = symbol.upper()
            crypto = any(code in upper for code in (
                "BTC", "ETH", "LTC", "XRP", "ADA", "SOL", "DOGE", "AVAX", "BNB"
            ))
            midpoint = (ask + bid) / 2.0 if ask > 0 and bid > 0 else 0.0
            spread_bps = (ask - bid) / midpoint * 10000.0 if midpoint else 0.0
            self.prices[symbol] = SymbolPrice(
                symbol=symbol, bid=bid, ask=ask,
                spread_pips=spread_pips, trend=trend, adx=adx,
                spread_value=round(spread_bps if crypto else spread_pips, 2),
                spread_unit="bps" if crypto else "pips",
                asset_class="CRYPTO" if crypto else "FX/CFD",
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    def update_symbol_decision(self, symbol: str, **values: Any) -> None:
        with self._lock:
            decision = self.symbol_decisions.get(symbol) or SymbolDecision(symbol=symbol)
            for key, value in values.items():
                if hasattr(decision, key):
                    setattr(decision, key, value)
            decision.updated_at = datetime.now(timezone.utc).isoformat()
            self.symbol_decisions[symbol] = decision

    def update_market_fit(self, symbol: str, fit: Dict[str, Any]) -> None:
        with self._lock:
            self.market_fits[symbol] = dict(fit)

    def retain_market_scope(self, symbols: List[str]) -> None:
        """Remove quote/fit rows which are outside the current market session."""
        allowed = {
            str(symbol).strip().upper()
            for symbol in symbols
            if str(symbol).strip()
        }
        with self._lock:
            self.market_fits = {
                symbol: fit
                for symbol, fit in self.market_fits.items()
                if symbol.upper() in allowed
            }
            self.prices = {
                symbol: price
                for symbol, price in self.prices.items()
                if symbol.upper() in allowed
            }
            self.tick_stream = [
                tick
                for tick in self.tick_stream
                if str(tick.get("symbol", "")).upper() in allowed
            ]

    def update_readiness(
        self,
        *,
        ready: bool,
        severity: str,
        code: str,
        reason: str,
        checks: List[Dict[str, Any]],
    ) -> None:
        with self._lock:
            self.readiness = ExecutionReadiness(
                ready=ready,
                severity=severity,
                code=code,
                reason=reason,
                checks=list(checks),
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    def update_llm(
        self, action: str, symbol: str, confidence: float,
        reasoning: str, trade_management: str,
        inference_time_s: float, prompt_tokens: int,
        provider: str = "", model: str = "", trace_id: str = "",
    ) -> None:
        with self._lock:
            self.llm = LLMMetrics(
                last_action=action,
                last_symbol=symbol,
                confidence=confidence,
                reasoning=reasoning,
                trade_management=trade_management,
                inference_time_s=round(inference_time_s, 2),
                prompt_tokens=prompt_tokens,
                provider=provider,
                model=model,
                trace_id=trace_id,
                last_updated=datetime.now().strftime("%H:%M:%S"),
            )

    def update_system(self, metrics) -> None:
        with self._lock:
            self.system = SystemMetrics(
                cpu_pct=metrics.cpu_pct,
                ram_used_gb=metrics.ram_used_gb,
                ram_total_gb=metrics.ram_total_gb,
                ram_pct=metrics.ram_pct,
                gpu_util_pct=metrics.gpu_util_pct,
                gpu_mem_used_mb=metrics.gpu_mem_used_mb,
                gpu_mem_total_mb=metrics.gpu_mem_total_mb,
                gpu_mem_pct=metrics.gpu_mem_pct,
                gpu_temp_c=metrics.gpu_temp_c,
            )

    def update_shadow(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self.shadow, key):
                    setattr(self.shadow, key, value)
            self.shadow.updated_at = datetime.now(timezone.utc).isoformat()

    def append_log(self, line: str) -> None:
        with self._lock:
            self.logs.append(line)
            if len(self.logs) > 200:
                self.logs = self.logs[-200:]

    def add_tick(self, symbol: str, bid: float, ask: float) -> None:
        with self._lock:
            self.tick_stream.append({
                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 5)
            })
            if len(self.tick_stream) > 50:
                self.tick_stream = self.tick_stream[-50:]

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            self._payload_sequence += 1
            return {
                "sequence": self._payload_sequence,
                "server_time_utc": datetime.now(timezone.utc).isoformat(),
                "engine_running": self.engine_running,
                "account": asdict(self.account),
                "llm": asdict(self.llm),
                "system": asdict(self.system),
                "shadow": asdict(self.shadow),
                "automation": asdict(self.automation),
                "readiness": asdict(self.readiness),
                "positions": [asdict(p) for p in self.positions],
                "closed_trades": [asdict(t) for t in self.closed_trades[:20]],
                "prices": {k: asdict(v) for k, v in self.prices.items()},
                "symbol_decisions": {k: asdict(v) for k, v in self.symbol_decisions.items()},
                "market_fits": dict(self.market_fits),
                "logs": self.logs[-50:],
                "tick_stream": self.tick_stream,
            }


# Global singleton
dashboard_state = DashboardState()

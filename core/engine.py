"""
core/engine.py — Production local AI trading engine loop.

Orchestrates data collection, technical indicators, swing/breakout analysis,
prompt building, provider-aware model decisions, risk management, and order execution.
Sends real-time updates to the global dashboard state.
"""
import asyncio
import copy
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from mt5.safe_api import mt5

from app_config.settings import settings
from ui.state import dashboard_state
from core.analysis_engine import MarketAnalysisEngine
from core.forex_context import build_currency_context
from core.market_selector import AdaptiveMarketSelector
from core.market_universe import discovery_batch, tradable_symbols
from core.opportunities import screen_opportunities
from core.profit_retention import RetentionState, advance_retention
from core.entry_retest import annotate_retest_continuation
from core.entry_momentum import aligned_structure_allows_adx_decline
from core.range_reversion import annotate_range_reversion
from core.shadow_outcomes import (
    evaluate_exit_counterfactual,
    evaluate_shadow_candidate,
)
from core.trade_planner import DeterministicTradePlanner, TradePlan
from prompt_builder.generator import PromptGenerator
from risk.manager import RiskManager
from risk.instruments import downside_risk_usd, pip_size
from risk.execution_costs import configured_execution_cost_usd
from core.validator import DecisionValidator
from llm.client import DeterministicDecisionProvider
from core.evidence import (
    build_evidence_ids,
    has_directional_trigger,
    permitted_entry_actions,
)
from database.replay_logger import TradeReplayLogger
from app_config.paths import DEFAULT_DB_PATH
from database.reconciliation import BrokerHistoryReconciler
from mt5.timebase import (
    infer_positive_server_offset_seconds,
    normalized_broker_epoch,
)

logger = logging.getLogger("TradingSystem.CoreEngine")


def is_weekend() -> bool:
    """Checks if standard Forex/Stock markets are closed in UTC."""
    now = datetime.now(timezone.utc)
    day = now.weekday()  # 5=Saturday, 6=Sunday
    if day == 4 and now.hour >= 22:
        return True
    if day == 5:
        return True
    if day == 6 and now.hour < 22:
        return True
    return False


class TradingEngine:
    """
    Core trading engine running an asynchronous evaluation loop.
    Tuned for 24/7 autonomous operation.
    """

    def __init__(
        self,
        connection_manager,
        data_reader,
        executor,
        llm_client,
        database,
        risk_manager: RiskManager,
    ) -> None:
        self.conn = connection_manager
        self.reader = data_reader
        self.executor = executor
        self.llm = llm_client
        self.db = database
        self.risk = risk_manager
        self.analyzer = MarketAnalysisEngine()

        self.is_running: bool = False
        self.loop_task: Optional[asyncio.Task] = None
        self.tick_loop_task: Optional[asyncio.Task] = None
        self.protection_task: Optional[asyncio.Task] = None
        self.autonomy_health_task: Optional[asyncio.Task] = None
        self.shadow_outcome_task: Optional[asyncio.Task] = None
        self.market_selection_task: Optional[asyncio.Task] = None
        self._pending_market_selection_account: Optional[Dict[str, Any]] = None
        self._failure_task: Optional[asyncio.Task] = None
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
        self.position_exit_tasks: Dict[str, asyncio.Task] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()
        # Local GPU inference is normally serialized; remote inference can run
        # concurrently within a configured bound so symbols do not queue for
        # an entire candle. Order submission remains separately serialized.
        self._decision_semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        self._provider_probe_lock = asyncio.Lock()
        self._decision_provider_inference_ready: bool = False
        self._consecutive_decision_failures: int = 0
        self._main_loop_heartbeat_monotonic: float = 0.0
        self._broker_poll_heartbeat_monotonic: float = 0.0
        self._protection_heartbeat_monotonic: float = 0.0
        self._protection_state_healthy: bool = True
        self.last_scan_times: Dict[str, datetime] = {}
        self.last_bar_times: Dict[str, str] = {}
        # Ranking can refresh while symbols from the same M5 close are still
        # being evaluated. Keep an immutable per-bar admission record so a
        # refreshed top-three cannot enqueue a fourth late model request.
        self._entry_model_admissions: Dict[str, set[str]] = {}
        self._previous_trend_states: Dict[str, Dict[str, str]] = {}
        self._broker_universe: List[str] = []
        self._broker_scan_symbols: Tuple[str, ...] = ()
        self._broker_universe_cursor = 0
        self._broker_universe_refreshed = 0.0
        self._last_stale_bars: Dict[str, str] = {}
        initial_candidates = self._market_candidates_for_current_market()
        self._selected_symbols: Tuple[str, ...] = tuple(
            initial_candidates[: settings.dynamic_market_max_symbols]
        )
        self._market_selection_initialized: bool = False
        self._last_market_selection_monotonic: float = 0.0
        self._market_rankings: Dict[str, Dict[str, Any]] = {}
        self._forex_context_by_symbol: Dict[str, Dict[str, Any]] = {}
        # Symbol analysis tasks run independently. Keep their global dashboard
        # summary behind one lock so a late stale-data result cannot overwrite
        # another symbol that is actively analyzing or has completed normally.
        self._scan_status_lock = threading.RLock()
        self._active_scan_symbols: Tuple[str, ...] = tuple(
            symbol.upper() for symbol in self._symbols_for_current_market()
        )
        self._symbol_scan_states: Dict[str, str] = {
            symbol: "WAITING" for symbol in self._active_scan_symbols
        }
        self._last_positions_tickets: List[int] = []
        self._last_position_bot_owned: Dict[int, bool] = {}
        # MT5 labels application-submitted closes as EXPERT. Keep the exact
        # strategy cause until broker history confirms the position is gone.
        self._pending_close_reasons: Dict[int, str] = {}
        self._peak_profits: Dict[int, float] = {}
        self._peak_profit_usd: Dict[int, float] = {}
        self._trough_profits: Dict[int, float] = {}
        self._trough_profit_usd: Dict[int, float] = {}
        self._peak_state_loaded: set[int] = set()
        self._peak_persisted_usd: Dict[int, float] = {}
        self._trough_persisted_usd: Dict[int, float] = {}
        self._profit_lock_tickets: set[int] = set()
        self._profit_lock_levels: Dict[int, float] = {}
        self._profit_retention_states: Dict[int, RetentionState] = {}
        self._profit_retention_loaded: set = set()
        self._profit_retention_persisted: Dict[int, RetentionState] = {}
        self._breakeven_tickets: set[int] = set()
        self._protection_failures: Dict[Tuple[int, str], Tuple[str, float]] = {}
        self._initial_risk_pips: Dict[int, float] = {}
        self._initial_risk_usd: Dict[int, float] = {}
        self._adverse_momentum_streaks: Dict[int, int] = {}
        self.last_exit_bar_times: Dict[str, str] = {}
        self._baseline_error_tickets: set[int] = set()
        self._last_reconcile_monotonic: float = 0.0
        self._last_tick_error_log_monotonic: float = 0.0
        self._symbol_point_cache: Dict[str, float] = {}
        self._last_positions_error_log_monotonic: float = 0.0
        self._position_state_healthy: bool = True
        self._position_risk_healthy: bool = True
        self._manual_trade_candidates: Dict[str, Dict[str, Any]] = {}
        # A valid model signal can arrive while a newly started live engine is
        # still disarmed. Remember only the affected symbols so arming causes a
        # fresh analysis of the latest completed candle instead of silently
        # discarding the opportunity or submitting a stale cached order.
        self._signals_waiting_for_rearm: set[str] = set()
        self._active_account_login: Optional[int] = None
        self._active_account_mode: str = "UNKNOWN"
        self._active_account_identity: Optional[Dict[str, Any]] = None
        self._armed_account_identity: Optional[Dict[str, Any]] = None
        self.autonomous_enabled: bool = False
        self._autonomous_account_identity: Optional[Dict[str, Any]] = None
        self._last_autonomy_health_state: str = ""
        self._history_ready: bool = False
        self.entries_armed: bool = bool(settings.allow_new_trades_on_start and settings.dry_run)
        replay_db_path = getattr(database, "db_path", None) or DEFAULT_DB_PATH
        self.replay_logger = TradeReplayLogger(str(replay_db_path))
        history_symbols = list(dict.fromkeys(
            settings.trading_symbols
            + settings.market_candidate_symbols
            + settings.weekend_symbols
        ))
        self.history_reconciler = BrokerHistoryReconciler(
            settings.strategy_magic, history_symbols
        )
        dashboard_state.update_automation(
            entries_armed=self.entries_armed,
            autonomous_enabled=False,
            autonomous_status="DISABLED",
            autonomous_account_suffix="—",
            last_health_check="—",
            engine_heartbeat_utc="",
            broker_poll_heartbeat_utc="",
            decision_provider_inference_ready=False,
            safety_status="PAPER" if settings.dry_run else "LOCKED",
            safety_reason=(
                "Dry-run entries may be armed" if settings.dry_run
                else "Entries follow the active Pepperstone account after confirmation"
            ),
            dry_run=settings.dry_run,
            strategy_magic=settings.strategy_magic,
            config_fingerprint=settings.config_fingerprint,
            provider=settings.llm_provider,
            model=settings.decision_model,
            scan_timeframe=(
                f"M5 entries · {settings.position_exit_review_timeframe} exits · "
                f"{settings.decision_poll_seconds:.1f}s protection"
                if settings.fast_exit_review_enabled
                else "M5 close"
            ),
            confirmation_timeframes="M15 / H1 / H4",
            scan_status="WAITING FOR ENGINE",
            active_symbols=list(self._active_scan_symbols),
            history_healthy=False,
            history_status="NOT RECONCILED",
        )
        logger.info(
            "Loaded effective configuration fingerprint=%s",
            settings.config_fingerprint,
        )
        for symbol in history_symbols:
            dashboard_state.update_symbol_decision(symbol, stage="WAITING")

    @staticmethod
    def _market_candidates_for_current_market() -> List[str]:
        """Return the bounded configured universe eligible for selection."""
        symbols: List[str] = list(
            settings.market_candidate_symbols
            if settings.dynamic_market_selection_enabled
            else settings.trading_symbols
        )
        symbols = list(dict.fromkeys(settings.trading_symbols + symbols))
        if is_weekend():
            if settings.weekend_trading_enabled or settings.crypto_only_on_weekend:
                symbols = list(settings.weekend_symbols)
            else:
                symbols = []
        return list(dict.fromkeys(str(symbol).strip() for symbol in symbols if str(symbol).strip()))

    def _symbols_for_current_market(self) -> List[str]:
        """Return selected markets; discovery continues over the wider pool."""
        candidates = self._candidate_universe()
        if not settings.dynamic_market_selection_enabled:
            return candidates
        selected = [symbol for symbol in self._selected_symbols if symbol in candidates]
        if self._market_selection_initialized:
            return selected
        return selected or candidates[: settings.dynamic_market_max_symbols]

    def _candidate_universe(self) -> List[str]:
        configured = self._market_candidates_for_current_market()
        if not settings.dynamic_market_selection_enabled or not settings.broker_market_discovery_enabled or is_weekend():
            return configured
        return list(dict.fromkeys([*configured, *getattr(self, "_broker_scan_symbols", ())]))

    async def _broker_discovery_batch(self, configured: List[str]) -> List[str]:
        if not settings.broker_market_discovery_enabled or is_weekend():
            return configured
        now = time.monotonic()
        if not getattr(self, "_broker_universe_refreshed", 0.0) or now - self._broker_universe_refreshed >= settings.broker_market_refresh_seconds:
            instruments = await asyncio.to_thread(mt5.symbols_get, group=settings.broker_market_group)
            catalog = instruments
            if instruments is not None and settings.broker_market_group != "*":
                catalog = await asyncio.to_thread(mt5.symbols_get)
            if instruments is None or catalog is None:
                logger.warning("Broker market discovery unavailable; retaining the previous universe")
            else:
                # Register the full catalog so an instrument selected by a
                # different group cannot overwrite an exact broker spelling.
                mt5.register_symbols(catalog)
                unique_contracts = set(tradable_symbols(catalog))
                refreshed = [name.upper() for name in tradable_symbols(instruments) if name in unique_contracts]
                available = set(refreshed)
                previous = getattr(self, "_broker_universe", [])
                retained = [name for name in previous if name in available]
                retained_set = set(retained)
                # Symbol selection changes Market Watch visibility. Preserve
                # catalog order on refresh so it cannot disrupt rotation.
                self._broker_universe = retained + [name for name in refreshed if name not in retained_set]
                if retained != previous:
                    self._broker_universe_cursor = 0
                self._broker_universe_refreshed = now
        batch, self._broker_universe_cursor = discovery_batch(
            configured, getattr(self, "_selected_symbols", ()),
            getattr(self, "_broker_universe", []),
            getattr(self, "_broker_universe_cursor", 0), settings.broker_market_batch_size,
        )
        return batch

    async def _confirmed_entry_analyses(self, symbol: str, m5_frame: Any, confirmations=None) -> Dict[str, Any]:
        """Use the same completed-candle setup construction in discovery and entry."""
        frames = confirmations if confirmations is not None else await asyncio.gather(*(
            self.reader.get_ohlcv(symbol, timeframe, count=max(220, settings.analysis_history_bars))
            for timeframe in ("M15", "H1", "H4")
        ))
        if any(frame is None or frame.empty or frame.attrs.get("is_stale", False) for frame in frames):
            return {}
        analyses = {}
        for timeframe, frame in zip(("M5", "M15", "H1", "H4"), (m5_frame, *frames)):
            analysis = await asyncio.to_thread(self.analyzer.analyze, symbol, timeframe, frame)
            if not analysis:
                return {}
            analyses[timeframe] = copy.deepcopy(analysis)
        # A retest transition is a property of adjacent completed candles,
        # not of when this market happened to enter the active watch list.
        previous = None
        if len(m5_frame) >= 2:
            # Do not turn a weekend/session gap into a fresh retest.
            current_time = m5_frame.iloc[-1]["time"]
            previous_time = m5_frame.iloc[-2]["time"]
            try:
                adjacent = (current_time - previous_time).total_seconds() == 300
            except (TypeError, AttributeError):
                adjacent = False
            if adjacent:
                previous = await asyncio.to_thread(self.analyzer.analyze, symbol, "M5", m5_frame.iloc[:-1])
        previous_state = (previous or {}).get("market_structure", {}).get("trend_state", "NEUTRAL")
        annotate_retest_continuation(analyses["M5"], analyses["M15"], analyses["H1"], {"M5": previous_state})
        annotate_range_reversion(analyses["M5"], analyses["M15"], analyses["H1"], analyses["H4"])
        return analyses

    def _symbols_with_position_priority(
        self, positions: List[Dict[str, Any]]
    ) -> List[str]:
        """Put managed open positions ahead of new-entry market candidates."""
        managed_symbols = [
            str(position.get("symbol", "")).strip().upper()
            for position in positions
            if str(position.get("symbol", "")).strip()
            and (
                position.get("bot_owned")
                or settings.manage_external_positions
            )
        ]
        return list(dict.fromkeys(
            managed_symbols + self._symbols_for_current_market()
        ))

    @staticmethod
    def _scan_issue_summary(states: List[str]) -> List[str]:
        labels = (
            ("STALE", "STALE"),
            ("DATA_UNAVAILABLE", "UNAVAILABLE"),
            ("DATA_ERROR", "DATA ERROR"),
            ("ERROR", "SCAN ERROR"),
            ("LLM_ERROR", "LLM ERROR"),
        )
        return [
            f"{states.count(state)} {label}"
            for state, label in labels
            if states.count(state)
        ]

    def _aggregate_scan_status_locked(self) -> str:
        """Build one global status without discarding symbol-level outcomes."""
        symbols = self._active_scan_symbols
        if not symbols:
            return "NO ACTIVE M5 MARKETS"

        states = {
            symbol: self._symbol_scan_states.get(symbol, "WAITING")
            for symbol in symbols
        }
        llm_active = [symbol for symbol, state in states.items() if state == "LLM_INFERENCE"]
        analyzing = [symbol for symbol, state in states.items() if state == "ANALYZING"]
        issues = self._scan_issue_summary(list(states.values()))

        active_parts: List[str] = []
        if llm_active:
            active_parts.append(f"LLM INFERENCE · {', '.join(llm_active)}")
        if analyzing:
            active_parts.append(f"ANALYZING {', '.join(analyzing)}")
        if active_parts:
            return " · ".join(active_parts + issues)

        llm_errors = [symbol for symbol, state in states.items() if state == "LLM_ERROR"]
        healthy_count = sum(
            state in {"WAITING", "COMPLETE"} for state in states.values()
        )
        if llm_errors:
            parts = [f"LLM ERROR · {', '.join(llm_errors)}"]
            if healthy_count:
                parts.append(f"{healthy_count} READY")
            parts.extend(item for item in issues if not item.endswith("LLM ERROR"))
            return " · ".join(parts)

        if healthy_count:
            if issues:
                return " · ".join(
                    ["MONITORING M5", f"{healthy_count} READY", *issues]
                )
            return "MONITORING TICKS - NEXT ENTRY ON M5 CLOSE"

        if all(state == "STALE" for state in states.values()):
            return f"STALE M5 DATA · {', '.join(symbols)}"
        if all(state == "DATA_UNAVAILABLE" for state in states.values()):
            return f"M5 DATA UNAVAILABLE · {', '.join(symbols)}"
        return " · ".join(["M5 DATA ISSUES", *issues])

    def _set_active_scan_symbols(self, symbols: List[str]) -> None:
        """Change the aggregation scope when weekday/weekend markets rotate."""
        normalized = tuple(
            dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
        )
        with self._scan_status_lock:
            if normalized == self._active_scan_symbols:
                return
            self._active_scan_symbols = normalized
            for symbol in normalized:
                self._symbol_scan_states.setdefault(symbol, "WAITING")
            dashboard_state.update_automation(
                scan_status=self._aggregate_scan_status_locked(),
                active_symbols=list(normalized),
            )

    async def _assess_market_candidate(
        self, symbol: str, account: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a non-executing M5 affordability/rank snapshot."""
        frame = await self.reader.get_ohlcv(
            symbol, "M5", count=max(220, settings.analysis_history_bars)
        )
        if frame is None or frame.empty or bool(frame.attrs.get("is_stale", False)):
            result = {
                "symbol": symbol,
                "status": "DATA UNAVAILABLE",
                "capital_fit": False,
                "reason": "Current completed M5 history is unavailable or stale",
                "selection_score": 0.0,
                "selection_regime": "UNKNOWN",
                "selected": False,
            }
            dashboard_state.update_market_fit(symbol, result)
            return result
        analyses = await self._confirmed_entry_analyses(symbol, frame)
        if not analyses:
            result = {
                "symbol": symbol,
                "status": "DATA ERROR",
                "capital_fit": False,
                "reason": "Completed-candle confirmation history or indicator warm-up is incomplete",
                "selection_score": 0.0,
                "selection_regime": "UNKNOWN",
                "selected": False,
            }
            dashboard_state.update_market_fit(symbol, result)
            return result
        analysis = analyses["M5"]
        capital_fit = await asyncio.to_thread(
            DeterministicTradePlanner.assess_capital_fit,
            symbol,
            analysis,
            account,
        )
        history = await self.db.get_closed_positions(
            symbol=mt5.broker_symbol_name(symbol),
            # Manual terminal/mobile closes are excluded from strategy
            # expectancy. Read extra broker rows so those exclusions do not
            # silently shrink the configured strategy sample.
            limit=max(
                settings.market_performance_lookback,
                settings.market_performance_lookback * 3,
            ),
            account_login=int(account.get("login", 0) or 0),
        )
        capital_fit["performance"] = AdaptiveMarketSelector.summarize_performance(
            history,
            strategy_magic=settings.strategy_magic,
        )
        capital_fit["m5_impulse_atr"] = float(
            analysis.get("indicators", {}).get("candle_return_atr", 0.0) or 0.0
        )
        capital_fit.update(screen_opportunities(analyses, capital_fit, history))
        prefilter_reason = self._entry_prefilter_reason(
            analysis,
            analyses["M15"], analyses["H1"],
        )
        capital_fit["entry_prefilter_reason"] = prefilter_reason
        capital_fit["model_eligible"] = bool(
            capital_fit.get("model_eligible")
            and not prefilter_reason
        )
        capital_fit["opportunity_bar"] = str(analysis.get("timestamp", ""))
        capital_fit["opportunity_status"] = (
            "READY FOR REVIEW" if capital_fit["model_eligible"] else
            "WAITING FOR SETUP" if not capital_fit["actionable_entry_evidence"] else "BLOCKED"
        )
        capital_fit.update(AdaptiveMarketSelector.score(symbol, analysis, capital_fit))
        dashboard_state.update_market_fit(symbol, capital_fit)
        return capital_fit

    async def _refresh_dynamic_market_selection(
        self, account: Dict[str, Any]
    ) -> None:
        """Discover and rank broker-open markets without placing orders."""
        if not settings.dynamic_market_selection_enabled:
            return
        configured_candidates = self._market_candidates_for_current_market()
        candidates = await self._broker_discovery_batch(configured_candidates)
        active_identity = getattr(self, "_active_account_identity", None)
        if active_identity is not None and not self._same_account(account, active_identity):
            return
        if configured_candidates != self._market_candidates_for_current_market():
            return
        self._broker_scan_symbols = tuple(candidates)
        dashboard_state.retain_market_scope(candidates)
        if not candidates:
            self._selected_symbols = tuple()
            self._market_selection_initialized = True
            self._set_active_scan_symbols([])
            return

        semaphore = asyncio.Semaphore(4)

        async def assess(symbol: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self._assess_market_candidate(symbol, account)
                except Exception as exc:
                    logger.warning("Market discovery failed for %s: %s", symbol, exc)
                    result = {
                        "symbol": symbol,
                        "status": "DISCOVERY ERROR",
                        "capital_fit": False,
                        "reason": str(exc),
                        "selection_score": 0.0,
                        "selection_regime": "UNKNOWN",
                        "selected": False,
                    }
                    dashboard_state.update_market_fit(symbol, result)
                    return result

        ranked = await asyncio.gather(*(assess(symbol) for symbol in candidates))
        # Discovery runs independently of the entry lane. Never let a slow
        # refresh from an account or market-session that is no longer active
        # overwrite the current selection.
        active_identity = getattr(self, "_active_account_identity", None)
        if active_identity is not None and not self._same_account(
            account, active_identity
        ):
            return
        if configured_candidates != self._market_candidates_for_current_market():
            return
        self._forex_context_by_symbol = build_currency_context(ranked)
        for item in ranked:
            symbol = str(item.get("symbol", "")).upper()
            item["forex_context"] = dict(
                self._forex_context_by_symbol.get(symbol, {})
            )
            if "selection_adx" in item and "selection_regime" in item:
                item.update(
                    AdaptiveMarketSelector.score(
                        symbol,
                        {
                            "indicators": {
                                "adx_14": item.get("selection_adx", 0.0),
                                "adx_delta": item.get(
                                    "selection_adx_delta", 0.0
                                ),
                            },
                            "market_structure": {
                                "trend_state": item.get(
                                    "selection_regime", "NEUTRAL"
                                ),
                                "structure_events": (
                                    [{"type": "BOS"}]
                                    if item.get("selection_has_structure")
                                    else []
                                ),
                            },
                        },
                        item,
                    )
                )
        selected = AdaptiveMarketSelector.select(
            ranked, settings.dynamic_market_max_symbols
        )
        selected_set = set(selected)
        self._market_rankings = {}
        sorted_ranked = sorted(
            ranked,
            key=lambda row: (bool(row.get("model_eligible")), float(row.get("selection_score", 0.0) or 0.0)),
            reverse=True,
        )
        eligible_rank = 0
        for rank, item in enumerate(
            sorted_ranked,
            start=1,
        ):
            item["selected"] = item.get("symbol") in selected_set
            item["selection_rank"] = (
                rank
                if item.get("broker_open", True)
                else None
            )
            if (
                item.get("selected")
                and item.get("capital_fit")
                and item.get("broker_open", True)
                and item.get("model_eligible")
            ):
                eligible_rank += 1
                item["model_selection_rank"] = eligible_rank
                item["model_selected"] = (
                    eligible_rank <= settings.llm_entry_candidates_per_bar
                )
            else:
                item["model_selection_rank"] = None
                item["model_selected"] = False
            dashboard_state.update_market_fit(str(item.get("symbol", "")), item)
            self._market_rankings[str(item.get("symbol", "")).upper()] = dict(item)

        previous = self._selected_symbols
        self._selected_symbols = tuple(selected)
        self._market_selection_initialized = True
        self._set_active_scan_symbols(selected)
        if previous != self._selected_symbols:
            summary = ", ".join(selected) if selected else "none currently eligible"
            self.log(f"Adaptive market selection: {summary}.")

    def _decision_work_is_active(self) -> bool:
        """Return whether entry/exit analysis currently has queue priority."""
        return any(
            not task.done()
            for task in getattr(self, "analysis_tasks", {}).values()
        ) or any(
            not task.done()
            for task in getattr(self, "position_exit_tasks", {}).values()
        )

    def _start_market_selection_task(
        self, account: Dict[str, Any]
    ) -> bool:
        """Start one background discovery refresh when the decision lane is idle."""
        if not self.is_running:
            return False
        active_identity = getattr(self, "_active_account_identity", None)
        if active_identity is not None and not self._same_account(
            account, active_identity
        ):
            return False
        current = getattr(self, "market_selection_task", None)
        if current is not None and not current.done():
            return False
        self._pending_market_selection_account = None
        self._last_market_selection_monotonic = (
            asyncio.get_running_loop().time()
        )
        task = asyncio.create_task(
            self._refresh_dynamic_market_selection(dict(account))
        )
        self.market_selection_task = task
        task.add_done_callback(self._market_selection_finished)
        return True

    def _request_market_selection_refresh(
        self, account: Dict[str, Any]
    ) -> None:
        """Run discovery only after fresh entry and exit evaluations drain."""
        if not settings.dynamic_market_selection_enabled:
            return
        now = asyncio.get_running_loop().time()
        last_refresh = float(
            getattr(self, "_last_market_selection_monotonic", 0.0) or 0.0
        )
        initialized = bool(
            getattr(self, "_market_selection_initialized", False)
        )
        due = (
            (not initialized and last_refresh <= 0.0)
            or now - last_refresh >= settings.market_selection_refresh_seconds
        )
        if not due:
            return
        current = getattr(self, "market_selection_task", None)
        if current is not None and not current.done():
            return
        if self._decision_work_is_active():
            self._pending_market_selection_account = dict(account)
            return
        self._start_market_selection_task(account)

    def _start_pending_market_selection_if_idle(self) -> None:
        pending = getattr(self, "_pending_market_selection_account", None)
        if not pending or self._decision_work_is_active():
            return
        if not self._start_market_selection_task(dict(pending)):
            active_identity = getattr(
                self, "_active_account_identity", None
            )
            if active_identity is not None and not self._same_account(
                pending, active_identity
            ):
                self._pending_market_selection_account = None

    def _set_symbol_scan_state(
        self,
        symbol: str,
        state: str,
        *,
        last_scan: Optional[str] = None,
    ) -> str:
        """Atomically publish a symbol result and its aggregate dashboard status."""
        normalized_symbol = str(symbol).strip().upper()
        normalized_state = str(state).strip().upper()
        with self._scan_status_lock:
            self._symbol_scan_states[normalized_symbol] = normalized_state
            status = self._aggregate_scan_status_locked()
            values: Dict[str, Any] = {"scan_status": status}
            if last_scan is not None:
                values["last_scan"] = last_scan
            # Publish while holding the aggregation lock. This prevents an older
            # computed summary from being written after a newer symbol result.
            dashboard_state.update_automation(**values)
            return status

    def log(self, message: str, level: str = "INFO") -> None:
        """Helper that logs to system logger and appends to dashboard UI logs."""
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
        
        # Strip timezone / simplify format for UI logs
        time_str = datetime.now().strftime("%H:%M:%S")
        dashboard_state.append_log(f"[{time_str}] [{level}] {message}")

    async def _log_replay_attempt(self, **values: Any) -> None:
        """Persist an audit attempt without blocking the trading event loop."""
        await asyncio.to_thread(
            self.replay_logger.log_replay_attempt,
            **values,
        )

    async def _update_replay_outcome(
        self,
        ticket: int,
        pnl: float,
        **values: Any,
    ) -> None:
        """Persist a broker outcome without stalling quotes or protections."""
        await asyncio.to_thread(
            self.replay_logger.update_replay_outcome,
            ticket,
            pnl,
            **values,
        )

    async def _remember_strategy_close_reason(
        self,
        ticket: int,
        reason: str,
    ) -> None:
        """Keep an application-led close cause across broker reconciliation."""
        ticket = int(ticket)
        normalized = str(reason or "").strip().upper()
        if ticket <= 0 or not normalized:
            return
        if not hasattr(self, "_pending_close_reasons"):
            self._pending_close_reasons = {}
        self._pending_close_reasons[ticket] = normalized
        account = dict(getattr(self, "_active_account_identity", None) or {})
        account_login = int(account.get("login", 0) or 0)
        writer = getattr(self.db, "set_strategy_close_reason", None)
        if account_login > 0 and writer is not None:
            await writer(account_login, ticket, normalized)

    async def _record_shadow_candidate(
        self,
        *,
        symbol: str,
        action: str,
        completed_bar: str,
        plan: Optional[TradePlan],
        stage: str,
        reason: str,
    ) -> None:
        """Persist a valid-but-rejected signal without affecting execution."""
        if not settings.shadow_outcomes_enabled or plan is None or not plan.valid:
            return
        await asyncio.to_thread(
            self.replay_logger.log_shadow_candidate,
            symbol=symbol,
            action=action,
            candle_time=completed_bar,
            entry=plan.entry,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
            rejection_stage=stage,
            rejection_reason=reason,
            horizon_minutes=settings.shadow_outcome_horizon_minutes,
        )

    async def _refresh_shadow_outcomes(self) -> None:
        pending = (
            await asyncio.to_thread(
                self.replay_logger.get_pending_shadow_candidates, 100
            )
            if settings.shadow_outcomes_enabled
            else []
        )
        if pending:
            by_symbol: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in pending:
                by_symbol.setdefault(str(candidate.get("symbol", "")).upper(), []).append(
                    candidate
                )
            for symbol, candidates in by_symbol.items():
                if not symbol:
                    continue
                bars = await self.reader.get_ohlcv(symbol, "M1", count=180)
                if bars is None or bars.empty:
                    continue
                for candidate in candidates:
                    resolution = evaluate_shadow_candidate(candidate, bars)
                    if resolution is None:
                        continue
                    await asyncio.to_thread(
                        self.replay_logger.resolve_shadow_candidate,
                        int(candidate["id"]),
                        status=resolution.status,
                        resolved_at_utc=resolution.resolved_at_utc,
                        exit_price=resolution.exit_price,
                        outcome_r=resolution.outcome_r,
                        mfe_r=resolution.mfe_r,
                        mae_r=resolution.mae_r,
                    )
        if getattr(settings, "exit_counterfactual_enabled", True):
            exit_candidates = await asyncio.to_thread(
                self.replay_logger.get_pending_exit_candidates, 100
            )
            if exit_candidates:
                by_symbol: Dict[str, List[Dict[str, Any]]] = {}
                for candidate in exit_candidates:
                    by_symbol.setdefault(
                        str(candidate.get("symbol", "")).upper(), []
                    ).append(candidate)
                for symbol, candidates in by_symbol.items():
                    if not symbol:
                        continue
                    max_horizon = max(
                        float(row.get("horizon_minutes", 60.0) or 60.0)
                        for row in candidates
                    )
                    bars = await self.reader.get_ohlcv(
                        symbol, "M1", count=max(180, int(max_horizon) + 30)
                    )
                    if bars is None or bars.empty:
                        continue
                    for candidate in candidates:
                        resolution = evaluate_exit_counterfactual(
                            candidate, bars
                        )
                        if resolution is None:
                            continue
                        await asyncio.to_thread(
                            self.replay_logger.resolve_exit_candidate,
                            int(candidate["id"]),
                            status=resolution.status,
                            resolved_at_utc=resolution.resolved_at_utc,
                            exit_price=resolution.exit_price,
                            outcome_r=resolution.outcome_r,
                            delta_vs_realized_r=(
                                resolution.delta_vs_realized_r
                            ),
                            post_exit_mfe_r=resolution.post_exit_mfe_r,
                            post_exit_mae_r=resolution.post_exit_mae_r,
                        )
        summary = await asyncio.to_thread(self.replay_logger.shadow_summary)
        exit_summary = await asyncio.to_thread(
            self.replay_logger.exit_counterfactual_summary
        )
        dashboard_state.update_shadow(
            enabled=settings.shadow_outcomes_enabled,
            exit_enabled=getattr(
                settings, "exit_counterfactual_enabled", True
            ),
            **summary,
            **exit_summary,
        )

    async def _shadow_outcome_loop(self) -> None:
        """Evaluate rejected signals off the execution path."""
        while self.is_running:
            cycle_started = time.monotonic()
            try:
                if (
                    settings.shadow_outcomes_enabled
                    or getattr(settings, "exit_counterfactual_enabled", True)
                ):
                    await self._refresh_shadow_outcomes()
                else:
                    dashboard_state.update_shadow(enabled=False)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Shadow outcome refresh failed: %s", exc)
            elapsed = time.monotonic() - cycle_started
            await asyncio.sleep(
                max(1.0, settings.shadow_outcome_poll_seconds - elapsed)
            )

    @staticmethod
    def _reported_inference_latency(
        telemetry: Dict[str, Any], total_elapsed: float
    ) -> float:
        """Return model execution time without semaphore queue delay."""
        if telemetry.get("trace_id"):
            try:
                return max(0.0, float(telemetry.get("latency_seconds", 0.0) or 0.0))
            except (TypeError, ValueError, OverflowError):
                pass
        return max(0.0, float(total_elapsed))

    def _touch_engine_heartbeat(self, *, broker_poll: bool = False) -> None:
        """Publish decision-loop progress that an external watchdog can verify."""
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc).isoformat()
        self._main_loop_heartbeat_monotonic = now_monotonic
        values: Dict[str, Any] = {"engine_heartbeat_utc": now_utc}
        if broker_poll:
            self._broker_poll_heartbeat_monotonic = now_monotonic
            values["broker_poll_heartbeat_utc"] = now_utc
        dashboard_state.update_automation(**values)

    def _touch_protection_heartbeat(self, *, healthy: bool = True) -> None:
        """Record progress of the independent broker-protection supervisor."""
        self._protection_heartbeat_monotonic = time.monotonic()
        self._protection_state_healthy = bool(healthy)

    def _protection_progress_health(self) -> Tuple[bool, str]:
        running = getattr(self, "is_running", None)
        heartbeat = getattr(self, "_protection_heartbeat_monotonic", None)
        if running is None and heartbeat is None:
            return True, "Protection heartbeat unavailable on legacy instance"
        if not running:
            return False, "Protection supervisor is stopped"
        healthy = bool(getattr(self, "_protection_state_healthy", True))
        if heartbeat is None:
            return False, "Protection heartbeat is unavailable"
        if heartbeat <= 0:
            return False, "Waiting for the first protection poll"
        age = time.monotonic() - heartbeat
        limit = settings.engine_heartbeat_stale_seconds
        if not healthy:
            return False, "Latest broker-protection poll failed"
        if age > limit:
            return (
                False,
                f"Protection supervisor heartbeat is stale "
                f"({age:.1f}s > {limit:.1f}s)",
            )
        return True, f"Advancing · {age:.1f}s ago"

    def _engine_progress_health(self) -> Tuple[bool, str]:
        """Return whether the decision and broker-poll loop is advancing."""
        if not self.is_running:
            return False, "Decision engine is stopped"
        main_heartbeat = getattr(
            self, "_main_loop_heartbeat_monotonic", None
        )
        broker_heartbeat = getattr(
            self, "_broker_poll_heartbeat_monotonic", None
        )
        # Legacy test doubles created with __new__ do not own heartbeat state.
        if main_heartbeat is None or broker_heartbeat is None:
            return True, "Progress heartbeat unavailable on legacy instance"
        if main_heartbeat <= 0 or broker_heartbeat <= 0:
            return False, "Waiting for the first broker-position poll"
        age = max(
            time.monotonic() - main_heartbeat,
            time.monotonic() - broker_heartbeat,
        )
        limit = settings.engine_heartbeat_stale_seconds
        if age > limit:
            return (
                False,
                f"Broker-position loop heartbeat is stale "
                f"({age:.1f}s > {limit:.1f}s)",
            )
        return True, f"Advancing · {age:.1f}s ago"

    async def _ensure_decision_provider_ready(
        self, health: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Require a real local completion before entries can be armed."""
        health = dict(health or await self.llm.health_check())
        metadata_ready = bool(
            health.get("online") and health.get("available")
        )
        if not metadata_ready:
            self._decision_provider_inference_ready = False
            dashboard_state.update_automation(
                llm_online=False,
                decision_provider_inference_ready=False,
            )
            return False, health

        provider = str(
            health.get("provider", getattr(self.llm, "provider_name", ""))
        ).lower()
        if (
            provider != "local"
            or not hasattr(self.llm, "readiness_probe")
        ):
            self._decision_provider_inference_ready = True
            dashboard_state.update_automation(
                llm_online=True,
                decision_provider_inference_ready=True,
            )
            return True, health

        if self._decision_provider_inference_ready:
            return True, health

        probe_lock = getattr(self, "_provider_probe_lock", None)

        async def run_probe() -> Dict[str, Any]:
            async with self._decision_semaphore:
                return await self.llm.readiness_probe()

        try:
            if probe_lock is None:
                probe = await run_probe()
            else:
                async with probe_lock:
                    if self._decision_provider_inference_ready:
                        return True, health
                    probe = await run_probe()
        except Exception as exc:
            probe = {
                **health,
                "inference_ready": False,
                "error": str(exc),
            }
        ready = bool(
            probe.get("online")
            and probe.get("available")
            and probe.get("inference_ready")
        )
        self._decision_provider_inference_ready = ready
        if ready:
            self._consecutive_decision_failures = 0
        dashboard_state.update_automation(
            llm_online=ready,
            decision_provider_inference_ready=ready,
            provider=str(probe.get("provider", provider)),
            model=str(
                probe.get("selected_model", settings.decision_model)
            ),
        )
        return ready, probe

    def _record_decision_provider_result(self, success: bool) -> None:
        """Open a provider circuit after repeated failed completions."""
        if success:
            self._consecutive_decision_failures = 0
            self._decision_provider_inference_ready = True
            dashboard_state.update_automation(
                llm_online=True,
                decision_provider_inference_ready=True,
            )
            return
        self._consecutive_decision_failures = (
            getattr(self, "_consecutive_decision_failures", 0) + 1
        )
        if (
            self._consecutive_decision_failures
            < settings.decision_provider_failure_threshold
        ):
            return
        self._decision_provider_inference_ready = False
        dashboard_state.update_automation(
            llm_online=False,
            decision_provider_inference_ready=False,
        )
        if self.entries_armed:
            self.disarm_entries(
                "Decision provider failed repeatedly; waiting for a "
                "successful inference readiness probe"
            )

    @staticmethod
    def _entry_prefilter_reason(
        m5_analysis: Dict[str, Any],
        m15_analysis: Optional[Dict[str, Any]] = None,
        h1_analysis: Optional[Dict[str, Any]] = None,
        *,
        allow_pending_range: bool = False,
        allow_provisional_aligned_decline: bool = False,
    ) -> str:
        """Skip model work when deterministic entry gates cannot pass."""
        indicators = (m5_analysis or {}).get("indicators", {})
        range_setup = (
            (m5_analysis or {}).get("market_structure", {}).get(
                "range_reversion", {}
            )
            or {}
        )
        range_can_continue = bool(
            range_setup.get("eligible")
            or (allow_pending_range and range_setup.get("candidate"))
        )
        try:
            adx = float(indicators.get("adx_14", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            adx = 0.0
        if adx < settings.entry_min_adx and not range_can_continue:
            return (
                "REJECTED [ADX Prefilter]: Weak ranging market "
                f"(ADX {adx:.2f} < {settings.entry_min_adx:.2f}). "
                "Model inference skipped because an entry cannot pass risk."
            )
        if settings.entry_require_adx_rising and not range_can_continue:
            try:
                adx_delta = float(indicators.get("adx_delta"))
            except (TypeError, ValueError, OverflowError):
                adx_delta = math.nan
            decline_limit = -settings.entry_adx_decline_tolerance
            if math.isfinite(adx_delta) and adx_delta < decline_limit:
                aligned_exception = aligned_structure_allows_adx_decline(
                    m5_analysis,
                    m15_analysis,
                    h1_analysis,
                    max_decline=(
                        settings.entry_aligned_adx_decline_tolerance
                    ),
                    min_m5_adx=settings.breakout_min_adx,
                    min_m15_adx=settings.confirmation_min_adx,
                    require_confirmation_alignment=(
                        not allow_provisional_aligned_decline
                    ),
                )
                if not aligned_exception:
                    return (
                        "REJECTED [ADX Prefilter]: M5 ADX is falling "
                        f"({adx_delta:+.2f}); allowed noise is "
                        f"{settings.entry_adx_decline_tolerance:.2f}. A "
                        "bounded exception requires fresh M5 structure with "
                        "strong, aligned M15/H1 confirmation."
                    )
        return ""

    @staticmethod
    def _entry_decision_fresh(
        analysis_started_monotonic: float,
    ) -> Tuple[bool, str]:
        age = max(0.0, time.monotonic() - analysis_started_monotonic)
        limit = settings.max_entry_decision_age_seconds
        if age > limit:
            return (
                False,
                f"Decision age {age:.1f}s exceeds the {limit:.1f}s "
                "entry deadline",
            )
        return True, ""

    def _reserve_entry_model_slot(
        self,
        symbol: str,
        completed_bar: str,
    ) -> Tuple[bool, str]:
        """Reserve one bounded local-model request for a completed M5 bar."""
        bar_key = str(completed_bar or "UNKNOWN")
        normalized_symbol = str(symbol or "").upper()
        admissions = self._entry_model_admissions.setdefault(bar_key, set())
        if normalized_symbol in admissions:
            return True, ""

        limit = max(1, int(settings.llm_entry_candidates_per_bar))
        if len(admissions) >= limit:
            admitted = ", ".join(sorted(admissions)) or "none"
            return False, (
                "Deterministic scan completed, but the bounded local-model "
                f"lane already admitted {limit} candidate"
                f"{'s' if limit != 1 else ''} for this M5 close "
                f"({admitted}). Waiting for the next completed candle avoids "
                "a late queued decision."
            )

        admissions.add(normalized_symbol)
        # Retain a small idempotency window for retries without growing state
        # throughout an unattended session.
        while len(self._entry_model_admissions) > 8:
            oldest = next(iter(self._entry_model_admissions))
            if oldest == bar_key:
                break
            self._entry_model_admissions.pop(oldest, None)
        return True, ""

    @staticmethod
    def _entry_inference_budget_seconds(
        analysis_started_monotonic: float,
        completed_bar_age_seconds: float,
    ) -> float:
        """Return time left before a new-entry inference becomes unusable.

        The budget is bounded by both the end-to-end decision deadline and the
        completed-candle execution window. It therefore includes time spent
        waiting for the shared local-model semaphore.
        """
        elapsed = max(
            0.0, time.monotonic() - analysis_started_monotonic
        )
        decision_remaining = (
            settings.max_entry_decision_age_seconds - elapsed
        )
        bar_remaining = (
            settings.max_entry_bar_age_seconds
            - max(0.0, completed_bar_age_seconds)
            - elapsed
        )
        return max(0.0, min(decision_remaining, bar_remaining))

    @staticmethod
    def _deterministic_entry_fast_path(
        decision_context: Dict[str, Any],
        entry_contract: Tuple[str, ...],
    ) -> Optional[Dict[str, Any]]:
        """Return a bounded no-network decision for an unambiguous setup.

        This is intentionally a strict subset of the normal entry flow. It is
        available only when completed-M5 evidence permits exactly one action,
        M5/M15/H1/H4 all point in that direction, and a directional M5
        structure trigger exists. The result does not place an order directly;
        it continues through every existing validation and risk gate.
        """
        if (
            not settings.deterministic_entry_fast_path_enabled
            or decision_context.get("has_open_position")
            or len(entry_contract) != 1
        ):
            return None

        action = str(entry_contract[0]).upper()
        expected = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(action)
        if expected is None:
            return None

        analyses = decision_context.get("analyses") or {}
        if not all(
            analyses.get(timeframe)
            for timeframe in ("M5", "M15", "H1", "H4")
        ):
            return None
        evidence_ids = build_evidence_ids(analyses)
        if not has_directional_trigger(
            evidence_ids,
            action,
            timeframes=("M5",),
        ):
            return None

        def direction(timeframe: str) -> str:
            structure = analyses[timeframe].get("market_structure") or {}
            value = (
                structure.get("trend_state_direction")
                or structure.get("trend")
            )
            return str(value or "NEUTRAL").upper()

        if any(
            direction(timeframe) != expected
            for timeframe in ("M5", "M15", "H1", "H4")
        ):
            return None

        decision = DeterministicDecisionProvider._decide(decision_context)
        if str(decision.get("action", "HOLD")).upper() != action:
            return None
        try:
            confidence = float(decision.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            not math.isfinite(confidence)
            or confidence < settings.confidence_threshold
        ):
            return None

        result = dict(decision)
        result["_decision_path"] = "DETERMINISTIC_FAST_PATH"
        return result

    def _defer_disarmed_signal(self, symbol: str) -> None:
        if not hasattr(self, "_signals_waiting_for_rearm"):
            self._signals_waiting_for_rearm = set()
        self._signals_waiting_for_rearm.add(str(symbol).upper())

    def _release_deferred_signals_for_rescan(self) -> Tuple[str, ...]:
        """Make disarmed signals eligible for a fresh post-arm evaluation."""
        symbols = tuple(sorted(getattr(self, "_signals_waiting_for_rearm", set())))
        for symbol in symbols:
            self.last_bar_times.pop(symbol, None)
            self.last_scan_times.pop(symbol, None)
        self._signals_waiting_for_rearm.clear()
        return symbols

    def _core_task_finished(self, name: str, task: asyncio.Task) -> None:
        """Escalate an unexpected core-loop exit instead of leaving a false RUNNING state."""
        if not self.is_running:
            return
        if self._failure_task is not None and not self._failure_task.done():
            return
        if task.cancelled():
            detail = "was cancelled unexpectedly"
        else:
            try:
                error = task.exception()
            except asyncio.CancelledError:
                error = None
            detail = f"failed: {error}" if error is not None else "stopped unexpectedly"
        reason = f"{name} loop {detail}"
        self.log(reason, "ERROR")
        self.entries_armed = False
        self._armed_account_identity = None
        self.executor.begin_shutdown()
        dashboard_state.engine_running = False
        dashboard_state.update_automation(
            entries_armed=False,
            safety_status="PAPER" if settings.dry_run else "LOCKED",
            safety_reason=reason,
            scan_status=f"ENGINE ERROR · {name.upper()} LOOP",
        )
        for analysis_task in list(self.analysis_tasks.values()):
            analysis_task.cancel()
        for exit_task in list(getattr(self, "position_exit_tasks", {}).values()):
            exit_task.cancel()
        selection_task = getattr(self, "market_selection_task", None)
        if selection_task is not None and not selection_task.done():
            selection_task.cancel()
        for sibling in (
            self.loop_task,
            self.tick_loop_task,
            getattr(self, "protection_task", None),
            getattr(self, "shadow_outcome_task", None),
        ):
            if sibling is not None and sibling is not task and not sibling.done():
                sibling.cancel()
        autonomy_task = getattr(self, "autonomy_health_task", None)
        if autonomy_task is not None and not autonomy_task.done():
            autonomy_task.cancel()
        self._failure_task = asyncio.create_task(
            self._shutdown_after_core_failure(name, reason)
        )

    async def _shutdown_after_core_failure(self, name: str, reason: str) -> None:
        async with self._lifecycle_lock:
            if self.is_running:
                await self._stop_locked()
            dashboard_state.update_automation(
                entries_armed=False,
                safety_status="PAPER" if settings.dry_run else "LOCKED",
                safety_reason=reason,
                scan_status=f"ENGINE ERROR · {name.upper()} LOOP",
            )
            self._update_execution_readiness(
                None, override=("ENGINE_LOOP_FAILED", reason)
            )

    @staticmethod
    def _account_identity(account: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "login": int(account.get("login", 0) or 0),
            "server": str(account.get("server", "") or ""),
            "company": str(account.get("company", "") or ""),
            "trade_mode": int(account.get("trade_mode", -1)),
            "trade_mode_name": str(account.get("trade_mode_name", "UNKNOWN")),
        }

    @classmethod
    def _account_scope(cls, account: Dict[str, Any]) -> str:
        """Stable, credential-free identity for account-scoped persistence."""
        identity = cls._account_identity(account)
        return "|".join(str(identity[key]) for key in (
            "company", "server", "login", "trade_mode"
        ))

    @classmethod
    def _autonomous_cache_key(cls, account: Dict[str, Any]) -> str:
        return f"autonomous_authorization:{cls._account_scope(account)}"

    @classmethod
    def _position_peak_cache_key(
        cls, account: Dict[str, Any], ticket: int
    ) -> str:
        return f"position_peak:{cls._account_scope(account)}:{int(ticket)}"

    async def _restore_and_update_position_peak(
        self,
        account: Dict[str, Any],
        *,
        ticket: int,
        profit_pips: float,
        observed_profit_usd: float,
    ) -> Tuple[float, float]:
        """Restore and persist broker-reported MFE and MAE.

        Writes occur only when an excursion moves by the configured delta, so
        high-frequency position monitoring does not turn into high-frequency
        SQLite I/O.
        """
        ticket = int(ticket)
        for name in (
            "_peak_profits",
            "_peak_profit_usd",
            "_trough_profits",
            "_trough_profit_usd",
            "_peak_persisted_usd",
            "_trough_persisted_usd",
        ):
            if not hasattr(self, name):
                setattr(self, name, {})
        if not hasattr(self, "_peak_state_loaded"):
            self._peak_state_loaded = set()
        if ticket not in self._peak_state_loaded:
            stored = await self.db.get_cache(
                self._position_peak_cache_key(account, ticket)
            )
            if isinstance(stored, dict):
                try:
                    stored_pips = max(
                        0.0, float(stored.get("peak_profit_pips", 0.0) or 0.0)
                    )
                    stored_usd = max(
                        0.0,
                        float(
                            stored.get(
                                "peak_profit_usd",
                                stored.get("peak_net_profit_usd", 0.0),
                            )
                            or 0.0
                        ),
                    )
                    stored_trough_pips = min(
                        0.0,
                        float(stored.get("trough_profit_pips", 0.0) or 0.0),
                    )
                    stored_trough_usd = min(
                        0.0,
                        float(stored.get("trough_profit_usd", 0.0) or 0.0),
                    )
                    if math.isfinite(stored_pips):
                        self._peak_profits[ticket] = max(
                            self._peak_profits.get(ticket, 0.0), stored_pips
                        )
                    if math.isfinite(stored_usd):
                        self._peak_profit_usd[ticket] = max(
                            self._peak_profit_usd.get(ticket, 0.0), stored_usd
                        )
                        self._peak_persisted_usd[ticket] = stored_usd
                    if math.isfinite(stored_trough_pips):
                        self._trough_profits[ticket] = min(
                            self._trough_profits.get(ticket, 0.0),
                            stored_trough_pips,
                        )
                    if math.isfinite(stored_trough_usd):
                        self._trough_profit_usd[ticket] = min(
                            self._trough_profit_usd.get(ticket, 0.0),
                            stored_trough_usd,
                        )
                        self._trough_persisted_usd[ticket] = stored_trough_usd
                except (TypeError, ValueError, OverflowError):
                    self.log(
                        f"Ignored invalid persisted profit peak for ticket {ticket}.",
                        "WARNING",
                    )
            self._peak_state_loaded.add(ticket)

        peak_pips = max(
            self._peak_profits.get(ticket, 0.0),
            float(profit_pips),
            0.0,
        )
        peak_usd = max(
            self._peak_profit_usd.get(ticket, 0.0),
            float(observed_profit_usd),
            0.0,
        )
        self._peak_profits[ticket] = peak_pips
        self._peak_profit_usd[ticket] = peak_usd
        trough_pips = min(
            self._trough_profits.get(ticket, 0.0),
            float(profit_pips),
            0.0,
        )
        trough_usd = min(
            self._trough_profit_usd.get(ticket, 0.0),
            float(observed_profit_usd),
            0.0,
        )
        self._trough_profits[ticket] = trough_pips
        self._trough_profit_usd[ticket] = trough_usd

        persisted_peak = self._peak_persisted_usd.get(ticket, 0.0)
        persisted_trough = self._trough_persisted_usd.get(ticket, 0.0)
        if (
            ticket not in self._peak_persisted_usd
            or peak_usd - persisted_peak
            >= settings.profit_peak_persist_delta_usd - 1e-9
            or persisted_trough - trough_usd
            >= settings.profit_peak_persist_delta_usd - 1e-9
        ):
            observed_at = datetime.now(timezone.utc).isoformat()
            stored_ok = await self.db.set_cache(
                self._position_peak_cache_key(account, ticket),
                {
                    "peak_profit_pips": round(peak_pips, 5),
                    "peak_profit_usd": round(peak_usd, 5),
                    "trough_profit_pips": round(trough_pips, 5),
                    "trough_profit_usd": round(trough_usd, 5),
                    "updated_at_utc": observed_at,
                },
            )
            if stored_ok:
                self._peak_persisted_usd[ticket] = peak_usd
                self._trough_persisted_usd[ticket] = trough_usd
                try:
                    await self.db.update_position_excursion(
                        self._account_scope(account),
                        ticket,
                        profit_pips=profit_pips,
                        profit_usd=observed_profit_usd,
                        observed_at=observed_at,
                    )
                except Exception as exc:
                    self.log(
                        f"Could not persist position excursion for ticket "
                        f"{ticket}: {exc}",
                        "WARNING",
                    )
        return peak_pips, peak_usd

    def _forget_position_runtime_state(self, ticket: int) -> None:
        """Drop only in-memory state after broker-confirmed position closure."""
        ticket = int(ticket)
        for name in (
            "_peak_profits",
            "_peak_profit_usd",
            "_trough_profits",
            "_trough_profit_usd",
            "_peak_persisted_usd",
            "_trough_persisted_usd",
            "_initial_risk_pips",
            "_initial_risk_usd",
            "_adverse_momentum_streaks",
            "_profit_lock_levels",
            "_profit_retention_states",
            "_profit_retention_persisted",
        ):
            getattr(self, name, {}).pop(ticket, None)
        for name in (
            "_peak_state_loaded",
            "_profit_lock_tickets",
            "_profit_retention_loaded",
            "_breakeven_tickets",
            "_baseline_error_tickets",
        ):
            getattr(self, name, set()).discard(ticket)
        failures = getattr(self, "_protection_failures", {})
        for key in [key for key in failures if key[0] == ticket]:
            failures.pop(key, None)

    def _log_protection_failure(
        self, ticket: int, symbol: str, stage: str, error: Any
    ) -> None:
        """Publish protection failures without flooding the live event log."""
        if not hasattr(self, "_protection_failures"):
            self._protection_failures = {}
        detail = str(error or "unknown broker error")
        key = (int(ticket), str(stage))
        now_mono = time.monotonic()
        previous_detail, previous_at = self._protection_failures.get(
            key, ("", 0.0)
        )
        if detail != previous_detail or now_mono - previous_at >= 30.0:
            self.log(
                f"{stage} failed for ticket {ticket} ({symbol}): {detail}. "
                "Protection remains pending and will be retried.",
                "ERROR",
            )
            self._protection_failures[key] = (detail, now_mono)

    @staticmethod
    def _daily_loss_reset_cache_key(account: Dict[str, Any]) -> str:
        return (
            f"daily_loss_reset:{account.get('server', 'unknown')}:"
            f"{account.get('login', 'unknown')}:{account.get('trade_mode', 'unknown')}"
        )

    @staticmethod
    def _loss_cooldown_reset_cache_key(account: Dict[str, Any]) -> str:
        return (
            f"loss_cooldown_reset:{account.get('server', 'unknown')}:"
            f"{account.get('login', 'unknown')}:{account.get('trade_mode', 'unknown')}"
        )

    @staticmethod
    def _loss_streak_reset_cache_key(account: Dict[str, Any]) -> str:
        return (
            f"loss_streak_reset:{account.get('server', 'unknown')}:"
            f"{account.get('login', 'unknown')}:{account.get('trade_mode', 'unknown')}"
        )

    async def _restore_daily_loss_reset(self, account: Dict[str, Any]) -> None:
        marker = await self.db.get_cache(self._daily_loss_reset_cache_key(account))
        if marker:
            self.risk.restore_daily_loss_reset(marker)

    async def _restore_loss_cooldown_reset(self, account: Dict[str, Any]) -> None:
        marker = await self.db.get_cache(self._loss_cooldown_reset_cache_key(account))
        if marker:
            self.risk.restore_loss_cooldown_reset(marker)

    async def _restore_loss_streak_reset(self, account: Dict[str, Any]) -> None:
        marker = await self.db.get_cache(self._loss_streak_reset_cache_key(account))
        if marker:
            self.risk.restore_losing_streak_reset(marker)

    def _offer_manual_trade_candidate(
        self,
        symbol: str,
        candidate: Dict[str, Any],
        rejection_reason: str,
    ) -> None:
        stored = dict(candidate)
        stored["original_rejection"] = str(rejection_reason)
        if not hasattr(self, "_manual_trade_candidates"):
            self._manual_trade_candidates = {}
        self._manual_trade_candidates[symbol.upper()] = stored
        dashboard_state.update_symbol_decision(
            symbol, manual_override_available=True
        )

    def _clear_manual_trade_candidate(self, symbol: str) -> None:
        getattr(self, "_manual_trade_candidates", {}).pop(symbol.upper(), None)
        dashboard_state.update_symbol_decision(
            symbol, manual_override_available=False
        )

    async def _persist_initial_risk(
        self,
        *,
        account: Dict[str, Any],
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        initial_sl: float,
        volume: float,
        info: Any,
    ) -> Tuple[float, bool]:
        """Insert-once risk baseline and return the canonical persisted R."""
        one_pip = pip_size(info)
        values = (entry_price, initial_sl, volume, one_pip)
        if (
            ticket <= 0
            or not all(math.isfinite(float(value)) for value in values)
            or entry_price <= 0
            or initial_sl <= 0
            or volume <= 0
            or one_pip <= 0
        ):
            return 0.0, False
        if direction == "BUY" and initial_sl >= entry_price:
            return 0.0, False
        if direction == "SELL" and initial_sl <= entry_price:
            return 0.0, False
        initial_risk_pips = abs(float(entry_price) - float(initial_sl)) / one_pip
        if not math.isfinite(initial_risk_pips) or initial_risk_pips <= 0:
            return 0.0, False
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        try:
            initial_risk_result = await asyncio.to_thread(
                mt5.order_calc_profit,
                order_type,
                symbol,
                float(volume),
                float(entry_price),
                float(initial_sl),
            )
            initial_risk_usd = downside_risk_usd(initial_risk_result)
        except Exception as exc:
            if int(ticket) not in self._baseline_error_tickets:
                self.log(
                    f"Could not calculate immutable initial risk for ticket {ticket}: {exc}",
                    "ERROR",
                )
                self._baseline_error_tickets.add(int(ticket))
            return initial_risk_pips, False
        if initial_risk_usd <= 0:
            if int(ticket) not in self._baseline_error_tickets:
                self.log(
                    f"Immutable initial risk for ticket {ticket} is not a downside loss",
                    "ERROR",
                )
                self._baseline_error_tickets.add(int(ticket))
            return initial_risk_pips, False
        try:
            row = await self.db.set_position_risk_baseline(
                self._account_scope(account),
                {
                    "ticket": int(ticket),
                    "symbol": str(symbol),
                    "direction": str(direction),
                    "entry_price": float(entry_price),
                    "initial_sl": float(initial_sl),
                    "initial_risk_pips": initial_risk_pips,
                    "initial_risk_usd": initial_risk_usd,
                    "initial_volume": float(volume),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            canonical = float(row.get("initial_risk_pips", 0.0) or 0.0)
            canonical_usd = float(
                row.get("initial_risk_usd", initial_risk_usd)
                or initial_risk_usd
            )
            if not math.isfinite(canonical) or canonical <= 0:
                raise ValueError("Persisted initial risk is invalid")
            if not math.isfinite(canonical_usd) or canonical_usd <= 0:
                raise ValueError("Persisted initial dollar risk is invalid")
            self._initial_risk_pips[int(ticket)] = canonical
            if not hasattr(self, "_initial_risk_usd"):
                self._initial_risk_usd = {}
            self._initial_risk_usd[int(ticket)] = canonical_usd
            self._baseline_error_tickets.discard(int(ticket))
            return canonical, True
        except Exception as exc:
            if int(ticket) not in self._baseline_error_tickets:
                self.log(
                    f"Could not persist immutable initial risk for ticket {ticket}: {exc}",
                    "ERROR",
                )
                self._baseline_error_tickets.add(int(ticket))
            return initial_risk_pips, False

    async def _initial_risk_for_position(
        self, account: Dict[str, Any], position: Any, info: Any
    ) -> Tuple[float, bool]:
        ticket = int(position.ticket)
        cached = self._initial_risk_pips.get(ticket)
        if cached is not None and math.isfinite(cached) and cached > 0:
            return cached, True
        try:
            stored = await self.db.get_position_risk_baseline(
                self._account_scope(account), ticket
            )
            if stored is not None:
                canonical = float(stored.get("initial_risk_pips", 0.0) or 0.0)
                if not math.isfinite(canonical) or canonical <= 0:
                    raise ValueError("Stored initial risk is invalid")
                self._initial_risk_pips[ticket] = canonical
                canonical_usd = float(
                    stored.get("initial_risk_usd", 0.0) or 0.0
                )
                if not math.isfinite(canonical_usd) or canonical_usd <= 0:
                    raise ValueError("Stored initial dollar risk is invalid")
                if not hasattr(self, "_initial_risk_usd"):
                    self._initial_risk_usd = {}
                self._initial_risk_usd[ticket] = canonical_usd
                self._baseline_error_tickets.discard(ticket)
                return canonical, True

            # Upgrade path for positions opened before the baseline table was
            # introduced: recover the original SL from the local OPEN record
            # when possible, otherwise freeze the first broker-observed SL.
            record = await self.db.get_trade_record(ticket)
            initial_sl = float(position.sl or 0.0)
            if (
                record
                and str(record.get("symbol", "")).upper() == str(position.symbol).upper()
                and str(record.get("action", "")).upper() in {"BUY", "SELL"}
                and float(record.get("sl", 0.0) or 0.0) > 0
            ):
                initial_sl = float(record["sl"])
            return await self._persist_initial_risk(
                account=account,
                ticket=ticket,
                symbol=str(position.symbol),
                direction="BUY" if int(position.type) == mt5.POSITION_TYPE_BUY else "SELL",
                entry_price=float(position.price_open),
                initial_sl=initial_sl,
                volume=float(position.volume),
                info=info,
            )
        except Exception as exc:
            if ticket not in self._baseline_error_tickets:
                self.log(
                    f"Could not restore immutable initial risk for ticket {ticket}: {exc}",
                    "ERROR",
                )
                self._baseline_error_tickets.add(ticket)
            return 0.0, False

    @classmethod
    def _same_account(cls, account: Dict[str, Any], expected: Optional[Dict[str, Any]]) -> bool:
        if not expected:
            return False
        actual = cls._account_identity(account)
        return all(actual.get(key) == expected.get(key) for key in (
            "login", "server", "company", "trade_mode"
        ))

    async def start(self) -> bool:
        async with self._lifecycle_lock:
            return await self._start_locked()

    async def _start_locked(self) -> bool:
        """Starts the core trading orchestration loop."""
        if self.is_running:
            self.log("Trading engine is already running.")
            return True

        self.entries_armed = False
        self._armed_account_identity = None
        self._failure_task = None
        getattr(self, "_entry_model_admissions", {}).clear()
        self.executor.resume()
        dashboard_state.update_automation(
            entries_armed=False,
            safety_status="PAPER" if settings.dry_run else "LOCKED",
            safety_reason="Entries must be re-armed after every engine start",
        )
        self.log("Initializing trading engine connection systems...")
        if not await self.conn.initialize():
            self.log("Failed to initialize MT5 connection on startup.", "ERROR")
            return False

        # Load initial peak balance into risk manager
        account = await self.conn.get_account_info()
        if not account:
            self.log("MT5 account information is unavailable on startup.", "ERROR")
            return False
        self._active_account_login = int(account.get("login", 0) or 0)
        self._active_account_mode = str(account.get("trade_mode_name", "UNKNOWN"))
        self._active_account_identity = self._account_identity(account)
        await self._restore_daily_loss_reset(account)
        await self._restore_loss_cooldown_reset(account)
        await self._restore_loss_streak_reset(account)
        peak_key = (
            f"peak_balance:{account.get('server', 'unknown')}:"
            f"{account.get('login', 'unknown')}:{account.get('trade_mode', 'unknown')}"
        )
        persisted_peak = await self.db.get_cache(peak_key)
        self.risk.update_peak_balance(
            max(float(account.get("balance", 0.0)), float(persisted_peak or 0.0))
        )
        await self.db.set_cache(peak_key, self.risk.peak_balance)
        dashboard_state.update_account(account)
        self._history_ready = await self._reconcile_history(account)

        health = await self.llm.health_check()
        metadata_ready = bool(
            health.get("online") and health.get("available")
        )
        provider_name = str(
            health.get("provider", settings.llm_provider)
        ).lower()
        # Local metadata is not sufficient: LM Studio can expose a loaded
        # instance several seconds before its first completion succeeds.
        self._decision_provider_inference_ready = bool(
            metadata_ready and provider_name != "local"
        )
        self._consecutive_decision_failures = 0
        llm_ready = self._decision_provider_inference_ready
        dashboard_state.update_automation(
            llm_online=llm_ready,
            decision_provider_inference_ready=llm_ready,
            provider=str(health.get("provider", settings.llm_provider)),
            model=str(health.get("selected_model", settings.decision_model)),
            scan_status=(
                "WAITING FOR M5 CLOSE"
                if llm_ready
                else "MODEL WARMING UP"
                if metadata_ready
                else "LLM UNAVAILABLE"
            ),
        )
        provider = str(health.get("provider", settings.llm_provider)).upper()
        selected_model = str(health.get("selected_model", settings.decision_model))
        if llm_ready:
            self.log(
                f"{provider} decision model {selected_model} is ready. Scanning each "
                "completed M5 candle for entries with M15, H1 and H4 confirmation; "
                f"managed positions use completed "
                f"{settings.position_exit_review_timeframe} exit reviews and "
                f"{settings.decision_poll_seconds:.1f}s deterministic protection."
            )
        elif metadata_ready:
            self.log(
                f"{provider} decision model {selected_model} is loaded; "
                "entries remain locked until a bounded inference readiness "
                "probe succeeds."
            )
        else:
            self.log(
                f"{provider} decision model is unavailable: "
                f"{health.get('error', selected_model)}",
                "WARNING",
            )

        self.is_running = True
        dashboard_state.engine_running = True
        dashboard_state.engine_start_time = datetime.now()
        self._touch_engine_heartbeat(broker_poll=True)
        await self._restore_autonomous_authorization(
            account, llm_ready=llm_ready
        )
        self.loop_task = asyncio.create_task(self._main_loop())
        self.tick_loop_task = asyncio.create_task(self._tick_loop())
        self.protection_task = asyncio.create_task(self._protection_loop())
        self.autonomy_health_task = asyncio.create_task(
            self._autonomy_health_loop()
        )
        self.shadow_outcome_task = asyncio.create_task(
            self._shadow_outcome_loop()
        )
        self.loop_task.add_done_callback(
            lambda task: self._core_task_finished("decision", task)
        )
        self.tick_loop_task.add_done_callback(
            lambda task: self._core_task_finished("tick", task)
        )
        self.protection_task.add_done_callback(
            lambda task: self._core_task_finished("protection", task)
        )
        self.autonomy_health_task.add_done_callback(
            lambda task: self._core_task_finished("autonomy", task)
        )
        self._update_execution_readiness(account)
        self.log("Orchestrated trading loop started successfully.")
        return True

    async def arm_entries(self, confirmation: str = "") -> Tuple[bool, str]:
        """Arm entries for this process session only."""
        if not self.is_running:
            return False, "Start the decision engine before arming entries"
        account = await self.conn.get_account_info()
        if not account:
            return False, "MT5 account is unavailable"
        account_mode = str(account.get("trade_mode_name", "UNKNOWN"))
        if account_mode not in {"DEMO", "LIVE"}:
            return False, "The active MT5 account mode could not be verified"
        if not self._history_ready:
            return False, "Broker history/risk state has not reconciled successfully"
        protection_ok, protection_detail = self._protection_progress_health()
        if not protection_ok:
            return False, protection_detail
        permission_checks = {
            "account trading": bool(account.get("account_trade_allowed")),
            "expert trading": bool(account.get("expert_trading_allowed")),
            "terminal connection": bool(account.get("terminal_connected")),
            "Algo Trading": bool(account.get("terminal_trade_allowed")),
            "Python trading API": not bool(account.get("tradeapi_disabled", True)),
        }
        if not settings.dry_run:
            failed = [name for name, ok in permission_checks.items() if not ok]
            if failed:
                return False, f"MT5 readiness check failed: {', '.join(failed)}"
        health = await self.llm.health_check()
        provider_ready, provider_status = (
            await self._ensure_decision_provider_ready(health)
        )
        if not provider_ready:
            return (
                False,
                "The configured decision model is not inference-ready: "
                + str(
                    provider_status.get("error")
                    or "readiness probe did not succeed"
                ),
            )
        live = account_mode == "LIVE" and not settings.dry_run
        required = (
            "ARM PAPER" if settings.dry_run
            else "ARM LIVE" if account_mode == "LIVE"
            else "ARM DEMO"
        )
        if confirmation.strip().upper() != required:
            return False, f"Type {required} to confirm"
        if account_mode == "LIVE" and settings.dry_run:
            # Dry-run orders are safe, but make the mode explicit in the UI.
            self.log("Paper entries armed while connected to a live account; no order will be sent.", "WARNING")
        self.entries_armed = True
        self._armed_account_identity = self._account_identity(account)
        rescan_symbols = self._release_deferred_signals_for_rescan()
        dashboard_state.update_automation(
            entries_armed=True,
            safety_status=(
                "ARMED PAPER" if settings.dry_run
                else "ARMED DEMO" if account_mode == "DEMO"
                else "ARMED LIVE"
            ),
            safety_reason="Entry engine is armed for this session",
        )
        self.log("New entries armed for this session.", "WARNING" if live else "INFO")
        if rescan_symbols:
            self.log(
                "Fresh post-arm evaluation queued for previously disarmed signal(s): "
                + ", ".join(rescan_symbols)
            )
        self._update_execution_readiness(account)
        return True, "Entries armed"

    @staticmethod
    def _entry_permission_failures(account: Dict[str, Any]) -> List[str]:
        checks = {
            "account trading": bool(account.get("account_trade_allowed")),
            "expert trading": bool(account.get("expert_trading_allowed")),
            "terminal connection": bool(account.get("terminal_connected")),
            "Algo Trading": bool(account.get("terminal_trade_allowed")),
            "Python trading API": not bool(
                account.get("tradeapi_disabled", True)
            ),
        }
        return [name for name, ok in checks.items() if not ok]

    def _activate_autonomous_entries(
        self, account: Dict[str, Any], reason: str
    ) -> None:
        """Restore entry authorization for one previously approved identity."""
        self.entries_armed = True
        self._armed_account_identity = self._account_identity(account)
        rescan_symbols = self._release_deferred_signals_for_rescan()
        mode = str(account.get("trade_mode_name", "UNKNOWN"))
        dashboard_state.update_automation(
            entries_armed=True,
            autonomous_enabled=True,
            autonomous_status="ACTIVE",
            autonomous_account_suffix=str(account.get("login", ""))[-4:] or "—",
            last_health_check=datetime.now().strftime("%H:%M:%S"),
            safety_status=(
                "ARMED PAPER"
                if settings.dry_run
                else "ARMED DEMO"
                if mode == "DEMO"
                else "ARMED LIVE"
            ),
            safety_reason=reason,
        )
        if rescan_symbols:
            self.log(
                "Autonomous recovery queued a fresh evaluation for: "
                + ", ".join(rescan_symbols)
            )

    async def enable_autonomous_mode(
        self, confirmation: str = ""
    ) -> Tuple[bool, str]:
        """Persist unattended entry authority for the exact active account."""
        if not self.is_running:
            return False, "Start the decision engine before enabling autonomy"
        account = await self.conn.get_account_info()
        if not account:
            return False, "MT5 account is unavailable"
        mode = str(account.get("trade_mode_name", "UNKNOWN"))
        if mode not in {"DEMO", "LIVE"}:
            return False, "The active MT5 account mode could not be verified"
        label = "PAPER" if settings.dry_run else mode
        required = f"ENABLE AUTONOMOUS {label}"
        if confirmation.strip().upper() != required:
            return False, f"Type {required} to confirm"

        arm_phrase = (
            "ARM PAPER"
            if settings.dry_run
            else "ARM LIVE"
            if mode == "LIVE"
            else "ARM DEMO"
        )
        armed, message = await self.arm_entries(arm_phrase)
        if not armed:
            return False, message

        identity = self._account_identity(account)
        authorization = {
            "enabled": True,
            "account_identity": identity,
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }
        persisted = await self.db.set_cache(
            self._autonomous_cache_key(account), authorization
        )
        if not persisted:
            self.disarm_entries(
                "Autonomous authorization could not be persisted; entries locked"
            )
            return False, "Could not persist autonomous authorization"

        self.autonomous_enabled = True
        self._autonomous_account_identity = identity
        self._last_autonomy_health_state = "ACTIVE"
        dashboard_state.update_automation(
            autonomous_enabled=True,
            autonomous_status="ACTIVE",
            autonomous_account_suffix=str(account.get("login", ""))[-4:] or "—",
            last_health_check=datetime.now().strftime("%H:%M:%S"),
            safety_reason=(
                f"Account-bound autonomous {label} authorization is active"
            ),
        )
        self._update_execution_readiness(account)
        self.log(
            f"Autonomous {label} mode enabled for account ending "
            f"{str(account.get('login', ''))[-4:]}; authorization survives "
            "engine restarts but never transfers to another account.",
            "WARNING",
        )
        return True, f"Autonomous {label} mode enabled for this exact MT5 account"

    async def disable_autonomous_mode(self) -> Tuple[bool, str]:
        """Disable persistent autonomy and immediately disarm new entries."""
        account = await self.conn.get_account_info()
        authorization_account = account or self._autonomous_account_identity
        persistence_ok = not self.autonomous_enabled
        if authorization_account:
            persistence_ok = await self.db.set_cache(
                self._autonomous_cache_key(authorization_account),
                {
                    "enabled": False,
                    "disabled_at_utc": datetime.now(timezone.utc).isoformat(),
                    "version": 1,
                },
            )
        self.autonomous_enabled = False
        self._autonomous_account_identity = None
        self._last_autonomy_health_state = "DISABLED"
        self.disarm_entries("Autonomous mode disabled by the operator")
        dashboard_state.update_automation(
            autonomous_enabled=False,
            autonomous_status="DISABLED",
            autonomous_account_suffix="—",
            last_health_check=datetime.now().strftime("%H:%M:%S"),
        )
        if account:
            self._update_execution_readiness(account)
        if not persistence_ok:
            self.log(
                "Autonomy is disabled for this process, but the persistent "
                "authorization record could not be updated.",
                "ERROR",
            )
            return (
                False,
                "Disarmed now, but persistent authorization could not be cleared",
            )
        return True, "Autonomous mode disabled and entries disarmed"

    async def operator_disarm_entries(self) -> Tuple[bool, str]:
        """Manual disarm must also prevent an autonomous health re-arm."""
        if self.autonomous_enabled:
            return await self.disable_autonomous_mode()
        self.disarm_entries()
        account = await self.conn.get_account_info()
        if account:
            self._update_execution_readiness(account)
        return True, "Entries disarmed"

    async def _restore_autonomous_authorization(
        self, account: Dict[str, Any], *, llm_ready: bool
    ) -> None:
        """Restore only an exact account-bound authorization after restart."""
        stored = await self.db.get_cache(self._autonomous_cache_key(account))
        identity = self._account_identity(account)
        stored_identity = (
            stored.get("account_identity") if isinstance(stored, dict) else None
        )
        if (
            not isinstance(stored, dict)
            or not bool(stored.get("enabled"))
            or stored_identity != identity
        ):
            self.autonomous_enabled = False
            self._autonomous_account_identity = None
            dashboard_state.update_automation(
                autonomous_enabled=False,
                autonomous_status="DISABLED",
                autonomous_account_suffix="—",
            )
            return

        self.autonomous_enabled = True
        self._autonomous_account_identity = identity
        failures = (
            self._entry_permission_failures(account)
            if not settings.dry_run
            else []
        )
        protection_ok, _ = self._protection_progress_health()
        healthy = bool(
            llm_ready
            and self._history_ready
            and getattr(self, "_position_state_healthy", True)
            and self._position_risk_healthy
            and protection_ok
            and not failures
        )
        if healthy:
            self._activate_autonomous_entries(
                account,
                "Account-bound autonomous authorization restored after restart",
            )
            self._last_autonomy_health_state = "ACTIVE"
            self.log(
                "Restored autonomous entry authorization for account ending "
                f"{str(account.get('login', ''))[-4:]}.",
                "WARNING",
            )
        else:
            detail = (
                "decision model unavailable"
                if not llm_ready
                else "broker history unavailable"
                if not self._history_ready
                else "broker position state unavailable"
                if not getattr(self, "_position_state_healthy", True)
                else "position stop risk unavailable"
                if not self._position_risk_healthy
                else "broker protection supervisor unavailable"
                if not protection_ok
                else "MT5 permissions unavailable"
            )
            self.entries_armed = False
            self._armed_account_identity = None
            self._last_autonomy_health_state = f"WAITING:{detail}"
            dashboard_state.update_automation(
                entries_armed=False,
                autonomous_enabled=True,
                autonomous_status="WAITING",
                autonomous_account_suffix=str(account.get("login", ""))[-4:]
                or "—",
                last_health_check=datetime.now().strftime("%H:%M:%S"),
                safety_reason=f"Autonomous authorization waiting: {detail}",
            )

    async def _autonomy_health_loop(self) -> None:
        """Supervise provider and autonomous authorization fail-closed."""
        while self.is_running:
            try:
                health = await self.llm.health_check()
                metadata_ready = bool(
                    health.get("online") and health.get("available")
                )
                if not metadata_ready:
                    self._decision_provider_inference_ready = False
                    llm_ready = False
                elif not self._decision_provider_inference_ready:
                    llm_ready, health = (
                        await self._ensure_decision_provider_ready(health)
                    )
                else:
                    llm_ready = True
                dashboard_state.update_automation(
                    llm_online=llm_ready,
                    decision_provider_inference_ready=llm_ready,
                    provider=str(
                        health.get("provider", settings.llm_provider)
                    ),
                    model=str(
                        health.get(
                            "selected_model", settings.decision_model
                        )
                    ),
                    last_health_check=datetime.now().strftime("%H:%M:%S"),
                )

                if not llm_ready and self.entries_armed:
                    self.disarm_entries(
                        "Decision provider is not inference-ready"
                    )

                if self.autonomous_enabled:
                    account = await self.conn.get_account_info()
                    progress_ok, progress_detail = (
                        self._engine_progress_health()
                    )
                    protection_ok, protection_detail = (
                        self._protection_progress_health()
                    )
                    failure = ""
                    if not account:
                        failure = "MT5 account unavailable"
                    elif not self._same_account(
                        account, self._autonomous_account_identity
                    ):
                        failure = "active account does not match authorization"
                    elif not progress_ok:
                        failure = progress_detail
                    elif not protection_ok:
                        failure = protection_detail
                    elif not llm_ready:
                        failure = (
                            "decision model inference probe has not succeeded"
                        )
                    elif not self._history_ready:
                        failure = "broker history is not reconciled"
                    elif not getattr(self, "_position_state_healthy", True):
                        failure = "broker position state is unavailable"
                    elif not self._position_risk_healthy:
                        failure = "open-position stop risk is unavailable"
                    elif not settings.dry_run:
                        permission_failures = self._entry_permission_failures(
                            account
                        )
                        if permission_failures:
                            failure = (
                                "MT5 permissions unavailable: "
                                + ", ".join(permission_failures)
                            )

                    if failure:
                        state = f"WAITING:{failure}"
                        if self.entries_armed:
                            self.disarm_entries(
                                f"Autonomous fail-closed: {failure}"
                            )
                        dashboard_state.update_automation(
                            autonomous_enabled=True,
                            autonomous_status="WAITING",
                            safety_reason=(
                                f"Autonomous authorization waiting: {failure}"
                            ),
                        )
                    else:
                        state = "ACTIVE"
                        if not self.entries_armed:
                            self._activate_autonomous_entries(
                                account,
                                "Autonomous authorization restored after "
                                "verified service recovery",
                            )
                        dashboard_state.update_automation(
                            autonomous_enabled=True,
                            autonomous_status="ACTIVE",
                        )
                    self._update_execution_readiness(account)

                    if state != self._last_autonomy_health_state:
                        self.log(
                            "Autonomous health state: "
                            + (
                                "ACTIVE"
                                if state == "ACTIVE"
                                else state.replace("WAITING:", "WAITING · ")
                            ),
                            "INFO" if state == "ACTIVE" else "WARNING",
                        )
                        self._last_autonomy_health_state = state
            except asyncio.CancelledError:
                break
            except Exception as exc:
                state = f"ERROR:{exc}"
                if self.entries_armed:
                    self.disarm_entries(
                        "Autonomous fail-closed: health supervisor error"
                    )
                dashboard_state.update_automation(
                    autonomous_status="ERROR",
                    last_health_check=datetime.now().strftime("%H:%M:%S"),
                    safety_reason="Autonomous health supervisor error",
                )
                if state != self._last_autonomy_health_state:
                    self.log(
                        f"Autonomous health supervisor error: {exc}", "ERROR"
                    )
                    self._last_autonomy_health_state = state
            await asyncio.sleep(settings.autonomous_health_interval_seconds)

    async def reset_daily_loss_stop(self, confirmation: str = "") -> Tuple[bool, str]:
        """Reset today's strategy loss counter and global loss cooldown."""
        if confirmation.strip().upper() != "RESET DAILY LOSS":
            return False, "Type RESET DAILY LOSS to confirm"
        account = await self.conn.get_account_info()
        if not account:
            return False, "MT5 account is unavailable"
        if self._active_account_identity and not self._same_account(
            account, self._active_account_identity
        ):
            return False, "The active MT5 account changed; reset was not applied"

        marker = datetime.now(timezone.utc)
        persisted = await self.db.set_cache(
            self._daily_loss_reset_cache_key(account), marker.isoformat()
        )
        if not persisted:
            return False, "Could not persist the UTC daily-loss reset"
        self.risk.reset_daily_loss_for_today(marker)
        for symbol, decision in dashboard_state.symbol_decisions_snapshot().items():
            if "Loss Cooldown]" in str(decision.get("gate_reason", "")):
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="WAITING",
                    action="—",
                    confidence=0.0,
                    gate_reason="",
                    reasoning="Loss cooldown cleared; waiting for the next completed M5 candle.",
                )
        self._update_execution_readiness(account)
        self.log(
            "UTC daily-loss counter reset to $0.00 and all loss cooldown state "
            "cleared by the operator; new strategy losses will count "
            "from this point.",
            "WARNING",
        )
        return True, "UTC daily-loss counter reset to $0.00; loss cooldown state cleared"

    async def reset_loss_cooldown(self, confirmation: str = "") -> Tuple[bool, str]:
        """Clear only the current account-wide post-loss cooldown."""
        if confirmation.strip().upper() != "RESET LOSS COOLDOWN":
            return False, "Type RESET LOSS COOLDOWN to confirm"
        account = await self.conn.get_account_info()
        if not account:
            return False, "MT5 account is unavailable"
        if self._active_account_identity and not self._same_account(
            account, self._active_account_identity
        ):
            return False, "The active MT5 account changed; reset was not applied"

        marker = datetime.now(timezone.utc)
        persisted = await self.db.set_cache(
            self._loss_cooldown_reset_cache_key(account), marker.isoformat()
        )
        if not persisted:
            return False, "Could not persist the loss-cooldown reset"
        self.risk.reset_loss_cooldown(marker)
        for symbol, decision in dashboard_state.symbol_decisions_snapshot().items():
            if "Loss Cooldown]" not in str(decision.get("gate_reason", "")):
                continue
            manual_available = bool(decision.get("manual_override_available", False))
            dashboard_state.update_symbol_decision(
                symbol,
                stage="MANUAL REVIEW" if manual_available else "WAITING",
                gate_reason="",
                reasoning=(
                    "Loss cooldown cleared. This rejected candidate remains available "
                    "for manual review; automation will evaluate the next completed M5 candle."
                    if manual_available
                    else "Loss cooldown cleared; waiting for the next completed M5 candle."
                ),
            )
        self._update_execution_readiness(account)
        self.log(
            "Account-wide loss cooldown reset by the operator; UTC daily-loss totals "
            "and all other risk protections were left unchanged.",
            "WARNING",
        )
        return True, "Loss cooldown reset; daily-loss totals were not changed"

    async def reset_losing_streak(self, confirmation: str = "") -> Tuple[bool, str]:
        """Clear the current account's adaptive losing-streak window."""
        if confirmation.strip().upper() != "RESET LOSING STREAK":
            return False, "Type RESET LOSING STREAK to confirm"
        account = await self.conn.get_account_info()
        if not account:
            return False, "MT5 account is unavailable"
        if self._active_account_identity and not self._same_account(
            account, self._active_account_identity
        ):
            return False, "The active MT5 account changed; reset was not applied"

        marker = datetime.now(timezone.utc)
        persisted = await self.db.set_cache(
            self._loss_streak_reset_cache_key(account), marker.isoformat()
        )
        if not persisted:
            return False, "Could not persist the losing-streak reset"
        self.risk.reset_losing_streak(marker)
        for symbol, decision in dashboard_state.symbol_decisions_snapshot().items():
            if "[Losing Streak Pause]" not in str(decision.get("gate_reason", "")):
                continue
            manual_available = bool(decision.get("manual_override_available", False))
            dashboard_state.update_symbol_decision(
                symbol,
                stage="MANUAL REVIEW" if manual_available else "WAITING",
                gate_reason="",
                reasoning=(
                    "Losing-streak pause cleared. This rejected candidate remains "
                    "available for manual review; automation will evaluate the next "
                    "completed M5 candle."
                    if manual_available
                    else "Losing-streak pause cleared; waiting for the next completed M5 candle."
                ),
            )
        self._update_execution_readiness(account)
        self.log(
            "Account-wide losing-streak pause reset by the operator; broker history, "
            "balance, and UTC daily-loss totals were left unchanged.",
            "WARNING",
        )
        return True, "Losing-streak pause reset; prior broker losses remain in history"

    async def _prepare_manual_trade_override(
        self, symbol: str
    ) -> Tuple[bool, Dict[str, Any], Optional[Dict[str, Any]]]:
        symbol = str(symbol).strip().upper()
        candidate = self._manual_trade_candidates.get(symbol)
        if not candidate:
            return False, {"reason": "No current rejected trade is available for manual review"}, None
        if not self.is_running:
            return False, {"reason": "The decision engine is not running"}, None
        if not self._history_ready:
            return False, {"reason": "Broker history/risk state is not reconciled"}, None
        if not await self.conn.is_connected() and not await self.conn.initialize():
            return False, {"reason": "MT5 connection is unavailable"}, None

        account = await self.conn.get_account_info()
        if not account or str(account.get("trade_mode_name", "UNKNOWN")) not in {"DEMO", "LIVE"}:
            return False, {"reason": "Active MT5 account identity could not be verified"}, None
        if not self._same_account(account, candidate.get("account_identity")):
            self._clear_manual_trade_candidate(symbol)
            return False, {"reason": "The rejected signal belongs to a different MT5 account"}, None

        permission_checks = {
            "account trading": bool(account.get("account_trade_allowed")),
            "expert trading": bool(account.get("expert_trading_allowed")),
            "terminal connection": bool(account.get("terminal_connected")),
            "Algo Trading": bool(account.get("terminal_trade_allowed")),
            "Python trading API": not bool(account.get("tradeapi_disabled", True)),
        }
        if not settings.dry_run:
            failed = [name for name, ok in permission_checks.items() if not ok]
            if failed:
                return False, {"reason": f"MT5 readiness check failed: {', '.join(failed)}"}, None

        bar_is_current, bar_reason = await self._entry_bar_is_current(
            symbol, str(candidate.get("completed_bar", ""))
        )
        if not bar_is_current:
            self._clear_manual_trade_candidate(symbol)
            return False, {"reason": bar_reason}, None

        raw_positions = await asyncio.to_thread(mt5.positions_get)
        if raw_positions is None:
            return False, {"reason": f"MT5 position query failed: {mt5.last_error()}"}, None
        open_positions = [
            {
                "ticket": int(p.ticket),
                "symbol": str(p.symbol),
                "type": int(p.type),
                "volume": float(p.volume),
                "price_open": float(p.price_open),
                "sl": float(p.sl),
                "tp": float(p.tp),
                "magic": int(p.magic),
                "bot_owned": int(p.magic) == settings.strategy_magic,
            }
            for p in raw_positions
        ]
        action = str(candidate.get("action", "")).upper()
        decision = dict(candidate.get("decision") or {})
        stored_plan = dict(candidate.get("plan") or {})
        stop_loss = float(
            stored_plan.get("stop_loss", decision.get("stop_loss", 0.0)) or 0.0
        )
        take_profit = float(
            stored_plan.get("take_profit", decision.get("take_profit", 0.0)) or 0.0
        )
        if stop_loss <= 0 or take_profit <= 0:
            return False, {"reason": "The rejected signal has no broker-valid SL/TP plan"}, None

        symbol_info, tick = await asyncio.gather(
            asyncio.to_thread(mt5.symbol_info, symbol),
            asyncio.to_thread(mt5.symbol_info_tick, symbol),
        )
        if symbol_info is None or tick is None:
            return False, {"reason": "The current broker quote is unavailable"}, None
        entry = float(tick.ask if action == "BUY" else tick.bid)
        if (action == "BUY" and not (stop_loss < entry < take_profit)) or (
            action == "SELL" and not (take_profit < entry < stop_loss)
        ):
            self._clear_manual_trade_candidate(symbol)
            return False, {
                "reason": "Price moved beyond the rejected signal's SL/TP range; wait for a new signal"
            }, None
        risk_distance = abs(entry - stop_loss)
        raw_rr = abs(take_profit - entry) / risk_distance if risk_distance > 0 else 0.0
        plan = TradePlan(
            valid=True,
            action=action,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_distance=risk_distance,
            planned_rr=raw_rr,
            source="Stored rejected-signal levels",
        )
        decision.update(
            action=action,
            entry=plan.entry,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
        )

        class RiskSnapshot:
            def __init__(self, bid, ask, point, contract_size, spread):
                class Metrics:
                    def __init__(self, bid, ask, point, contract_size, spread):
                        self.bid = bid
                        self.ask = ask
                        self.point = point
                        self.contract_size = contract_size
                        self.spread = spread
                self.metrics = Metrics(bid, ask, point, contract_size, spread)

        risk_snapshot = RiskSnapshot(
            tick.bid,
            tick.ask,
            symbol_info.point,
            symbol_info.trade_contract_size,
            symbol_info.spread,
        )
        history = await self.db.get_closed_positions(
            symbol=mt5.broker_symbol_name(symbol),
            limit=20,
            account_login=self._active_account_login,
        )
        validation = await asyncio.to_thread(
            self.risk.validate_manual_override,
            symbol=symbol,
            action=action,
            decision=decision,
            account_info=account,
            open_positions=open_positions,
            market_snapshot=risk_snapshot,
            trade_history=history,
        )
        if not validation.approved or not validation.adjusted_lot:
            return False, {"reason": validation.reason}, None

        lot = float(validation.adjusted_lot)
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        entry = float(tick.ask if action == "BUY" else tick.bid)
        margin = await asyncio.to_thread(
            mt5.order_calc_margin, order_type, symbol, lot, entry
        )
        if margin is None or not math.isfinite(float(margin)) or float(margin) <= 0:
            return False, {"reason": "MT5 could not calculate projected margin"}, None
        margin = float(margin)
        free_margin = float(account.get("margin_free", 0.0) or 0.0)
        if margin > free_margin + 1e-9:
            return False, {
                "reason": (
                    f"Pepperstone margin requirement is ${margin:.2f}, but only "
                    f"${free_margin:.2f} free margin is available"
                )
            }, None
        equity = float(account.get("equity", 0.0) or 0.0)
        current_margin = float(account.get("margin", 0.0) or 0.0)
        projected_margin_pct = (
            (current_margin + margin) / equity * 100.0 if equity > 0 else 100.0
        )
        required_phrase = f"FORCE OPEN REJECTED {symbol}"
        preview = {
            "available": True,
            "symbol": symbol,
            "action": action,
            "lot": lot,
            "entry": entry,
            "stop_loss": float(plan.stop_loss),
            "take_profit": float(plan.take_profit),
            "risk_usd": float(validation.estimated_risk_usd),
            "reward_usd": float(validation.estimated_reward_usd),
            "risk_pct": float(validation.risk_percent_balance),
            "execution_risk_ceiling_usd": float(validation.risk_budget_usd),
            "net_rr": float(validation.planned_rr),
            "margin_usd": margin,
            "projected_margin_pct": projected_margin_pct,
            "original_rejection": str(candidate.get("original_rejection", "Rejected by strategy gate")),
            "confirmation_phrase": required_phrase,
            "expires": "next completed M5 candle",
            "automatic_trading_unchanged": True,
            "project_policy_gates_bypassed": True,
        }
        internal = {
            "candidate": candidate,
            "account": account,
            "symbol_info": symbol_info,
            "decision": decision,
            "plan": plan,
            "validation": validation,
            "preview": preview,
        }
        return True, preview, internal

    async def preview_rejected_trade(self, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        """Return a current, broker-calculated preview without sending an order."""
        async with self._execution_lock:
            ok, payload, _ = await self._prepare_manual_trade_override(symbol)
            return ok, payload

    async def open_rejected_trade(
        self, symbol: str, confirmation: str
    ) -> Tuple[bool, str]:
        """Submit one explicitly confirmed rejected signal through the safe executor."""
        symbol = str(symbol).strip().upper()
        required = f"FORCE OPEN REJECTED {symbol}"
        if confirmation.strip().upper() != required:
            return False, f"Type {required} to confirm"

        async with self._execution_lock:
            ok, payload, internal = await self._prepare_manual_trade_override(symbol)
            if not ok or internal is None:
                return False, str(payload.get("reason", "Manual override is unavailable"))
            preview = internal["preview"]
            validation = internal["validation"]
            plan = internal["plan"]
            account = internal["account"]
            candidate = internal["candidate"]
            action = str(preview["action"])
            lot = float(preview["lot"])
            result = await self.executor.open_trade(
                symbol=symbol,
                action=action,
                lot_size=lot,
                sl_price=float(plan.stop_loss),
                tp_price=float(plan.take_profit),
                comment=f"{settings.order_comment}-Override",
                expected_account=self._account_identity(account),
                max_risk_usd=float(validation.risk_budget_usd),
                operator_override=True,
            )
            if not result.success:
                if result.state_changed:
                    self.disarm_entries(
                        "Manual override changed broker state but could not be fully verified"
                    )
                return False, result.error or "Pepperstone rejected the manual override"

            actual_lot = float(result.volume or lot)
            await self.db.log_trade({
                "ticket": result.ticket,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "action": action,
                "lot_size": actual_lot,
                "price": result.price or preview["entry"],
                "sl": float(plan.stop_loss),
                "tp": float(plan.take_profit),
                "profit": 0.0,
                "reasoning": (
                    f"Operator override of: {candidate.get('original_rejection', '')}"
                ),
                "status": "OPEN_MANUAL_OVERRIDE",
            })
            baseline_ok = True
            if not settings.dry_run and result.ticket:
                _, baseline_ok = await self._persist_initial_risk(
                    account=account,
                    ticket=int(result.ticket),
                    symbol=symbol,
                    direction=action,
                    entry_price=float(result.price or preview["entry"]),
                    initial_sl=float(plan.stop_loss),
                    volume=actual_lot,
                    info=internal["symbol_info"],
                )
                if not baseline_ok:
                    self._position_risk_healthy = False
                    self.disarm_entries(
                        "Override opened, but immutable risk state could not be persisted"
                    )
            await self._log_replay_attempt(
                symbol=symbol,
                action=action,
                prompt_text=str(candidate.get("prompt_text", "")),
                llm_json=internal["decision"],
                indicators=(candidate.get("m5_analysis") or {}).get("indicators", {}),
                market_structure=(candidate.get("m5_analysis") or {}).get("market_structure", {}),
                quality_score=float(candidate.get("quality_score", 0.0)),
                confluence_score=float(candidate.get("confluence_score", 0.0)),
                status="OPEN_MANUAL_OVERRIDE",
                ticket=result.ticket,
            )
            self._clear_manual_trade_candidate(symbol)
            dashboard_state.update_symbol_decision(
                symbol,
                stage="OPENED · MANUAL OVERRIDE",
                gate_reason=f"Ticket {result.ticket}",
                manual_override_available=False,
            )
            self.log(
                f"Operator opened rejected {action} candidate for {symbol}; "
                f"ticket={result.ticket}, volume={actual_lot:g}.",
                "WARNING",
            )
            return True, f"Manual override opened and verified; ticket {result.ticket}"

    def disarm_entries(self, reason: str = "Disarmed by operator") -> None:
        self.entries_armed = False
        self._armed_account_identity = None
        dashboard_state.update_automation(
            entries_armed=False,
            safety_status="PAPER" if settings.dry_run else "LOCKED",
            safety_reason=reason,
        )
        self.log(reason, "WARNING")

    async def close_position(self, ticket: int, percent: float, confirmation: str) -> Tuple[bool, str]:
        async with self._execution_lock:
            if not await self.conn.is_connected() and not await self.conn.initialize():
                return False, "MT5 connection is unavailable"
            account = await self.conn.get_account_info()
            if not account or str(account.get("trade_mode_name", "UNKNOWN")) not in {"DEMO", "LIVE"}:
                return False, "Active MT5 account identity could not be verified"
            positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
            if positions is None:
                return False, f"MT5 position query failed: {mt5.last_error()}"
            if not positions:
                return False, "Position not found"
            live = bool(account.get("trade_mode_name") == "LIVE" and not settings.dry_run)
            if live and confirmation.strip().upper() != f"CLOSE {ticket}":
                return False, f"Type CLOSE {ticket} to confirm"
            percent = max(1.0, min(float(percent), 100.0))
            position = positions[0]
            if percent >= 99.999:
                result = await self.executor.close_position(
                    ticket,
                    comment=f"{settings.order_comment}-UI",
                    expected_account=self._account_identity(account),
                )
            else:
                result = await self.executor.partial_close(
                    ticket,
                    position.volume * percent / 100.0,
                    comment=f"{settings.order_comment}-UI",
                    expected_account=self._account_identity(account),
                )
            if not result.success:
                return False, result.error or "Close failed"
            if percent >= 99.999 and int(position.magic) == settings.strategy_magic:
                await self._remember_strategy_close_reason(
                    int(ticket), "OPERATOR_CLOSE"
                )
            self.log(f"Operator close verified for ticket {ticket} ({percent:.0f}%).")
            return True, "Close verified"

    async def modify_position(
        self, ticket: int, sl: Optional[float], tp: Optional[float], confirmation: str
    ) -> Tuple[bool, str]:
        async with self._execution_lock:
            if not await self.conn.is_connected() and not await self.conn.initialize():
                return False, "MT5 connection is unavailable"
            account = await self.conn.get_account_info()
            if not account or str(account.get("trade_mode_name", "UNKNOWN")) not in {"DEMO", "LIVE"}:
                return False, "Active MT5 account identity could not be verified"
            positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
            if positions is None:
                return False, f"MT5 position query failed: {mt5.last_error()}"
            if not positions:
                return False, "Position not found"
            live = bool(account.get("trade_mode_name") == "LIVE" and not settings.dry_run)
            if live and confirmation.strip().upper() != f"PROTECT {ticket}":
                return False, f"Type PROTECT {ticket} to confirm"
            result = await self.executor.modify_sl_tp(
                ticket,
                sl_price=sl,
                tp_price=tp,
                expected_account=self._account_identity(account),
            )
            if not result.success:
                return False, result.error or "Modification failed"
            self.log(f"Protection updated for ticket {ticket}.")
            return True, "Protection updated"

    async def _enrich_reconciled_rows(
        self,
        rows: List[Dict[str, Any]],
        account: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Attach immutable-risk and excursion telemetry to broker outcomes."""
        if not rows:
            return rows
        baselines = await self.db.get_position_risk_baselines(
            self._account_scope(account)
        )
        by_ticket = {
            int(row.get("ticket", 0) or 0): row for row in baselines
        }
        account_login = int(account.get("login", 0) or 0)
        position_ids = [
            int(row.get("position_id", 0) or 0) for row in rows
        ]
        reason_reader = getattr(self.db, "get_strategy_close_reasons", None)
        strategy_close_reasons = (
            await reason_reader(account_login, position_ids)
            if reason_reader is not None and account_login > 0
            else {}
        )
        strategy_close_reasons.update(
            {
                int(ticket): str(reason).strip().upper()
                for ticket, reason in getattr(
                    self, "_pending_close_reasons", {}
                ).items()
                if int(ticket) in position_ids and str(reason).strip()
            }
        )
        generic_broker_reasons = {
            "", "CLIENT", "MOBILE", "WEB", "EXPERT", "OTHER",
            "UNKNOWN", "SIGNAL",
        }
        symbols = {
            str(row.get("symbol", "")).strip()
            for row in rows
            if str(row.get("symbol", "")).strip()
        }

        def _load_symbol_info() -> Dict[str, Any]:
            return {symbol: mt5.symbol_info(symbol) for symbol in symbols}

        info_by_symbol = await asyncio.to_thread(_load_symbol_info)
        for row in rows:
            direction = str(row.get("direction", "")).upper()
            open_price = float(row.get("open_price", 0.0) or 0.0)
            close_price = float(row.get("close_price", 0.0) or 0.0)
            info = info_by_symbol.get(str(row.get("symbol", "")))
            one_pip = pip_size(info) if info is not None else 0.0
            signed_move = (
                close_price - open_price
                if direction == "BUY"
                else open_price - close_price
            )
            profit_pips = signed_move / one_pip if one_pip > 0 else 0.0
            baseline = by_ticket.get(int(row.get("position_id", 0) or 0), {})
            position_id = int(row.get("position_id", 0) or 0)
            broker_close_reason = str(
                row.get("close_reason", "") or ""
            ).strip().upper()
            strategy_close_reason = strategy_close_reasons.get(position_id, "")
            if (
                broker_close_reason in generic_broker_reasons
                and strategy_close_reason
            ):
                row["close_reason"] = strategy_close_reason
            initial_risk_pips = float(
                baseline.get("initial_risk_pips", 0.0) or 0.0
            )
            initial_risk_usd = float(
                baseline.get("initial_risk_usd", 0.0) or 0.0
            )
            net_profit = float(row.get("net_profit", 0.0) or 0.0)
            row.update(
                profit_pips=round(profit_pips, 5),
                initial_risk_pips=round(initial_risk_pips, 5),
                initial_risk_usd=round(initial_risk_usd, 5),
                mfe_pips=round(
                    max(
                        0.0,
                        float(baseline.get("mfe_pips", 0.0) or 0.0),
                        profit_pips,
                    ),
                    5,
                ),
                mae_pips=round(
                    min(
                        0.0,
                        float(baseline.get("mae_pips", 0.0) or 0.0),
                        profit_pips,
                    ),
                    5,
                ),
                mfe_usd=round(
                    max(
                        0.0,
                        float(baseline.get("mfe_usd", 0.0) or 0.0),
                        net_profit,
                    ),
                    5,
                ),
                mae_usd=round(
                    min(
                        0.0,
                        float(baseline.get("mae_usd", 0.0) or 0.0),
                        net_profit,
                    ),
                    5,
                ),
                rr_achieved=round(
                    net_profit / initial_risk_usd
                    if initial_risk_usd > 0
                    else 0.0,
                    5,
                ),
            )
        return rows

    async def _reconcile_history(self, account: Dict[str, Any]) -> bool:
        try:
            rows = await asyncio.to_thread(self.history_reconciler.fetch, 90)
            rows = await self._enrich_reconciled_rows(rows, account)
            account_login = int(account.get("login", 0) or 0)
            await self.db.upsert_closed_positions(rows, account_login=account_login)
            history = await self.db.get_closed_positions(
                limit=200, account_login=account_login
            )
            self.risk.synchronize_closed_trades(history, float(account.get("balance", 0.0)))
            peak_key = (
                f"peak_balance:{account.get('server', 'unknown')}:"
                f"{account.get('login', 'unknown')}:{account.get('trade_mode', 'unknown')}"
            )
            persisted_peak = await self.db.get_cache(peak_key)
            self.risk.update_peak_balance(max(self.risk.peak_balance, float(persisted_peak or 0.0)))
            await self.db.set_cache(peak_key, self.risk.peak_balance)
            dashboard_state.load_broker_history(history)
            dashboard_state.update_automation(
                last_reconciled=datetime.now().strftime("%H:%M:%S"),
                history_healthy=True,
                history_status="BROKER SYNCED",
            )
            for trade in history[:20]:
                await self._update_replay_outcome(
                    int(trade["position_id"]),
                    float(trade["net_profit"]),
                    close_time=str(trade.get("close_time", "")),
                    close_reason=str(trade.get("close_reason", "")),
                    mfe_pips=float(trade.get("mfe_pips", 0.0) or 0.0),
                    mae_pips=float(trade.get("mae_pips", 0.0) or 0.0),
                    mfe_usd=float(trade.get("mfe_usd", 0.0) or 0.0),
                    mae_usd=float(trade.get("mae_usd", 0.0) or 0.0),
                    rr_achieved=float(trade.get("rr_achieved", 0.0) or 0.0),
                )
            if getattr(settings, "exit_counterfactual_enabled", True):
                await asyncio.to_thread(
                    self.replay_logger.log_exit_candidates,
                    account_login=account_login,
                    trades=history[:20],
                    horizons_minutes=list(
                        settings.exit_counterfactual_horizons_minutes
                    ),
                )
            self._history_ready = True
            return True
        except Exception as exc:
            self._history_ready = False
            dashboard_state.update_automation(
                history_healthy=False,
                history_status="SYNC ERROR",
            )
            if self.entries_armed:
                self.disarm_entries("Broker history reconciliation failed; entries locked")
            self.log(f"Broker history reconciliation failed: {exc}", "ERROR")
            return False

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stops the core loop and disconnects cleanly."""
        if not self.is_running:
            return

        self.log("Requesting trading engine stop...")
        self.disarm_entries(
            "Engine stopped; account-bound autonomy will be restored on restart"
            if getattr(self, "autonomous_enabled", False)
            else "Engine stopped; entries require explicit re-arming"
        )
        for symbol in list(getattr(self, "_manual_trade_candidates", {})):
            self._clear_manual_trade_candidate(symbol)
        self.executor.begin_shutdown()
        self.is_running = False
        dashboard_state.engine_running = False
        
        # Cancel loops
        if self.loop_task:
            self.loop_task.cancel()
        if self.tick_loop_task:
            self.tick_loop_task.cancel()
        protection_task = getattr(self, "protection_task", None)
        if protection_task:
            protection_task.cancel()
        autonomy_task = getattr(self, "autonomy_health_task", None)
        if autonomy_task:
            autonomy_task.cancel()
        shadow_task = getattr(self, "shadow_outcome_task", None)
        if shadow_task:
            shadow_task.cancel()
        for task in self.analysis_tasks.values():
            task.cancel()
        for task in getattr(self, "position_exit_tasks", {}).values():
            task.cancel()
        selection_task = getattr(self, "market_selection_task", None)
        if selection_task:
            selection_task.cancel()
            
        try:
            if self.loop_task:
                await self.loop_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.log(f"Decision loop stopped with error during shutdown: {exc}", "ERROR")
        if self.analysis_tasks:
            await asyncio.gather(*self.analysis_tasks.values(), return_exceptions=True)
        self.analysis_tasks.clear()
        getattr(self, "_entry_model_admissions", {}).clear()
        if getattr(self, "position_exit_tasks", {}):
            await asyncio.gather(
                *self.position_exit_tasks.values(), return_exceptions=True
            )
            self.position_exit_tasks.clear()
        if selection_task:
            await asyncio.gather(selection_task, return_exceptions=True)
        if shadow_task:
            await asyncio.gather(shadow_task, return_exceptions=True)
        try:
            if self.tick_loop_task:
                await self.tick_loop_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.log(f"Tick loop stopped with error during shutdown: {exc}", "ERROR")
        try:
            if protection_task:
                await protection_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.log(
                f"Protection loop stopped with error during shutdown: {exc}",
                "ERROR",
            )
        try:
            if autonomy_task:
                await autonomy_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.log(
                f"Autonomous health loop stopped with error during shutdown: {exc}",
                "ERROR",
            )
            
        self.loop_task = None
        self.tick_loop_task = None
        self.protection_task = None
        self.autonomy_health_task = None
        self.shadow_outcome_task = None
        self.market_selection_task = None
        self._pending_market_selection_account = None

        drained = await self.executor.wait_until_idle(timeout=30.0)
        if drained:
            await self.conn.shutdown()
        else:
            self.log(
                "MT5 native work did not drain; terminal connection left intact to avoid a shutdown race.",
                "ERROR",
            )
        self._update_execution_readiness(None)
        self.log("Trading engine shutdown complete.")

    async def _protection_loop(self) -> None:
        """Priority broker-position supervisor independent of market scans.

        Market ranking, SQLite reconciliation, and model inference may take
        seconds. Profit floors and deterministic exits must not inherit that
        latency, so this loop owns the fast protection cadence.
        """
        last_error_log = 0.0
        while self.is_running:
            cycle_started = time.monotonic()
            try:
                account = dict(self._active_account_identity or {})
                if not account:
                    self._touch_protection_heartbeat(healthy=False)
                    await asyncio.sleep(settings.decision_poll_seconds)
                    continue
                raw_positions = await asyncio.to_thread(mt5.positions_get)
                if raw_positions is None:
                    self._touch_protection_heartbeat(healthy=False)
                    now_mono = time.monotonic()
                    if now_mono - last_error_log >= 30.0:
                        self.log(
                            f"Priority protection position refresh failed: "
                            f"{mt5.last_error()}",
                            "ERROR",
                        )
                        last_error_log = now_mono
                    await asyncio.sleep(settings.decision_poll_seconds)
                    continue

                managed: List[Dict[str, Any]] = []
                for position in raw_positions:
                    bot_owned = int(position.magic) == settings.strategy_magic
                    if not (bot_owned or settings.manage_external_positions):
                        continue
                    info = await asyncio.to_thread(
                        mt5.symbol_info, position.symbol
                    )
                    one_pip = pip_size(info) if info is not None else 0.0
                    if one_pip > 0 and int(position.type) == mt5.POSITION_TYPE_BUY:
                        profit_pips = (
                            float(position.price_current)
                            - float(position.price_open)
                        ) / one_pip
                    elif one_pip > 0:
                        profit_pips = (
                            float(position.price_open)
                            - float(position.price_current)
                        ) / one_pip
                    else:
                        # Dollar peak/floor protection is still valid when a
                        # transient metadata read prevents pip/R calculation.
                        profit_pips = 0.0
                    if info is not None and one_pip > 0:
                        initial_risk_pips, baseline_ok = (
                            await self._initial_risk_for_position(
                                account, position, info
                            )
                        )
                    else:
                        initial_risk_pips, baseline_ok = 0.0, False
                    if not baseline_ok or initial_risk_pips <= 0:
                        initial_risk_pips = 0.0
                    peak_pips, peak_usd = (
                        await self._restore_and_update_position_peak(
                            account,
                            ticket=int(position.ticket),
                            profit_pips=profit_pips,
                            observed_profit_usd=float(position.profit),
                        )
                    )
                    try:
                        configured_cost = (
                            configured_execution_cost_usd(
                                position.symbol, info, float(position.volume)
                            )
                            if info is not None
                            else 0.0
                        )
                    except (TypeError, ValueError, OverflowError):
                        configured_cost = 0.0
                    now_epoch = datetime.now(timezone.utc).timestamp()
                    live_tick = await asyncio.to_thread(
                        mt5.symbol_info_tick, position.symbol
                    )
                    server_offset = infer_positive_server_offset_seconds(
                        float(getattr(live_tick, "time", 0.0) or 0.0),
                        now_epoch=now_epoch,
                    )
                    opened_epoch = normalized_broker_epoch(
                        float(position.time), server_offset
                    )
                    duration_min = max(
                        0.0, (now_epoch - opened_epoch) / 60.0
                    )
                    managed.append(
                        {
                            "ticket": int(position.ticket),
                            "symbol": str(position.symbol),
                            "type": int(position.type),
                            "volume": float(position.volume),
                            "price_open": float(position.price_open),
                            "price_current": float(position.price_current),
                            "sl": float(position.sl or 0.0),
                            "tp": float(position.tp or 0.0),
                            "profit": float(position.profit),
                            "estimated_net_profit_usd": (
                                float(position.profit)
                                + float(getattr(position, "swap", 0.0) or 0.0)
                                - configured_cost
                            ),
                            "profit_pips": profit_pips,
                            "peak_profit_pips": peak_pips,
                            "peak_profit_usd": peak_usd,
                            "initial_risk_pips": initial_risk_pips,
                            "initial_risk_usd": getattr(
                                self, "_initial_risk_usd", {}
                            ).get(
                                int(position.ticket), 0.0
                            ),
                            "duration_min": duration_min,
                            "bot_owned": bot_owned,
                            "account_identity": dict(account),
                        }
                    )
                if managed:
                    async with self._execution_lock:
                        await self._apply_protections(
                            managed,
                            refresh_from_broker=True,
                        )
                self._touch_protection_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._touch_protection_heartbeat(healthy=False)
                now_mono = time.monotonic()
                if now_mono - last_error_log >= 30.0:
                    self.log(
                        f"Priority protection loop error: {exc}", "ERROR"
                    )
                    last_error_log = now_mono

            elapsed = time.monotonic() - cycle_started
            await asyncio.sleep(
                max(0.05, settings.decision_poll_seconds - elapsed)
            )

    async def _main_loop(self) -> None:
        """Main periodic loop execution."""
        while self.is_running:
            self._touch_engine_heartbeat()
            try:
                # 1. Heartbeat check
                if not await self.conn.is_connected():
                    self.log("MT5 connection is offline. Reinitializing...", "WARNING")
                    if not await self.conn.initialize():
                        await asyncio.sleep(10)
                        continue

                # 2. Update general account and position states
                account = await self.conn.get_account_info()
                if not account:
                    await asyncio.sleep(2)
                    continue

                current_login = int(account.get("login", 0) or 0)
                current_mode = str(account.get("trade_mode_name", "UNKNOWN"))
                current_identity = self._account_identity(account)
                if self._active_account_identity and current_identity != self._active_account_identity:
                    previous = str(self._active_account_login)[-4:]
                    current = str(current_login)[-4:]
                    self.disarm_entries(
                        f"Pepperstone account changed ({previous} -> {current}); re-arm entries for {current_mode}"
                    )
                    if self.autonomous_enabled:
                        self.autonomous_enabled = False
                        self._autonomous_account_identity = None
                        self._last_autonomy_health_state = "ACCOUNT_CHANGED"
                        dashboard_state.update_automation(
                            autonomous_enabled=False,
                            autonomous_status="ACCOUNT CHANGED",
                            autonomous_account_suffix="—",
                            safety_reason=(
                                "Autonomous authorization does not transfer "
                                "between MT5 accounts"
                            ),
                        )
                    # State from the previous account must not be interpreted as
                    # positions closing on the newly selected account.
                    self._last_positions_tickets.clear()
                    self._last_position_bot_owned.clear()
                    self._pending_close_reasons.clear()
                    self._peak_profits.clear()
                    self._peak_profit_usd.clear()
                    self._trough_profits.clear()
                    self._trough_profit_usd.clear()
                    self._peak_state_loaded.clear()
                    self._peak_persisted_usd.clear()
                    self._trough_persisted_usd.clear()
                    self._profit_lock_tickets.clear()
                    self._profit_lock_levels.clear()
                    self._profit_retention_states.clear()
                    self._profit_retention_loaded.clear()
                    self._profit_retention_persisted.clear()
                    self._breakeven_tickets.clear()
                    self._protection_failures.clear()
                    self._initial_risk_pips.clear()
                    self._initial_risk_usd.clear()
                    self._baseline_error_tickets.clear()
                    self.last_scan_times.clear()
                    self.last_bar_times.clear()
                    self._entry_model_admissions.clear()
                    self.last_exit_bar_times.clear()
                    self._previous_trend_states.clear()
                    self._last_stale_bars.clear()
                    selection_task = getattr(
                        self, "market_selection_task", None
                    )
                    if (
                        selection_task is not None
                        and not selection_task.done()
                    ):
                        selection_task.cancel()
                    self.market_selection_task = None
                    self._pending_market_selection_account = None
                    self._market_selection_initialized = False
                    self._last_market_selection_monotonic = 0.0
                    self._market_rankings.clear()
                    self._broker_universe = []
                    self._broker_scan_symbols = ()
                    self._broker_universe_cursor = 0
                    self._broker_universe_refreshed = 0.0
                    self.analyzer = MarketAnalysisEngine()
                    self._forex_context_by_symbol.clear()
                    self._symbol_point_cache.clear()
                    self._selected_symbols = tuple(
                        self._market_candidates_for_current_market()[
                            : settings.dynamic_market_max_symbols
                        ]
                    )
                    for symbol in list(getattr(self, "_manual_trade_candidates", {})):
                        self._clear_manual_trade_candidate(symbol)
                    self.risk.reset_for_account()
                    await self._restore_daily_loss_reset(account)
                    await self._restore_loss_cooldown_reset(account)
                    await self._restore_loss_streak_reset(account)
                    self._history_ready = await self._reconcile_history(account)
                self._active_account_login = current_login
                self._active_account_mode = current_mode
                self._active_account_identity = current_identity
                dashboard_state.update_account(account)
                self.risk.update_peak_balance(account.get("balance", 0.0))

                # Fetch all open positions across the terminal
                def _get_all_positions():
                    return mt5.positions_get()

                raw_positions = await asyncio.to_thread(_get_all_positions)
                if raw_positions is None:
                    self._position_state_healthy = False
                    if self.entries_armed:
                        self.disarm_entries("MT5 position state is unavailable; entries locked")
                    dashboard_state.update_automation(scan_status="POSITION DATA ERROR")
                    now_mono = asyncio.get_running_loop().time()
                    if now_mono - self._last_positions_error_log_monotonic >= 30.0:
                        self.log(f"MT5 positions_get failed: {mt5.last_error()}", "ERROR")
                        self._last_positions_error_log_monotonic = now_mono
                    self._update_execution_readiness(account, override=(
                        "POSITION_DATA_ERROR", "MT5 position state is unavailable"
                    ))
                    await asyncio.sleep(2.0)
                    continue
                self._position_state_healthy = True
                positions = []
                current_tickets = []
                current_position_bot_owned: Dict[int, bool] = {}
                position_risk_healthy = True
                if raw_positions is not None:
                    for p in raw_positions:
                        symbol_info = await asyncio.to_thread(
                            mt5.symbol_info, p.symbol
                        )
                        point = symbol_info.point if symbol_info else 0.00001
                        digits = symbol_info.digits if symbol_info else 5
                        one_pip = point * 10 if digits in (3, 5) else point
                        if p.type == 0:  # BUY
                            profit_pips = (p.price_current - p.price_open) / one_pip
                        else:  # SELL
                            profit_pips = (p.price_open - p.price_current) / one_pip
                        risk_usd = 0.0
                        if p.sl:
                            try:
                                risk_result = await asyncio.to_thread(
                                    mt5.order_calc_profit,
                                    p.type,
                                    p.symbol,
                                    p.volume,
                                    p.price_open,
                                    p.sl,
                                )
                                risk_usd = downside_risk_usd(risk_result)
                            except ValueError:
                                position_risk_healthy = False
                        if p.tp:
                            reward_result = await asyncio.to_thread(
                                mt5.order_calc_profit,
                                p.type,
                                p.symbol,
                                p.volume,
                                p.price_open,
                                p.tp,
                            )
                            reward_usd = abs(reward_result or 0.0)
                        else:
                            reward_usd = 0.0
                        bot_owned = p.magic == settings.strategy_magic
                        managed = bot_owned or settings.manage_external_positions
                        initial_risk_pips = (
                            abs(p.price_open - p.sl) / one_pip if p.sl and one_pip > 0 else 0.0
                        )
                        if managed:
                            if not p.sl or symbol_info is None:
                                position_risk_healthy = False
                                initial_risk_pips = 0.0
                            else:
                                initial_risk_pips, baseline_ok = await self._initial_risk_for_position(
                                    account, p, symbol_info
                                )
                                if not baseline_ok:
                                    position_risk_healthy = False
                        execution_cost_usd = 0.0
                        if symbol_info is not None:
                            try:
                                execution_cost_usd = configured_execution_cost_usd(
                                    p.symbol, symbol_info, float(p.volume)
                                )
                            except (TypeError, ValueError, OverflowError):
                                execution_cost_usd = 0.0
                        estimated_net_profit_usd = (
                            float(p.profit)
                            + float(getattr(p, "swap", 0.0) or 0.0)
                            - execution_cost_usd
                        )
                        peak_pips, peak_profit_usd = (
                            await self._restore_and_update_position_peak(
                                account,
                                ticket=int(p.ticket),
                                profit_pips=profit_pips,
                                observed_profit_usd=float(p.profit),
                            )
                        )
                        now_epoch = datetime.now(timezone.utc).timestamp()
                        live_tick = await asyncio.to_thread(
                            mt5.symbol_info_tick, p.symbol
                        )
                        server_offset = infer_positive_server_offset_seconds(
                            float(getattr(live_tick, "time", 0.0) or 0.0),
                            now_epoch=now_epoch,
                        )
                        opened_epoch = normalized_broker_epoch(
                            float(p.time), server_offset
                        )
                        duration_min = max(
                            0.0, (now_epoch - opened_epoch) / 60.0
                        )
                        pos_dict = {
                            "ticket": p.ticket,
                            "symbol": p.symbol,
                            "type": p.type,
                            "volume": p.volume,
                            "price_open": p.price_open,
                            "price_current": p.price_current,
                            "sl": p.sl,
                            "tp": p.tp,
                            "profit": p.profit,
                            "estimated_net_profit_usd": estimated_net_profit_usd,
                            "profit_pips": profit_pips,
                            "peak_profit_pips": peak_pips,
                            "peak_profit_usd": peak_profit_usd,
                            "trough_profit_pips": self._trough_profits.get(
                                int(p.ticket), 0.0
                            ),
                            "trough_profit_usd": self._trough_profit_usd.get(
                                int(p.ticket), 0.0
                            ),
                            # This reflects a broker-verified modification, not
                            # merely a threshold crossing.
                            "profit_lock_armed": (
                                int(p.ticket) in self._profit_lock_tickets
                            ),
                            "profit_lock_floor_usd": self._profit_lock_levels.get(
                                int(p.ticket), 0.0
                            ),
                            "profit_retention_floor_usd": getattr(
                                self, "_profit_retention_states", {}
                            ).get(int(p.ticket), RetentionState()).floor_usd,
                            "initial_risk_pips": initial_risk_pips,
                            "initial_risk_usd": getattr(
                                self, "_initial_risk_usd", {}
                            ).get(
                                int(p.ticket), 0.0
                            ),
                            "duration_min": duration_min,
                            "magic": p.magic,
                            "comment": p.comment,
                            "bot_owned": bot_owned,
                            "risk_to_sl_usd": risk_usd,
                            "risk_pct_balance": risk_usd / account["balance"] * 100.0 if account["balance"] else 0.0,
                            "reward_to_tp_usd": reward_usd,
                            "planned_rr": reward_usd / risk_usd if risk_usd else 0.0,
                        }
                        positions.append(pos_dict)
                        current_tickets.append(p.ticket)
                        current_position_bot_owned[int(p.ticket)] = bot_owned

                risk_health_changed = self._position_risk_healthy != position_risk_healthy
                self._position_risk_healthy = position_risk_healthy
                if not position_risk_healthy:
                    dashboard_state.update_automation(scan_status="POSITION STOP RISK UNAVAILABLE")
                    if self.entries_armed:
                        self.disarm_entries(
                            "MT5 could not calculate an open position's stop risk; entries locked"
                        )
                    elif risk_health_changed:
                        self.log(
                            "MT5 could not calculate an open position's stop risk; "
                            "execution readiness is blocked.",
                            "ERROR",
                        )

                dashboard_state.update_positions(positions)

                # Track position closures to record in dashboard history
                for ticket in self._last_positions_tickets:
                    if ticket not in current_tickets:
                        # Fetch the final profit from MT5 deal history matching this ticket
                        deals = await asyncio.to_thread(
                            mt5.history_deals_get, position=ticket
                        )
                        if deals is None:
                            self.log(
                                f"Position {ticket} disappeared but deal history is unavailable; "
                                "deferring outcome reconciliation.",
                                "ERROR",
                            )
                            self._history_ready = False
                            continue
                        profit_usd = sum(
                            float(d.profit) + float(d.commission) + float(d.swap) + float(d.fee)
                            for d in deals
                        )
                        
                        if self._last_position_bot_owned.get(int(ticket), False):
                            self.log(
                                f"Strategy position ticket {ticket} closed. "
                                f"Realized Profit: ${profit_usd:.2f}"
                            )
                            close_reason = self._pending_close_reasons.pop(
                                int(ticket), ""
                            )
                            await self._update_replay_outcome(
                                ticket,
                                profit_usd,
                                close_reason=close_reason,
                            )
                            self.risk.record_trade_closed(
                                profit_usd,
                                float(account.get("balance", 0.0)),
                                strategy_owned=True,
                            )
                        else:
                            self.log(
                                f"External/manual position ticket {ticket} closed. "
                                f"Realized Profit: ${profit_usd:.2f}; excluded from "
                                "strategy daily-loss accounting."
                            )
                        self._forget_position_runtime_state(ticket)
                self._last_positions_tickets = current_tickets
                self._last_position_bot_owned = current_position_bot_owned

                now_mono = asyncio.get_running_loop().time()
                if now_mono - self._last_reconcile_monotonic >= settings.reconcile_interval_seconds:
                    self._history_ready = await self._reconcile_history(account)
                    self._last_reconcile_monotonic = now_mono

                if dashboard_state.account.portfolio_risk_pct > settings.max_portfolio_risk_pct:
                    if self.entries_armed:
                        self.disarm_entries("Portfolio stop risk exceeds configured limit")
                    else:
                        dashboard_state.update_automation(
                            safety_reason="Portfolio stop risk exceeds configured limit"
                        )

                # The dedicated priority loop owns all broker protection
                # mutations. This loop only publishes the enriched position
                # state and schedules completed-candle exit analysis.
                managed_positions = [
                    p for p in positions
                    if p["bot_owned"] or settings.manage_external_positions
                ]
                self._touch_engine_heartbeat(broker_poll=True)
                self._update_execution_readiness(account)

                # Position exits have their own completed-M1 review lane. It is
                # scheduled before new-entry discovery so an open trade reaches
                # the bounded model queue first when M1 and M5 close together.
                managed_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
                for managed_position in managed_positions:
                    managed_by_symbol.setdefault(
                        str(managed_position["symbol"]).upper(), []
                    ).append(managed_position)
                if settings.fast_exit_review_enabled:
                    for managed_symbol, symbol_positions in managed_by_symbol.items():
                        existing_exit = self.position_exit_tasks.get(
                            managed_symbol
                        )
                        if existing_exit is not None and not existing_exit.done():
                            continue
                        exit_task = asyncio.create_task(
                            self._evaluate_position_exit(
                                managed_symbol,
                                dict(account),
                                [dict(item) for item in symbol_positions],
                            )
                        )
                        self.position_exit_tasks[managed_symbol] = exit_task
                        exit_task.add_done_callback(
                            lambda finished, active_symbol=managed_symbol: self._position_exit_finished(
                                active_symbol, finished
                            )
                        )

                # 4. Process the currently selected capital-fit symbols.
                # Open positions always receive first access to the completed-
                # candle decision queue, even when dynamic market ranking has
                # rotated their symbols out of the new-entry universe.
                symbols_to_scan = self._symbols_with_position_priority(positions)
                self._set_active_scan_symbols(symbols_to_scan)

                for symbol in symbols_to_scan:
                    if not self.is_running:
                        break
                    if (
                        settings.fast_exit_review_enabled
                        and symbol.upper() in managed_by_symbol
                    ):
                        continue
                    existing = self.analysis_tasks.get(symbol)
                    if existing is not None and not existing.done():
                        continue
                    task = asyncio.create_task(
                        self._evaluate_symbol(
                            symbol,
                            dict(account),
                            [dict(position) for position in positions],
                        )
                    )
                    self.analysis_tasks[symbol] = task
                    task.add_done_callback(
                        lambda finished, active_symbol=symbol: self._analysis_finished(
                            active_symbol, finished
                        )
                    )

                # 5. Refresh the wider broker/capital-aware market ranking in
                # the background. Entry tasks are created first so discovery
                # I/O cannot consume the fresh-candle execution window.
                self._request_market_selection_refresh(dict(account))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log(f"Error in engine main loop: {e}", "ERROR")

            # Poll quickly enough to begin a completed-candle decision without
            # introducing a multi-second scheduling delay. Candle freshness
            # checks still prevent any inference on an incomplete M5 bar.
            await asyncio.sleep(settings.decision_poll_seconds)

    def _analysis_finished(self, symbol: str, task: asyncio.Task) -> None:
        if self.analysis_tasks.get(symbol) is task:
            self.analysis_tasks.pop(symbol, None)
        if task.cancelled():
            self._start_pending_market_selection_if_idle()
            return
        error = task.exception()
        if error is not None:
            self.log(f"Analysis task failed for {symbol}: {error}", "ERROR")
            dashboard_state.update_symbol_decision(
                symbol, stage="ERROR", gate_reason=str(error)
            )
        self._start_pending_market_selection_if_idle()

    def _market_selection_finished(self, task: asyncio.Task) -> None:
        if getattr(self, "market_selection_task", None) is task:
            self.market_selection_task = None
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.log(
                f"Adaptive market refresh failed: {error}",
                "ERROR",
            )
            dashboard_state.update_automation(
                scan_status="MARKET DISCOVERY ERROR"
            )

    def _position_exit_finished(self, symbol: str, task: asyncio.Task) -> None:
        if self.position_exit_tasks.get(symbol) is task:
            self.position_exit_tasks.pop(symbol, None)
        if task.cancelled():
            self._start_pending_market_selection_if_idle()
            return
        error = task.exception()
        if error is not None:
            self.log(
                f"Fast position review failed for {symbol}: {error}", "ERROR"
            )
            dashboard_state.update_symbol_decision(
                symbol, stage="EXIT REVIEW ERROR", gate_reason=str(error)
            )
        self._start_pending_market_selection_if_idle()

    def _update_execution_readiness(
        self,
        account: Optional[Dict[str, Any]],
        override: Optional[Tuple[str, str]] = None,
    ) -> None:
        checks: List[Dict[str, Any]] = []

        def add(code: str, label: str, ok: bool, detail: str) -> None:
            checks.append({"code": code, "label": label, "ok": bool(ok), "detail": detail})

        add("ENGINE", "Decision engine", self.is_running,
            "Running" if self.is_running else "Stopped")
        progress_ok, progress_detail = self._engine_progress_health()
        add(
            "LOOP",
            "Broker-position loop",
            progress_ok,
            progress_detail,
        )
        protection_ok, protection_detail = self._protection_progress_health()
        add(
            "PROTECTION",
            "Broker protection supervisor",
            protection_ok,
            protection_detail,
        )
        verified = bool(account and str(account.get("trade_mode_name", "UNKNOWN")) in {"DEMO", "LIVE"})
        add("ACCOUNT", "MT5 account", verified,
            str(account.get("trade_mode_name")) if account else "Unavailable")
        add("HISTORY", "Broker history", self._history_ready,
            "Reconciled" if self._history_ready else "Not reconciled")
        connected = bool(account and account.get("terminal_connected"))
        add("CONNECTION", "Pepperstone connection", connected,
            "Connected" if connected else "Disconnected")
        permissions = bool(
            settings.dry_run
            or (
                account
                and account.get("account_trade_allowed")
                and account.get("expert_trading_allowed")
                and account.get("terminal_trade_allowed")
                and not account.get("tradeapi_disabled", True)
            )
        )
        add("PERMISSIONS", "MT5 trade permissions", permissions,
            "Enabled" if permissions else "Algo/Python trading is disabled")
        llm_ready = bool(
            dashboard_state.automation.llm_online
            and getattr(
                self, "_decision_provider_inference_ready", True
            )
        )
        add("LLM", "Decision provider", llm_ready,
            "Inference ready" if llm_ready else "Unavailable or warming up")
        position_state_ok = bool(
            getattr(self, "_position_state_healthy", True)
        )
        add(
            "POSITIONS",
            "Broker position state",
            position_state_ok,
            "Available" if position_state_ok else "Unavailable from MT5",
        )
        stop_risk_ok = bool(self._position_risk_healthy)
        add("STOP_RISK", "Open-position stop risk", stop_risk_ok,
            "Calculated" if stop_risk_ok else "Unavailable from MT5")
        if account:
            daily_loss_ok, daily_loss_detail = self.risk.daily_loss_status(account)
        else:
            daily_loss_ok, daily_loss_detail = False, "Account balance is unavailable"
        add("DAILY_LOSS", "UTC daily loss stop", daily_loss_ok, daily_loss_detail)
        identity_ok = bool(
            self.entries_armed and account and self._same_account(account, self._armed_account_identity)
        )
        add("ARMED", "Entry authorization", identity_ok,
            "Bound to active account" if identity_ok else "Disarmed or account changed")
        portfolio_ok = dashboard_state.account.portfolio_risk_pct <= settings.max_portfolio_risk_pct
        add("PORTFOLIO", "Portfolio stop risk", portfolio_ok,
            f"{dashboard_state.account.portfolio_risk_pct:.2f}% / {settings.max_portfolio_risk_pct:.2f}%")

        failed = next((item for item in checks if not item["ok"]), None)
        if override:
            code, reason = override
            ready = False
        elif failed:
            code = str(failed["code"])
            reason = str(failed["detail"])
            ready = False
        else:
            code = "READY"
            reason = "Broker channel is ready; every setup still needs symbol-level risk approval."
            ready = True
        dashboard_state.update_readiness(
            ready=ready,
            severity="READY" if ready else "BLOCKED",
            code=code,
            reason=reason,
            checks=checks,
        )

    async def _tick_loop(self) -> None:
        """High-frequency tick polling loop for prices and live tick stream."""
        last_ticks = {}
        while self.is_running:
            try:
                if not await self.conn.is_connected():
                    await asyncio.sleep(1.0)
                    continue

                # Market Watch covers the entire current broker candidate
                # universe. Only `_symbols_for_current_market()` proceeds to
                # expensive analysis/model decisions and possible execution.
                symbols_to_scan = self._candidate_universe()

                point_cache = getattr(self, "_symbol_point_cache", None)
                if point_cache is None:
                    point_cache = {}
                    self._symbol_point_cache = point_cache
                for symbol in symbols_to_scan:
                    tick = await self.reader.get_live_tick(
                        symbol, assume_connected=True
                    )
                    if tick:
                        key = symbol
                        last_t = last_ticks.get(key)
                        if last_t != tick["time_msc"]:
                            last_ticks[key] = tick["time_msc"]
                            point = point_cache.get(symbol.upper())
                            if point is None:
                                symbol_info = await asyncio.to_thread(
                                    mt5.symbol_info, symbol
                                )
                                point = (
                                    float(symbol_info.point)
                                    if symbol_info
                                    and float(symbol_info.point) > 0
                                    else 0.00001
                                )
                                point_cache[symbol.upper()] = point
                            
                            # Retrieve trend from current dashboard state instead of recalculating
                            existing_price = dashboard_state.prices.get(symbol)
                            temp_trend = existing_price.trend if existing_price else "NEUTRAL"
                            adx_val = existing_price.adx if existing_price else 25.0
                                    
                            dashboard_state.update_prices(
                                symbol=symbol,
                                bid=tick["bid"],
                                ask=tick["ask"],
                                point=point,
                                trend=temp_trend,
                                adx=adx_val
                            )
                            dashboard_state.add_tick(symbol, tick["bid"], tick["ask"])
            except Exception as exc:
                now_mono = asyncio.get_running_loop().time()
                if now_mono - self._last_tick_error_log_monotonic >= 30.0:
                    self.log(f"Live tick stream error: {exc}", "WARNING")
                    self._last_tick_error_log_monotonic = now_mono
            await asyncio.sleep(0.15)  # 150 ms check interval

    async def _entry_bar_is_current(
        self, symbol: str, analyzed_completed_bar: str
    ) -> Tuple[bool, str]:
        """Fail closed if an entry decision no longer belongs to the latest M5 bar."""
        refreshed = await self.reader.get_ohlcv(symbol, "M5", count=3)
        if refreshed is None or refreshed.empty:
            return False, "M5 candle refresh failed immediately before execution"
        if bool(refreshed.attrs.get("is_stale", False)):
            return False, "Latest completed M5 candle is stale immediately before execution"

        latest_completed_bar = str(refreshed.iloc[-1].get("time"))
        if latest_completed_bar != analyzed_completed_bar:
            return (
                False,
                f"Decision expired: analyzed M5 bar {analyzed_completed_bar}; "
                f"latest completed bar is {latest_completed_bar}",
            )
        try:
            bar_age = float(
                refreshed.attrs.get("age_after_close_seconds", math.inf)
            )
        except (TypeError, ValueError):
            bar_age = math.inf
        if (
            not math.isfinite(bar_age)
            or bar_age > settings.max_entry_bar_age_seconds
        ):
            return (
                False,
                (
                    f"Decision expired: completed M5 bar is {bar_age:.1f}s old; "
                    f"maximum is {settings.max_entry_bar_age_seconds:.0f}s"
                    if math.isfinite(bar_age)
                    else "Decision expired: completed M5 bar age is unavailable"
                ),
            )
        return True, ""

    @staticmethod
    def _confirmed_adverse_reversal(
        analyses: Dict[str, Dict[str, Any]],
        position_type: int,
    ) -> Tuple[bool, str]:
        """Confirm an adverse reversal without relying on model inference.

        A completed M1 structural event alone can be a brief pullback. The
        shortcut therefore also requires either matching completed-M5
        structure or the M5 fast trend to have turned in the same adverse
        direction. Ambiguous cases continue through the normal model review.
        """
        position_side = "BUY" if int(position_type) == 0 else "SELL"
        adverse_action = "SELL" if position_side == "BUY" else "BUY"
        adverse_direction = (
            "BEARISH" if adverse_action == "SELL" else "BULLISH"
        )
        evidence_ids = build_evidence_ids(analyses)
        m1_trigger = has_directional_trigger(
            evidence_ids, adverse_action, timeframes=("M1",)
        )
        if not m1_trigger:
            return False, ""

        m5_trigger = has_directional_trigger(
            evidence_ids, adverse_action, timeframes=("M5",)
        )
        m5_structure = (
            (analyses.get("M5") or {}).get("market_structure", {}) or {}
        )
        m5_fast_direction = str(
            m5_structure.get("fast_trend")
            or m5_structure.get("trend_state_direction")
            or "NEUTRAL"
        ).upper()
        if not (m5_trigger or m5_fast_direction == adverse_direction):
            return False, ""

        adverse_ids = [
            identifier
            for identifier in evidence_ids
            if (
                identifier.startswith("M1_")
                and adverse_direction in identifier
                and any(
                    marker in identifier
                    for marker in ("_BOS_", "_CHOCH_", "_BREAKOUT_")
                )
            )
            or (
                identifier.startswith("M5_")
                and adverse_direction in identifier
                and any(
                    marker in identifier
                    for marker in ("_BOS_", "_CHOCH_", "_BREAKOUT_")
                )
            )
        ]
        confirmation = (
            ", ".join(adverse_ids[:3])
            or f"M1 structure + M5 fast trend {adverse_direction}"
        )
        return (
            True,
            f"Confirmed adverse reversal against {position_side}: "
            f"{confirmation}",
        )

    def _confirmed_adverse_momentum(
        self,
        analyses: Dict[str, Dict[str, Any]],
        position: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Confirm persistent adverse M1 momentum before the full stop.

        This is deliberately narrower than the model exit lane: it requires
        two or more completed M1 confirmations and a broker-observed loss of a
        configured fraction of initial risk. It therefore cannot close a
        profitable trade or react to a single noisy pullback.
        """
        ticket = int(position.get("ticket", 0) or 0)
        if not hasattr(self, "_adverse_momentum_streaks"):
            self._adverse_momentum_streaks = {}
        if not settings.momentum_deterioration_exit_enabled or ticket <= 0:
            self._adverse_momentum_streaks.pop(ticket, None)
            return False, ""

        m1 = analyses.get("M1") or {}
        structure = m1.get("market_structure", {}) or {}
        indicators = m1.get("indicators", {}) or {}
        position_side = (
            "BUY" if int(position.get("type", -1)) == 0 else "SELL"
        )
        adverse_direction = "BEARISH" if position_side == "BUY" else "BULLISH"
        m1_direction = str(
            structure.get("fast_trend")
            or structure.get("trend_state_direction")
            or structure.get("trend")
            or "NEUTRAL"
        ).upper()

        try:
            rsi = float(indicators.get("rsi_14"))
        except (TypeError, ValueError):
            rsi = math.nan
        try:
            macd_diff = float((indicators.get("macd") or {}).get("diff"))
        except (TypeError, ValueError):
            macd_diff = math.nan
        adverse_bodies = []
        direction_multiplier = 1.0 if position_side == "SELL" else -1.0
        for key in ("candle_body_atr_signed", "candle_return_atr"):
            try:
                value = float(indicators.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                adverse_bodies.append(direction_multiplier * value)
        adverse_body_atr = max(adverse_bodies) if adverse_bodies else 0.0

        if position_side == "BUY":
            oscillator_adverse = math.isfinite(rsi) and rsi <= 48.0
            macd_adverse = math.isfinite(macd_diff) and macd_diff < 0.0
        else:
            oscillator_adverse = math.isfinite(rsi) and rsi >= 52.0
            macd_adverse = math.isfinite(macd_diff) and macd_diff > 0.0
        body_adverse = (
            adverse_body_atr
            >= settings.momentum_deterioration_min_body_atr
        )
        signal = bool(
            m1_direction == adverse_direction
            and macd_adverse
            and (oscillator_adverse or body_adverse)
        )
        if signal:
            streak = self._adverse_momentum_streaks.get(ticket, 0) + 1
            self._adverse_momentum_streaks[ticket] = streak
        else:
            self._adverse_momentum_streaks.pop(ticket, None)
            return False, ""

        try:
            initial_risk_pips = float(
                position.get("initial_risk_pips", 0.0) or 0.0
            )
            profit_pips = float(position.get("profit_pips", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False, ""
        if not (
            math.isfinite(initial_risk_pips)
            and initial_risk_pips > 0
            and math.isfinite(profit_pips)
        ):
            return False, ""
        live_r = profit_pips / initial_risk_pips
        required_bars = settings.momentum_deterioration_confirm_bars
        loss_threshold = settings.momentum_deterioration_min_loss_r
        if streak < required_bars or live_r > -loss_threshold:
            return False, ""
        return (
            True,
            f"Persistent adverse M1 momentum against {position_side}: "
            f"{streak} completed bars, RSI {rsi:.1f}, MACD "
            f"{macd_diff:+.5f}, live {live_r:+.2f} R.",
        )

    async def _evaluate_position_exit(
        self,
        symbol: str,
        account_info: Dict[str, Any],
        symbol_positions: List[Dict[str, Any]],
    ) -> None:
        """Review a managed position from fresh completed M1/M5 evidence.

        This lane is exit-only. It cannot open or reverse a position, and the
        deterministic tick-level protections continue running while inference
        is in flight.
        """
        if not symbol_positions:
            return
        timeframe = str(settings.position_exit_review_timeframe).upper()
        if timeframe not in {"M1", "M5"}:
            timeframe = "M1"
        history_bars = max(220, settings.analysis_history_bars)
        exit_frame = await self.reader.get_ohlcv(
            symbol, timeframe, count=history_bars
        )
        if (
            exit_frame is None
            or exit_frame.empty
            or bool(exit_frame.attrs.get("is_stale", False))
        ):
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT DATA WAIT",
                gate_reason=f"Completed {timeframe} exit data is unavailable or stale",
            )
            return
        completed_exit_bar = str(exit_frame.iloc[-1].get("time"))
        if self.last_exit_bar_times.get(symbol) == completed_exit_bar:
            return

        dashboard_state.update_symbol_decision(
            symbol,
            stage=f"{timeframe} EXIT ANALYSIS",
            candle_time=completed_exit_bar,
            gate_reason="",
        )
        self.log(
            f"Fresh {timeframe} exit review for {symbol}: "
            f"{completed_exit_bar}."
        )

        m5_frame, m15_frame, h1_frame, h4_frame = await asyncio.gather(
            self.reader.get_ohlcv(symbol, "M5", count=history_bars),
            self.reader.get_ohlcv(symbol, "M15", count=history_bars),
            self.reader.get_ohlcv(symbol, "H1", count=history_bars),
            self.reader.get_ohlcv(symbol, "H4", count=history_bars),
        )
        if (
            m5_frame is None
            or m5_frame.empty
            or bool(m5_frame.attrs.get("is_stale", False))
        ):
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT DATA WAIT",
                gate_reason="Fresh completed M5 exit confirmation is unavailable",
            )
            return

        def analyze_optional(
            timeframe_name: str, frame: Any
        ) -> Optional[Dict[str, Any]]:
            if (
                frame is None
                or frame.empty
                or bool(frame.attrs.get("is_stale", False))
            ):
                return None
            return self.analyzer.analyze(symbol, timeframe_name, frame)

        m1_analysis = (
            self.analyzer.analyze(symbol, "M1", exit_frame)
            if timeframe == "M1"
            else None
        )
        m5_analysis = self.analyzer.analyze(symbol, "M5", m5_frame)
        m15_analysis = analyze_optional("M15", m15_frame)
        h1_analysis = analyze_optional("H1", h1_frame)
        h4_analysis = analyze_optional("H4", h4_frame)
        if not m5_analysis:
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT DATA WAIT",
                gate_reason="M5 exit indicator warm-up is incomplete",
            )
            return
        if timeframe == "M1" and not m1_analysis:
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT DATA WAIT",
                gate_reason="M1 exit indicator warm-up is incomplete",
            )
            return

        live_tick = await self.reader.get_live_tick(symbol)
        fresh_positions: List[Dict[str, Any]] = []
        for snapshot in symbol_positions:
            ticket = int(snapshot.get("ticket", 0) or 0)
            rows = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
            if rows is None or not rows:
                continue
            raw = rows[0]
            if str(raw.symbol).upper() != symbol.upper():
                continue
            updated = dict(snapshot)
            updated.update(
                type=int(raw.type),
                volume=float(raw.volume),
                price_open=float(raw.price_open),
                price_current=float(raw.price_current),
                sl=float(raw.sl or 0.0),
                tp=float(raw.tp or 0.0),
                profit=float(raw.profit),
            )
            fresh_positions.append(updated)
        if not fresh_positions:
            return
        # Mark the bar consumed only after the mandatory M1/M5 data and exact
        # broker position have been refreshed. Transient data failures can
        # therefore retry without waiting for another minute.
        self.last_exit_bar_times[symbol] = completed_exit_bar

        history = await self.db.get_closed_positions(
            symbol=mt5.broker_symbol_name(symbol),
            limit=20,
            account_login=self._active_account_login,
        )
        analyses = {
            "M1": m1_analysis or {},
            "M5": m5_analysis,
            "M15": m15_analysis,
            "H1": h1_analysis,
            "H4": h4_analysis,
        }
        allowed_evidence_ids = build_evidence_ids(analyses)

        # A confirmed completed-M1 reversal with matching fast-M5 direction is
        # an exit fact, not a price prediction. Execute it before model
        # inference so a slow or unavailable model cannot delay loss control.
        confirmed_reversals = []
        for position in fresh_positions:
            confirmed, reversal_reason = self._confirmed_adverse_reversal(
                analyses, int(position.get("type", -1))
            )
            if not confirmed:
                confirmed, reversal_reason = self._confirmed_adverse_momentum(
                    analyses, position
                )
            if confirmed:
                confirmed_reversals.append((position, reversal_reason))
        if confirmed_reversals:
            async with self._execution_lock:
                fresh_account = await self.conn.get_account_info()
                if not fresh_account or not self._same_account(
                    fresh_account, self._account_identity(account_info)
                ):
                    self.disarm_entries(
                        "Account changed during deterministic reversal exit"
                    )
                    return
                for position, reversal_reason in confirmed_reversals:
                    ticket = int(position.get("ticket", 0) or 0)
                    broker_positions = await asyncio.to_thread(
                        mt5.positions_get, ticket=ticket
                    )
                    if broker_positions is None or not broker_positions:
                        continue
                    broker_position = broker_positions[0]
                    if (
                        str(broker_position.symbol).upper() != symbol.upper()
                        or (
                            int(broker_position.magic) != settings.strategy_magic
                            and not settings.manage_external_positions
                        )
                    ):
                        continue
                    result = await self.executor.close_position(
                        ticket,
                        expected_account=self._account_identity(fresh_account),
                    )
                    if result.success:
                        await self._remember_strategy_close_reason(
                            ticket,
                            f"DETERMINISTIC_{timeframe.upper()}_REVERSAL",
                        )
                        self.log(
                            f"Deterministic {timeframe} reversal exit closed "
                            f"ticket {ticket} ({symbol}): {reversal_reason}"
                        )
                        dashboard_state.update_symbol_decision(
                            symbol,
                            stage="REVERSAL CLOSE SUBMITTED",
                            action="CLOSE",
                            confidence=1.0,
                            reasoning=reversal_reason,
                            trade_management=(
                                "Closed from deterministic completed-candle "
                                "adverse structure or persistent momentum."
                            ),
                            candle_time=completed_exit_bar,
                        )
                    else:
                        self.log(
                            f"Deterministic reversal exit failed for ticket "
                            f"{ticket}: {result.error}",
                            "ERROR",
                        )
                        dashboard_state.update_symbol_decision(
                            symbol,
                            stage="EXECUTION ERROR",
                            gate_reason=(
                                result.error
                                or "Deterministic reversal exit failed"
                            ),
                        )
            return

        if not settings.exit_model_confirmation_enabled:
            dashboard_state.update_symbol_decision(
                symbol,
                stage=f"MONITORING {timeframe}",
                action="HOLD",
                confidence=0.0,
                reasoning=(
                    "No deterministic adverse reversal is confirmed on the "
                    f"completed {timeframe}/M5 evidence."
                ),
                trade_management=(
                    "Broker SL/TP, adaptive profit floor, break-even, trailing, "
                    "and deterministic reversal protection remain active."
                ),
                candle_time=completed_exit_bar,
                gate_reason="",
            )
            return

        decision_context = {
            "symbol": symbol,
            "completed_bar": completed_exit_bar,
            "exit_timeframe": timeframe,
            "has_open_position": True,
            "open_positions": fresh_positions,
            "analyses": analyses,
            "live_tick": live_tick or {},
        }
        if getattr(self.llm, "provider_name", "") == "deterministic":
            system_prompt = "deterministic-rules-v2-adaptive-exit"
            user_prompt = (
                f"{symbol} completed {timeframe} exit bar "
                f"{completed_exit_bar}"
            )
        else:
            system_prompt, user_prompt = PromptGenerator.generate(
                symbol=symbol,
                timeframe=timeframe,
                analysis_data=m5_analysis,
                m15_analysis=m15_analysis,
                h1_analysis=h1_analysis,
                h4_analysis=h4_analysis,
                account_info=account_info,
                open_positions=fresh_positions,
                trade_history=history,
                calendar_events=None,
                fast_exit_review=True,
                m1_analysis=m1_analysis,
                live_tick=live_tick,
            )

        dashboard_state.update_symbol_decision(
            symbol,
            stage=f"{timeframe} EXIT INFERENCE",
            candle_time=completed_exit_bar,
        )
        started = asyncio.get_running_loop().time()
        async with self._decision_semaphore:
            if hasattr(self.llm, "request_decision"):
                response = await self.llm.request_decision(
                    system_prompt, user_prompt, context=decision_context
                )
                raw_decision = response.decision
                telemetry = response.telemetry.to_dict()
            else:
                raw_decision = await self.llm.get_trading_decision(
                    system_prompt, user_prompt
                )
                telemetry = {
                    "trace_id": "",
                    "provider": getattr(self.llm, "provider_name", "legacy"),
                    "model": getattr(
                        self.llm, "model_name", settings.decision_model
                    ),
                    "latency_seconds": 0.0,
                    "success": bool(raw_decision),
                    "error": getattr(self.llm, "last_error", ""),
                    "prompt_tokens_estimate": len(
                        system_prompt + user_prompt
                    ) // 4,
                }
        elapsed = asyncio.get_running_loop().time() - started
        inference_latency = self._reported_inference_latency(
            telemetry, elapsed
        )
        position_side = (
            "BUY" if int(fresh_positions[0].get("type", -1)) == 0 else "SELL"
        )
        valid, decision, validation_error = DecisionValidator.validate_decision(
            raw_decision,
            allowed_evidence_ids=allowed_evidence_ids,
            close_position_side=position_side,
            close_trigger_timeframes=("M1", "M5"),
        )
        if telemetry.get("trace_id"):
            await asyncio.to_thread(
                self.replay_logger.log_decision_trace,
                symbol=symbol,
                candle_time=f"{timeframe}:{completed_exit_bar}",
                telemetry=telemetry,
                decision=raw_decision,
                validation_status=(
                    "VALID" if valid else f"INVALID: {validation_error}"
                ),
            )
        if not valid or decision is None:
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT MODEL ERROR",
                inference_time_s=round(inference_latency, 2),
                candle_time=completed_exit_bar,
                gate_reason=validation_error,
            )
            self.log(
                f"Exit decision validation rejected for {symbol}: "
                f"{validation_error}",
                "WARNING",
            )
            return

        action = str(decision.get("action", "HOLD")).upper()
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        reasoning = str(decision.get("reasoning", ""))
        management = str(decision.get("trade_management", ""))
        dashboard_state.update_llm(
            action=action,
            symbol=symbol,
            confidence=confidence,
            reasoning=reasoning,
            trade_management=management,
            inference_time_s=inference_latency,
            prompt_tokens=int(
                telemetry.get("prompt_tokens_estimate")
                or len(system_prompt + user_prompt) // 4
            ),
            provider=str(
                telemetry.get("provider") or settings.llm_provider
            ),
            model=str(telemetry.get("model") or settings.decision_model),
            trace_id=str(telemetry.get("trace_id") or ""),
        )
        if action == "HOLD":
            dashboard_state.update_symbol_decision(
                symbol,
                stage=f"MONITORING {timeframe}",
                action="HOLD",
                confidence=confidence,
                reasoning=reasoning,
                trade_management=management,
                inference_time_s=round(inference_latency, 2),
                candle_time=completed_exit_bar,
                gate_reason="",
            )
            return
        if action != "CLOSE":
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT REJECTED",
                action=action,
                gate_reason="Fast exit review is limited to HOLD or CLOSE",
            )
            return
        if confidence < settings.confidence_threshold:
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT REJECTED",
                action=action,
                confidence=confidence,
                gate_reason=(
                    f"Close confidence {confidence:.0%} is below "
                    f"{settings.confidence_threshold:.0%}"
                ),
            )
            return

        ticket = int(decision.get("ticket_to_close") or 0)
        allowed_tickets = {
            int(position["ticket"]) for position in fresh_positions
        }
        if ticket not in allowed_tickets:
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT REJECTED",
                gate_reason="Model selected an unmanaged or unknown ticket",
            )
            return

        latest_exit_frame = await self.reader.get_ohlcv(
            symbol, timeframe, count=3
        )
        if (
            latest_exit_frame is None
            or latest_exit_frame.empty
            or bool(latest_exit_frame.attrs.get("is_stale", False))
            or str(latest_exit_frame.iloc[-1].get("time"))
            != completed_exit_bar
        ):
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXIT EXPIRED",
                gate_reason=(
                    f"{timeframe} evidence changed during model inference"
                ),
            )
            return

        async with self._execution_lock:
            fresh_account = await self.conn.get_account_info()
            if not fresh_account or not self._same_account(
                fresh_account, self._account_identity(account_info)
            ):
                self.disarm_entries("Account changed during exit inference")
                return
            broker_positions = await asyncio.to_thread(
                mt5.positions_get, ticket=ticket
            )
            if broker_positions is None or not broker_positions:
                return
            broker_position = broker_positions[0]
            if (
                str(broker_position.symbol).upper() != symbol.upper()
                or (
                    int(broker_position.magic) != settings.strategy_magic
                    and not settings.manage_external_positions
                )
            ):
                return
            result = await self.executor.close_position(
                ticket,
                expected_account=self._account_identity(fresh_account),
            )
        if result.success:
            await self._remember_strategy_close_reason(
                ticket,
                f"FAST_{timeframe.upper()}_MODEL_EXIT",
            )
            self.log(
                f"Fast {timeframe} exit closed ticket {ticket} ({symbol}): "
                f"{reasoning}"
            )
            dashboard_state.update_symbol_decision(
                symbol,
                stage="CLOSE SUBMITTED",
                action="CLOSE",
                confidence=confidence,
                reasoning=reasoning,
                trade_management=management,
                inference_time_s=round(inference_latency, 2),
                candle_time=completed_exit_bar,
            )
        else:
            self.log(
                f"Fast exit failed for ticket {ticket}: {result.error}",
                "ERROR",
            )
            dashboard_state.update_symbol_decision(
                symbol,
                stage="EXECUTION ERROR",
                gate_reason=result.error or "Fast exit failed",
            )

    async def _evaluate_symbol(
        self, symbol: str, account_info: Dict[str, Any], open_positions: List[Dict[str, Any]]
    ) -> None:
        """Evaluate one completed M5 candle and, if approved, submit atomically."""
        now = datetime.now()
        last_scan = self.last_scan_times.get(symbol)
        if last_scan and (now - last_scan).total_seconds() < settings.analysis_interval_seconds:
            return
        self.last_scan_times[symbol] = now
        symbol_positions = [
            p for p in open_positions
            if p["symbol"].upper() == symbol.upper()
            and (p.get("bot_owned") or settings.manage_external_positions)
        ]
        has_open_position = bool(symbol_positions)
        if is_weekend():
            if not settings.weekend_trading_enabled and not settings.crypto_only_on_weekend:
                return
            if symbol.upper() not in [s.upper() for s in settings.weekend_symbols]:
                return

        history_bars = max(220, settings.analysis_history_bars)
        df_m5 = await self.reader.get_ohlcv(symbol, "M5", count=history_bars)
        if df_m5 is None or df_m5.empty:
            self._set_symbol_scan_state(symbol, "DATA_UNAVAILABLE")
            dashboard_state.update_symbol_decision(
                symbol, stage="DATA ERROR", gate_reason="M5 candle history is unavailable"
            )
            self.log(f"Could not retrieve M5 candles for {symbol}. Skipping...", "WARNING")
            return
        completed_bar = str(df_m5.iloc[-1].get("time"))
        if bool(df_m5.attrs.get("is_stale", False)):
            self._set_symbol_scan_state(symbol, "STALE")
            dashboard_state.update_symbol_decision(
                symbol,
                stage="STALE DATA",
                candle_time=completed_bar,
                gate_reason="Latest completed M5 candle is stale",
            )
            if self._last_stale_bars.get(symbol) != completed_bar:
                age_minutes = float(df_m5.attrs.get("age_after_close_seconds", 0.0)) / 60.0
                self.log(
                    f"Stale M5 candle for {symbol}: {completed_bar} "
                    f"({age_minutes:.1f} minutes after expected close). LLM scan skipped.",
                    "WARNING",
                )
                self._last_stale_bars[symbol] = completed_bar
            return
        self._last_stale_bars.pop(symbol, None)
        try:
            completed_bar_age = float(
                df_m5.attrs.get("age_after_close_seconds", math.inf)
            )
        except (TypeError, ValueError):
            completed_bar_age = math.inf
        if self.last_bar_times.get(symbol) == completed_bar:
            logger.debug(
                "Waiting for next completed M5 candle for %s (latest=%s).",
                symbol,
                completed_bar,
            )
            self._set_symbol_scan_state(symbol, "WAITING")
            # Preserve the completed candle's terminal outcome (including a
            # rejection reason) until a genuinely new bar starts analysis.
            return
        if (
            not has_open_position
            and (
                not math.isfinite(completed_bar_age)
                or completed_bar_age > settings.max_entry_bar_age_seconds
            )
        ):
            self.last_bar_times[symbol] = completed_bar
            reason = (
                f"Completed M5 setup is {completed_bar_age:.1f}s old; "
                f"maximum executable age is "
                f"{settings.max_entry_bar_age_seconds:.0f}s. Waiting for the "
                "next candle instead of entering a late signal."
                if math.isfinite(completed_bar_age)
                else "Completed M5 setup age is unavailable; waiting for a fresh candle."
            )
            dashboard_state.update_symbol_decision(
                symbol,
                stage="MISSED CANDLE",
                action="HOLD",
                confidence=0.0,
                reasoning=reason,
                inference_time_s=0.0,
                gate_reason=reason,
                candle_time=completed_bar,
            )
            self._set_symbol_scan_state(
                symbol,
                "COMPLETE",
                last_scan=f"{datetime.now().strftime('%H:%M:%S')} / {symbol}",
            )
            return
        analysis_started_monotonic = time.monotonic()
        self.log(f"New completed M5 candle for {symbol}: {completed_bar}. Starting analysis...")
        self._set_symbol_scan_state(
            symbol,
            "ANALYZING",
            last_scan=datetime.now().strftime("%H:%M:%S"),
        )
        self._clear_manual_trade_candidate(symbol)
        dashboard_state.update_symbol_decision(
            symbol,
            stage="ANALYZING",
            action="—",
            confidence=0.0,
            reasoning="Analyzing the new completed M5 candle.",
            trade_management="",
            inference_time_s=0.0,
            candle_time=completed_bar,
            gate_reason="",
            entry=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            planned_rr=0.0,
            manual_override_available=False,
        )

        df_m15, df_h1, df_h4 = await asyncio.gather(
            self.reader.get_ohlcv(symbol, "M15", count=history_bars),
            self.reader.get_ohlcv(symbol, "H1", count=history_bars),
            self.reader.get_ohlcv(symbol, "H4", count=history_bars),
        )
        confirmations = {"M15": df_m15, "H1": df_h1, "H4": df_h4}
        for timeframe, frame in confirmations.items():
            if frame is None or frame.empty or bool(frame.attrs.get("is_stale", False)):
                reason = f"{timeframe} confirmation data is unavailable or stale"
                dashboard_state.update_symbol_decision(
                    symbol, stage="DATA ERROR", candle_time=completed_bar, gate_reason=reason
                )
                self._set_symbol_scan_state(symbol, "DATA_ERROR")
                self.log(f"{symbol} analysis skipped: {reason}.", "WARNING")
                return

        prepared = await self._confirmed_entry_analyses(symbol, df_m5, [df_m15, df_h1, df_h4])
        m5_analysis = prepared.get("M5")
        m15_analysis = prepared.get("M15")
        h1_analysis = prepared.get("H1")
        h4_analysis = prepared.get("H4")
        if not all((m5_analysis, m15_analysis, h1_analysis, h4_analysis)):
            dashboard_state.update_symbol_decision(
                symbol, stage="DATA ERROR", gate_reason="Indicator warm-up is incomplete"
            )
            self._set_symbol_scan_state(symbol, "DATA_ERROR")
            return

        previous_trend_states = dict(self._previous_trend_states.get(symbol, {}))
        current_trend_states = {
            timeframe: str(
                analysis.get("market_structure", {}).get("trend_state", "NEUTRAL")
            ).upper()
            for timeframe, analysis in {
                "M5": m5_analysis,
                "M15": m15_analysis,
                "H1": h1_analysis,
                "H4": h4_analysis,
            }.items()
        }
        self._previous_trend_states[symbol] = current_trend_states

        tick = await self.reader.get_live_tick(symbol)
        if tick:
            symbol_info = await asyncio.to_thread(mt5.symbol_info, symbol)
            point = symbol_info.point if symbol_info else 0.00001
            # Reuse the completed-candle analysis. This keeps Market Watch and
            # the LLM on one definition and avoids recalculating indicators.
            market_structure = m5_analysis.get("market_structure", {})
            temp_trend = market_structure.get(
                "trend_state", market_structure.get("trend", "NEUTRAL")
            )
            adx_val = float(
                m5_analysis.get("indicators", {}).get("adx_14", 0.0) or 0.0
            )
            dashboard_state.update_prices(
                symbol=symbol,
                bid=tick["bid"],
                ask=tick["ask"],
                point=point,
                trend=temp_trend,
                adx=round(adx_val, 1),
            )
        else:
            dashboard_state.update_symbol_decision(
                symbol, stage="ANALYZING", gate_reason="Live quote unavailable; entries will be blocked"
            )

        capital_fit = await asyncio.to_thread(
            DeterministicTradePlanner.assess_capital_fit,
            symbol,
            m5_analysis,
            account_info,
        )
        cached_rank = self._market_rankings.get(symbol.upper(), {})
        if cached_rank:
            # Ranking is refreshed as one atomic cross-market snapshot every
            # selection cycle.  Preserve that score/context until the next
            # refresh so a single-symbol candle update cannot show a score
            # that contradicts its published rank.
            for key, value in cached_rank.items():
                if (
                    key.startswith("selection_")
                    or key in {
                        "selected",
                        "model_eligible",
                        "model_selected",
                        "model_selection_rank",
                        "entry_prefilter_reason",
                        "opportunity_status",
                        "viable_entry_actions",
                        "opportunity_rejections",
                        "opportunity_bar",
                        "performance",
                        "performance_blocked",
                        "performance_probation",
                        "forex_context",
                    }
                ):
                    capital_fit[key] = value
        else:
            capital_fit.update(
                AdaptiveMarketSelector.score(symbol, m5_analysis, capital_fit)
            )
        dashboard_state.update_market_fit(symbol, capital_fit)
        if (
            settings.dynamic_market_selection_enabled
            and self._market_selection_initialized
            and not capital_fit.get("capital_fit")
            and not has_open_position
        ):
            self.last_bar_times[symbol] = completed_bar
            reason = str(capital_fit.get("reason", "Market is not capital-fit"))
            dashboard_state.update_symbol_decision(
                symbol,
                stage="CAPITAL FILTERED",
                action="HOLD",
                confidence=0.0,
                reasoning=reason,
                inference_time_s=0.0,
                gate_reason=reason,
                candle_time=completed_bar,
            )
            self._set_symbol_scan_state(symbol, "COMPLETE")
            return

        decision_analyses = {
            "M5": m5_analysis,
            "M15": m15_analysis,
            "H1": h1_analysis,
            "H4": h4_analysis,
        }
        allowed_evidence_ids = build_evidence_ids(decision_analyses)
        entry_contract = permitted_entry_actions(allowed_evidence_ids)
        decision_context = {
            "symbol": symbol,
            "completed_bar": completed_bar,
            "has_open_position": has_open_position,
            "open_positions": symbol_positions,
            "previous_trend_states": previous_trend_states,
            "market": capital_fit,
            "live_tick": tick or {},
            "forex_context": dict(
                self._forex_context_by_symbol.get(symbol.upper(), {})
            ),
            "analyses": {**decision_analyses},
        }
        fast_path_decision: Optional[Dict[str, Any]] = None
        history = await self.db.get_closed_positions(
            symbol=mt5.broker_symbol_name(symbol), limit=20, account_login=self._active_account_login,
        )
        if not has_open_position:
            prefilter_reason = self._entry_prefilter_reason(
                m5_analysis,
                m15_analysis,
                h1_analysis,
            )
            opportunity = screen_opportunities(decision_analyses, capital_fit, history)
            capital_fit.update(opportunity)
            capital_fit["entry_prefilter_reason"] = prefilter_reason
            capital_fit["model_eligible"] = bool(opportunity["model_eligible"] and not prefilter_reason)
            capital_fit["opportunity_bar"] = completed_bar
            capital_fit["opportunity_status"] = (
                "READY FOR REVIEW" if capital_fit["model_eligible"] else
                "WAITING FOR SETUP" if not entry_contract else "BLOCKED"
            )
            dashboard_state.update_market_fit(symbol, capital_fit)
            if prefilter_reason:
                self.last_bar_times[symbol] = completed_bar
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="PREFILTERED",
                    action="HOLD",
                    confidence=0.0,
                    reasoning=prefilter_reason,
                    inference_time_s=0.0,
                    gate_reason=prefilter_reason,
                    candle_time=completed_bar,
                )
                self._set_symbol_scan_state(
                    symbol,
                    "COMPLETE",
                    last_scan=(
                        f"{datetime.now().strftime('%H:%M:%S')} / {symbol}"
                    ),
                )
                return
            if not entry_contract:
                self.last_bar_times[symbol] = completed_bar
                reason = (
                    "No actionable completed-M5 BOS, confirmed CHoCH, "
                    "breakout, verified retest, or eligible range setup is "
                    "available. The local-model lane remains free for markets "
                    "that can produce a valid BUY or SELL decision."
                )
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="NO ENTRY EVIDENCE",
                    action="HOLD",
                    confidence=0.0,
                    reasoning=reason,
                    inference_time_s=0.0,
                    gate_reason=reason,
                    candle_time=completed_bar,
                )
                self._set_symbol_scan_state(
                    symbol,
                    "COMPLETE",
                    last_scan=(
                        f"{datetime.now().strftime('%H:%M:%S')} / {symbol}"
                    ),
                )
                return
            if not opportunity["viable_entry_actions"]:
                self.last_bar_times[symbol] = completed_bar
                reason = "; ".join(f"{side}: {detail}" for side, detail in opportunity["opportunity_rejections"].items())
                dashboard_state.update_symbol_decision(
                    symbol, stage="SETUP FILTERED", action="HOLD", confidence=0.0,
                    reasoning=reason, gate_reason=reason, inference_time_s=0.0, candle_time=completed_bar,
                )
                self._set_symbol_scan_state(symbol, "COMPLETE")
                return
            fast_path_decision = self._deterministic_entry_fast_path(
                decision_context,
                entry_contract,
            )
            if fast_path_decision is None:
                # A previous ranking snapshot cannot veto a fresh setup
                # when higher-ranked markets failed the current preflight.
                # The per-bar admission limit still bounds total model work.
                admitted, admission_reason = self._reserve_entry_model_slot(
                    symbol,
                    completed_bar,
                )
                if not admitted:
                    self.last_bar_times[symbol] = completed_bar
                    dashboard_state.update_symbol_decision(
                        symbol,
                        stage="RANKED STANDBY",
                        action="HOLD",
                        confidence=0.0,
                        reasoning=admission_reason,
                        inference_time_s=0.0,
                        gate_reason=admission_reason,
                        candle_time=completed_bar,
                    )
                    self._set_symbol_scan_state(
                        symbol,
                        "COMPLETE",
                        last_scan=(
                            f"{datetime.now().strftime('%H:%M:%S')} / {symbol}"
                        ),
                    )
                    return
        if (
            fast_path_decision is None
            and hasattr(self.llm, "readiness_probe")
            and not self._decision_provider_inference_ready
        ):
            dashboard_state.update_symbol_decision(
                symbol,
                stage="MODEL WARMING UP",
                action="HOLD",
                confidence=0.0,
                inference_time_s=0.0,
                gate_reason=(
                    "Waiting for a successful decision-provider inference "
                    "readiness probe"
                ),
                candle_time=completed_bar,
            )
            self._set_symbol_scan_state(symbol, "WAITING_FOR_MODEL")
            return
        if fast_path_decision is not None:
            system_prompt = "deterministic-entry-fast-path-v1"
            user_prompt = f"{symbol} completed M5 bar {completed_bar}"
        elif getattr(self.llm, "provider_name", "") == "deterministic":
            # The rules provider consumes the structured context directly. Do
            # not spend time rendering a model prompt or include account data.
            system_prompt = "deterministic-rules-v2-adaptive"
            user_prompt = f"{symbol} completed M5 bar {completed_bar}"
        else:
            system_prompt, user_prompt = PromptGenerator.generate(
                symbol=symbol,
                timeframe="M5",
                analysis_data=m5_analysis,
                m15_analysis=m15_analysis,
                h1_analysis=h1_analysis,
                h4_analysis=h4_analysis,
                account_info=account_info,
                open_positions=symbol_positions,
                trade_history=history,
                calendar_events=None,
                fast_exit_review=has_open_position,
                live_tick=tick,
                forex_context=decision_context["forex_context"],
            )

        if fast_path_decision is not None:
            self.log(
                f"Using deterministic entry fast path for {symbol}; "
                "all normal execution gates remain active."
            )
            self._set_symbol_scan_state(symbol, "FAST_PATH")
            dashboard_state.update_symbol_decision(
                symbol,
                stage="FAST PATH",
                candle_time=completed_bar,
            )
        else:
            self.log(
                f"Querying {settings.llm_provider.upper()} decision service for {symbol}..."
            )
            self._set_symbol_scan_state(symbol, "LLM_INFERENCE")
            dashboard_state.update_symbol_decision(
                symbol, stage="LLM INFERENCE", candle_time=completed_bar
            )
            logger.debug("Prepared bounded decision prompt for %s", symbol)
        start_time = asyncio.get_event_loop().time()
        telemetry: Dict[str, Any]

        async def request_model_decision() -> Tuple[Dict[str, Any], Dict[str, Any]]:
            if hasattr(self.llm, "request_decision"):
                model_response = await self.llm.request_decision(
                    system_prompt, user_prompt, context=decision_context
                )
                return (
                    model_response.decision,
                    model_response.telemetry.to_dict(),
                )
            # Compatibility for simple test doubles and older provider
            # adapters. Production providers use request_decision so each
            # concurrent call owns its telemetry.
            legacy_decision = await self.llm.get_trading_decision(
                system_prompt, user_prompt
            )
            return legacy_decision, {
                "trace_id": "",
                "provider": getattr(self.llm, "provider_name", "legacy"),
                "model": getattr(
                    self.llm, "model_name", settings.decision_model
                ),
                "latency_seconds": 0.0,
                "success": bool(legacy_decision),
                "error": getattr(self.llm, "last_error", ""),
                "prompt_tokens_estimate": len(
                    system_prompt + user_prompt
                ) // 4,
            }

        used_fast_path = fast_path_decision is not None
        if used_fast_path:
            decision = fast_path_decision
            telemetry = {
                "trace_id": uuid.uuid4().hex,
                "provider": "deterministic-fast-path",
                "model": "rules-v2-adaptive",
                "latency_seconds": 0.0,
                "success": True,
                "error": "",
                "prompt_tokens_estimate": 0,
            }
        elif has_open_position:
            # Managed-position reviews are not entry requests and remain
            # exempt from the completed-M5 entry deadline.
            async with self._decision_semaphore:
                decision, telemetry = await request_model_decision()
        else:
            inference_budget = self._entry_inference_budget_seconds(
                analysis_started_monotonic,
                completed_bar_age,
            )
            try:
                if inference_budget <= 0.0:
                    raise asyncio.TimeoutError
                async with asyncio.timeout(inference_budget):
                    async with self._decision_semaphore:
                        decision, telemetry = await request_model_decision()
            except asyncio.TimeoutError:
                self.last_bar_times[symbol] = completed_bar
                elapsed_budget = max(
                    0.0,
                    time.monotonic() - analysis_started_monotonic,
                )
                reason = (
                    "Entry inference deadline expired after "
                    f"{elapsed_budget:.1f}s, including the local-model queue. "
                    "The stale request was cancelled; waiting for the next "
                    "completed M5 candle."
                )
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="MISSED CANDLE",
                    action="HOLD",
                    confidence=0.0,
                    reasoning=reason,
                    inference_time_s=0.0,
                    gate_reason=reason,
                    candle_time=completed_bar,
                )
                self._set_symbol_scan_state(
                    symbol,
                    "COMPLETE",
                    last_scan=(
                        f"{datetime.now().strftime('%H:%M:%S')} / {symbol}"
                    ),
                )
                self.log(
                    f"Entry inference cancelled for {symbol}: {reason}",
                    "WARNING",
                )
                return
        elapsed = asyncio.get_event_loop().time() - start_time
        inference_latency = self._reported_inference_latency(telemetry, elapsed)
        if not used_fast_path:
            self._record_decision_provider_result(
                bool(telemetry.get("success"))
            )
        queue_delay = max(0.0, elapsed - inference_latency)
        if queue_delay >= 1.0:
            logger.debug(
                "%s waited %.2fs for the bounded inference slot; model execution took %.2fs",
                symbol,
                queue_delay,
                inference_latency,
            )
        close_position_side = None
        if symbol_positions:
            close_position_side = (
                "BUY" if int(symbol_positions[0].get("type", -1)) == 0 else "SELL"
            )
        is_valid, validated_decision, error_msg = DecisionValidator.validate_decision(
            decision,
            allowed_evidence_ids=allowed_evidence_ids,
            close_position_side=close_position_side,
            permitted_actions=entry_contract,
        )
        if telemetry.get("trace_id"):
            await asyncio.to_thread(
                self.replay_logger.log_decision_trace,
                symbol=symbol,
                candle_time=completed_bar,
                telemetry=telemetry,
                decision=decision,
                validation_status="VALID" if is_valid else f"INVALID: {error_msg}",
            )
        if not is_valid:
            # One completed candle receives one decision attempt. Retrying a
            # malformed response every poll only repeats the same market input,
            # overloads the provider, and delays other symbols.
            self.last_bar_times[symbol] = completed_bar
            dashboard_state.update_automation(
                llm_online=self._decision_provider_inference_ready,
                decision_provider_inference_ready=(
                    self._decision_provider_inference_ready
                ),
                model=str(telemetry.get("model") or settings.decision_model),
            )
            self._set_symbol_scan_state(symbol, "LLM_ERROR")
            dashboard_state.update_symbol_decision(
                symbol,
                stage="LLM ERROR",
                candle_time=completed_bar,
                inference_time_s=round(inference_latency, 2),
                gate_reason=error_msg,
            )
            self.log(f"Decision Validation Rejected for {symbol}: {error_msg}", "WARNING")
            await self._log_replay_attempt(
                symbol=symbol,
                action=decision.get("action", "HOLD") if decision else "HOLD",
                prompt_text=user_prompt,
                llm_json=decision if decision else {},
                indicators=m5_analysis.get("indicators", {}) if m5_analysis else {},
                market_structure=m5_analysis.get("market_structure", {}) if m5_analysis else {},
                quality_score=0.0,
                confluence_score=0.0,
                status=f"REJECTED: {error_msg}"
            )
            return
        decision = validated_decision
        decision["_forex_context"] = dict(
            decision_context.get("forex_context", {})
        )
        self.last_bar_times[symbol] = completed_bar
        automation_update = {
            # A per-symbol fast-path result should not relabel the configured
            # provider in the global readiness header.
            "model": (
                settings.decision_model
                if used_fast_path
                else str(telemetry.get("model") or settings.decision_model)
            ),
        }
        if not used_fast_path:
            automation_update.update(
                llm_online=True,
                decision_provider_inference_ready=True,
            )
        dashboard_state.update_automation(**automation_update)
        self._set_symbol_scan_state(
            symbol,
            "COMPLETE",
            last_scan=f"{datetime.now().strftime('%H:%M:%S')} / {symbol}",
        )
        action = decision.get("action", "HOLD").upper()
        confidence = decision.get("confidence", 0.0)
        reasoning = decision.get("reasoning", "")
        mgmt = decision.get("trade_management", "")
        if has_open_position and action in {"BUY", "SELL"}:
            reason = "Exit review is limited to HOLD or CLOSE while this symbol has a position"
            dashboard_state.update_symbol_decision(
                symbol,
                stage="REJECTED",
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                trade_management=mgmt,
                inference_time_s=round(inference_latency, 2),
                candle_time=completed_bar,
                gate_reason=reason,
            )
            self.log(f"Rejected {action} from exit-only review for {symbol}.", "WARNING")
            return

        plan = None
        manual_candidate = None
        if action in {"BUY", "SELL"}:
            plan = await asyncio.to_thread(
                DeterministicTradePlanner.build, symbol, action, m5_analysis
            )
            if plan.valid:
                decision.update(
                    entry=plan.entry,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                )
                manual_candidate = {
                    "symbol": symbol.upper(),
                    "action": action,
                    "decision": dict(decision),
                    "plan": plan.to_dict(),
                    "completed_bar": completed_bar,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "account_identity": self._account_identity(account_info),
                    "m5_analysis": m5_analysis,
                    "history": list(history),
                    "prompt_text": user_prompt,
                    "reasoning": reasoning,
                    "quality_score": 0.0,
                    "confluence_score": 0.0,
                }
        dashboard_state.update_llm(
            action=action,
            symbol=symbol,
            confidence=confidence,
            reasoning=reasoning,
            trade_management=mgmt,
            inference_time_s=inference_latency,
            prompt_tokens=int(
                telemetry.get("prompt_tokens_estimate")
                or len(system_prompt + user_prompt) // 4
            ),
            provider=str(telemetry.get("provider") or settings.llm_provider),
            model=str(telemetry.get("model") or settings.decision_model),
            trace_id=str(telemetry.get("trace_id") or ""),
        )
        dashboard_state.update_symbol_decision(
            symbol,
            stage="DECIDED" if action == "HOLD" else "SIGNAL",
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            trade_management=mgmt,
            inference_time_s=round(inference_latency, 2),
            candle_time=completed_bar,
            gate_reason="",
            entry=plan.entry if plan and plan.valid else 0.0,
            stop_loss=plan.stop_loss if plan and plan.valid else 0.0,
            take_profit=plan.take_profit if plan and plan.valid else 0.0,
            planned_rr=plan.planned_rr if plan and plan.valid else 0.0,
        )
        self.log(f"Decision for {symbol}: {action} ({confidence * 100:.0f}% confidence)")
        if action == "HOLD":
            return

        if action == "CLOSE" and has_open_position:
            if confidence < settings.confidence_threshold:
                reason = (
                    f"Close confidence {confidence:.0%} is below "
                    f"{settings.confidence_threshold:.0%}"
                )
                dashboard_state.update_symbol_decision(
                    symbol, stage="REJECTED", gate_reason=reason
                )
                self.log(reason, "WARNING")
                return
            ticket = int(decision.get("ticket_to_close") or 0)
            allowed_tickets = {int(position["ticket"]) for position in symbol_positions}
            if ticket not in allowed_tickets:
                self.log(f"Rejected LLM close for unmanaged/unknown ticket {ticket}.", "WARNING")
                return
            self.log(f"LLM requested close of position ticket {ticket} ({reasoning})")
            async with self._execution_lock:
                fresh_account = await self.conn.get_account_info()
                if not fresh_account or not self._same_account(
                    fresh_account, self._account_identity(account_info)
                ):
                    self.disarm_entries("Account changed during LLM inference")
                    dashboard_state.update_symbol_decision(
                        symbol, stage="REJECTED", gate_reason="Account changed during inference"
                    )
                    return
                fresh_positions = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
                if fresh_positions is None or not fresh_positions:
                    dashboard_state.update_symbol_decision(
                        symbol, stage="REJECTED", gate_reason="Position could not be re-verified"
                    )
                    return
                fresh_position = fresh_positions[0]
                if str(fresh_position.symbol).upper() != symbol.upper() or (
                    int(fresh_position.magic) != settings.strategy_magic
                    and not settings.manage_external_positions
                ):
                    dashboard_state.update_symbol_decision(
                        symbol, stage="REJECTED", gate_reason="Position ownership changed"
                    )
                    return
                res = await self.executor.close_position(
                    ticket, expected_account=self._account_identity(fresh_account)
                )
            if res.success:
                await self._remember_strategy_close_reason(
                    ticket, "MODEL_EXIT"
                )
                self.log(f"Successfully closed position ticket {ticket}.")
                await self.db.log_trade({
                    "ticket": ticket,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "action": "CLOSE",
                    "lot_size": 0.0,
                    "price": res.price or 0.0,
                    "sl": 0.0,
                    "tp": 0.0,
                    "profit": 0.0,
                    "reasoning": reasoning,
                    "status": "CLOSE_SUBMITTED"
                })
                dashboard_state.update_symbol_decision(symbol, stage="CLOSE SUBMITTED")
            else:
                self.log(f"Close failed for ticket {ticket}: {res.error}", "ERROR")
                dashboard_state.update_symbol_decision(
                    symbol, stage="EXECUTION ERROR", gate_reason=res.error or "Close failed"
                )
            return

        if action == "CLOSE":
            dashboard_state.update_symbol_decision(
                symbol, stage="REJECTED", gate_reason="No managed position is open"
            )
            return
        if action not in {"BUY", "SELL"}:
            return
        if symbol.upper() in {
            configured.upper() for configured in settings.shadow_symbols
        }:
            reason = (
                f"{symbol} is in shadow evaluation: the signal is recorded but "
                "no live order is submitted."
            )
            dashboard_state.update_symbol_decision(
                symbol, stage="SHADOW", gate_reason=reason
            )
            await self._log_replay_attempt(
                symbol=symbol,
                action=action,
                prompt_text=user_prompt,
                llm_json=decision,
                indicators=m5_analysis.get("indicators", {}),
                market_structure=m5_analysis.get("market_structure", {}),
                quality_score=0.0,
                confluence_score=0.0,
                status="SHADOW",
            )
            await self._record_shadow_candidate(
                symbol=symbol,
                action=action,
                completed_bar=completed_bar,
                plan=plan,
                stage="SHADOW",
                reason=reason,
            )
            self.log(reason, "WARNING")
            return
        if (
            settings.dynamic_market_selection_enabled
            and self._market_selection_initialized
            and symbol.upper() not in self._selected_symbols
        ):
            reason = "Adaptive market ranking changed before this signal could execute"
            dashboard_state.update_symbol_decision(
                symbol, stage="RANK EXPIRED", gate_reason=reason
            )
            return
        if confidence < settings.confidence_threshold:
            reversal_qualified, reversal_detail = (
                self.risk.qualify_failed_thesis_reversal(
                    action,
                    confidence,
                    history,
                    m5_analysis,
                    m15_analysis,
                )
            )
            if reversal_qualified:
                decision["_strategy"] = {
                    "mode": "FAILED_THESIS_REVERSAL",
                    "source": "DETERMINISTIC_FAILED_THESIS_REVERSAL",
                    "detail": reversal_detail,
                }
                if manual_candidate:
                    manual_candidate["decision"] = dict(decision)
                self.log(
                    f"Bounded failed-thesis reversal qualified for {symbol}: "
                    f"{reversal_detail}. Continuing through all normal risk gates."
                )
            else:
                reason = (
                    f"Confidence {confidence:.0%} is below the "
                    f"{settings.confidence_threshold:.0%} entry gate; bounded "
                    f"reversal exception not met ({reversal_detail})"
                )
                dashboard_state.update_symbol_decision(
                    symbol, stage="BELOW CONFIDENCE", gate_reason=reason
                )
                self.log(f"Entry rejected for {symbol}: {reason}", "WARNING")
                if manual_candidate:
                    self._offer_manual_trade_candidate(
                        symbol, manual_candidate, reason
                    )
                await self._record_shadow_candidate(
                    symbol=symbol,
                    action=action,
                    completed_bar=completed_bar,
                    plan=plan,
                    stage="BELOW CONFIDENCE",
                    reason=reason,
                )
                return
        if not plan or not plan.valid:
            reason = plan.reason if plan else "Deterministic order plan is unavailable"
            dashboard_state.update_symbol_decision(
                symbol, stage="PLAN REJECTED", gate_reason=reason
            )
            self.log(f"Plan Reject for {symbol}: {reason}", "WARNING")
            return
        if not self.entries_armed:
            self._defer_disarmed_signal(symbol)
            self.log(f"Entry signal for {symbol} ignored because entries are disarmed.", "WARNING")
            dashboard_state.update_symbol_decision(
                symbol, stage="SIGNAL · DISARMED", gate_reason="Entry authorization is disarmed"
            )
            if manual_candidate:
                self._offer_manual_trade_candidate(
                    symbol,
                    manual_candidate,
                    "Entry authorization is disarmed",
                )
            return

        decision_is_fresh, decision_age_reason = (
            self._entry_decision_fresh(analysis_started_monotonic)
        )
        if not decision_is_fresh:
            dashboard_state.update_symbol_decision(
                symbol,
                stage="DECISION EXPIRED",
                gate_reason=decision_age_reason,
            )
            self.log(
                f"Entry cancelled for {symbol}: {decision_age_reason}",
                "WARNING",
            )
            return

        async with self._execution_lock:
            protection_ok, protection_detail = (
                self._protection_progress_health()
            )
            if not protection_ok:
                self.disarm_entries(
                    f"Broker protection supervisor unavailable: "
                    f"{protection_detail}"
                )
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="REJECTED",
                    gate_reason=protection_detail,
                )
                return
            if (
                settings.dynamic_market_selection_enabled
                and self._market_selection_initialized
                and symbol.upper() not in self._selected_symbols
            ):
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="RANK EXPIRED",
                    gate_reason="Market left the active ranking before execution",
                )
                return
            fresh_account = await self.conn.get_account_info()
            if not fresh_account or not self._same_account(
                fresh_account, self._armed_account_identity
            ):
                self.disarm_entries("Active account no longer matches armed account")
                dashboard_state.update_symbol_decision(
                    symbol, stage="REJECTED", gate_reason="Armed account identity changed"
                )
                return
            if not self._history_ready:
                self.disarm_entries("Broker history is unhealthy; entries locked")
                dashboard_state.update_symbol_decision(
                    symbol, stage="REJECTED", gate_reason="Broker history is not reconciled"
                )
                return
            raw_fresh_positions = await asyncio.to_thread(mt5.positions_get)
            if raw_fresh_positions is None:
                self.disarm_entries("MT5 position refresh failed before execution")
                dashboard_state.update_symbol_decision(
                    symbol, stage="REJECTED", gate_reason="Position refresh failed"
                )
                return
            fresh_positions = [
                {
                    "ticket": int(p.ticket),
                    "symbol": str(p.symbol),
                    "type": int(p.type),
                    "volume": float(p.volume),
                    "price_open": float(p.price_open),
                    "sl": float(p.sl),
                    "tp": float(p.tp),
                    "magic": int(p.magic),
                    "bot_owned": int(p.magic) == settings.strategy_magic,
                }
                for p in raw_fresh_positions
            ]
            if any(p["symbol"].upper() == symbol.upper() for p in fresh_positions):
                dashboard_state.update_symbol_decision(
                    symbol, stage="REJECTED", gate_reason="A position already exists on this symbol"
                )
                return

            # Fetch one broker snapshot and use it for both planning and risk.
            # Previously the planner and risk manager fetched independent
            # quotes, which added latency and could display/approve levels from
            # slightly different ticks during a fast move.
            symbol_info = await asyncio.to_thread(mt5.symbol_info, symbol)
            final_tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)
            if symbol_info is None or final_tick is None:
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="REJECTED",
                    gate_reason="Final broker quote is unavailable",
                )
                return
            final_plan = await asyncio.to_thread(
                DeterministicTradePlanner.build,
                symbol,
                action,
                m5_analysis,
                info=symbol_info,
                tick=final_tick,
            )
            if not final_plan.valid:
                dashboard_state.update_symbol_decision(
                    symbol, stage="PLAN REJECTED", gate_reason=final_plan.reason
                )
                return
            decision.update(
                entry=final_plan.entry,
                stop_loss=final_plan.stop_loss,
                take_profit=final_plan.take_profit,
            )
            dashboard_state.update_symbol_decision(
                symbol,
                stage="RISK CHECK",
                entry=final_plan.entry,
                stop_loss=final_plan.stop_loss,
                take_profit=final_plan.take_profit,
                planned_rr=final_plan.planned_rr,
            )
            self.log(f"Validating {action} setup for {symbol} against Risk Manager...")

            class RiskSnapshot:
                def __init__(self, bid, ask, point, contract_size, spread):
                    class Metrics:
                        def __init__(self, bid, ask, point, contract_size, spread):
                            self.bid = bid
                            self.ask = ask
                            self.point = point
                            self.contract_size = contract_size
                            self.spread = spread
                    self.metrics = Metrics(bid, ask, point, contract_size, spread)
            risk_snap = RiskSnapshot(
                final_tick.bid,
                final_tick.ask,
                symbol_info.point,
                symbol_info.trade_contract_size,
                symbol_info.spread,
            )
            validation = await asyncio.to_thread(
                self.risk.validate,
                symbol=symbol,
                action=action,
                llm_decision=decision,
                account_info=fresh_account,
                open_positions=fresh_positions,
                market_snapshot=risk_snap,
                calendar_events=None,
                trade_history=history,
                m5_analysis=m5_analysis,
                m15_analysis=m15_analysis,
                h1_analysis=h1_analysis,
                h4_analysis=h4_analysis,
            )

            quality_score = validation.quality_score
            confluence_score = validation.confluence_score
            if not validation.approved:
                self.log(f"Risk Reject: {validation.reason}", "WARNING")
                dashboard_state.update_symbol_decision(
                    symbol, stage="RISK REJECTED", gate_reason=validation.reason
                )
                await self._log_replay_attempt(
                    symbol=symbol,
                    action=action,
                    prompt_text=user_prompt,
                    llm_json=decision,
                    indicators=m5_analysis.get("indicators", {}) if m5_analysis else {},
                    market_structure=m5_analysis.get("market_structure", {}) if m5_analysis else {},
                    quality_score=quality_score,
                    confluence_score=confluence_score,
                    status=f"REJECTED: {validation.reason}"
                )
                await self._record_shadow_candidate(
                    symbol=symbol,
                    action=action,
                    completed_bar=completed_bar,
                    plan=final_plan,
                    stage="RISK REJECTED",
                    reason=validation.reason,
                )
                if manual_candidate:
                    manual_candidate["quality_score"] = quality_score
                    manual_candidate["confluence_score"] = confluence_score
                    self._offer_manual_trade_candidate(
                        symbol, manual_candidate, validation.reason
                    )
                return
            lot = validation.adjusted_lot
            if lot is None or not math.isfinite(float(lot)) or float(lot) <= 0:
                reason = "Risk approval did not return a valid broker-sized volume"
                self.log(f"Risk Reject: {reason}", "ERROR")
                dashboard_state.update_symbol_decision(
                    symbol, stage="RISK REJECTED", gate_reason=reason
                )
                return
            lot = float(lot)
            entry_price = final_plan.entry
            sl = final_plan.stop_loss
            tp = final_plan.take_profit
            self.log(f"Executing {action} order for {symbol} size={lot}...")
            dashboard_state.update_symbol_decision(symbol, stage="ORDER CHECK")
            max_risk_usd = float(validation.risk_budget_usd)
            if not math.isfinite(max_risk_usd) or max_risk_usd <= 0:
                reason = "Risk approval did not return a valid execution budget"
                self.log(f"Risk Reject: {reason}", "ERROR")
                dashboard_state.update_symbol_decision(
                    symbol, stage="RISK REJECTED", gate_reason=reason
                )
                return
            bar_is_current, bar_reason = await self._entry_bar_is_current(
                symbol, completed_bar
            )
            if not bar_is_current:
                self.log(f"Entry cancelled for {symbol}: {bar_reason}", "WARNING")
                dashboard_state.update_symbol_decision(
                    symbol, stage="DECISION EXPIRED", gate_reason=bar_reason
                )
                return
            decision_is_fresh, decision_age_reason = (
                self._entry_decision_fresh(
                    analysis_started_monotonic
                )
            )
            if not decision_is_fresh:
                self.log(
                    f"Entry cancelled for {symbol}: "
                    f"{decision_age_reason}",
                    "WARNING",
                )
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="DECISION EXPIRED",
                    gate_reason=decision_age_reason,
                )
                return
            res = await self.executor.open_trade(
                symbol=symbol,
                action=action,
                lot_size=lot,
                sl_price=sl,
                tp_price=tp,
                expected_account=self._armed_account_identity,
                max_risk_usd=max_risk_usd,
            )
            if res.success:
                self._clear_manual_trade_candidate(symbol)
                actual_lot = float(res.volume or lot)
                fill_label = "partially filled" if res.partial else "executed"
                self.log(
                    f"Trade {fill_label}. Ticket={res.ticket} Price={res.price} "
                    f"Volume={actual_lot:g}/{lot:g} | "
                    f"risk=${validation.estimated_risk_usd:.2f} "
                    f"({validation.risk_percent_balance:.2f}%) RR={validation.planned_rr:.2f}",
                    "WARNING" if res.partial else "INFO",
                )
                # Log entry into database
                await self.db.log_trade({
                    "ticket": res.ticket,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "action": action,
                    "lot_size": actual_lot,
                    "price": res.price or entry_price,
                    "sl": sl or 0.0,
                    "tp": tp or 0.0,
                    "profit": 0.0,
                    "reasoning": reasoning,
                    "status": "OPEN_PARTIAL" if res.partial else "OPEN"
                })
                baseline_ok = True
                if not settings.dry_run and res.ticket:
                    _, baseline_ok = await self._persist_initial_risk(
                        account=fresh_account,
                        ticket=int(res.ticket),
                        symbol=symbol,
                        direction=action,
                        entry_price=float(res.price or entry_price),
                        initial_sl=float(sl or 0.0),
                        volume=actual_lot,
                        info=symbol_info,
                    )
                    if not baseline_ok:
                        self._position_risk_healthy = False
                        self.disarm_entries(
                            "Trade opened, but immutable risk state could not be persisted; entries locked"
                        )
                # Log active trade signal in replay log
                await self._log_replay_attempt(
                    symbol=symbol,
                    action=action,
                    prompt_text=user_prompt,
                    llm_json=decision,
                    indicators=m5_analysis.get("indicators", {}) if m5_analysis else {},
                    market_structure=m5_analysis.get("market_structure", {}) if m5_analysis else {},
                    quality_score=quality_score,
                    confluence_score=confluence_score,
                    status="OPEN_PARTIAL" if res.partial else "OPEN",
                    ticket=res.ticket
                )
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage=(
                        "OPENED · RISK STATE ERROR" if not baseline_ok
                        else "OPENED PARTIAL" if res.partial
                        else "OPENED"
                    ),
                    gate_reason=f"Ticket {res.ticket}",
                )
            else:
                if res.state_changed:
                    self.disarm_entries(
                        "Broker reported an incomplete/uncertain state change; entries locked for review"
                    )
                self.log(
                    f"Order {'changed broker state but was not fully verified' if res.state_changed else 'failed'} "
                    f"for {symbol}: {res.error}",
                    "ERROR",
                )
                dashboard_state.update_symbol_decision(
                    symbol,
                    stage="EXECUTION STATE UNCERTAIN" if res.state_changed else "EXECUTION ERROR",
                    gate_reason=res.error or "Order submission failed",
                )

    async def _refresh_protection_snapshot(
        self,
        snapshot: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Refresh one ticket after acquiring the execution mutation lock."""
        ticket = int(snapshot["ticket"])
        expected_account = snapshot.get("account_identity")
        if isinstance(expected_account, dict) and expected_account:
            active_account = getattr(self, "_active_account_identity", None)
            if active_account is not None and not self._same_account(
                expected_account, active_account
            ):
                raise RuntimeError(
                    f"Position {ticket} belongs to a superseded MT5 account "
                    "snapshot"
                )
            native_account = await asyncio.to_thread(mt5.account_info)
            if native_account is None:
                raise RuntimeError(
                    f"MT5 account could not be verified before protecting "
                    f"position {ticket}: {mt5.last_error()}"
                )
            actual_account = {
                "login": int(getattr(native_account, "login", 0) or 0),
                "server": str(getattr(native_account, "server", "") or ""),
                "company": str(getattr(native_account, "company", "") or ""),
                "trade_mode": int(
                    getattr(native_account, "trade_mode", -1)
                ),
            }
            if not self._same_account(actual_account, expected_account):
                raise RuntimeError(
                    f"MT5 account changed before protecting position {ticket}"
                )
        rows = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
        if rows is None:
            raise RuntimeError(
                f"Position {ticket} could not be refreshed before protection: "
                f"{mt5.last_error()}"
            )
        if not rows:
            self._forget_position_runtime_state(ticket)
            return None

        position = rows[0]
        info = await asyncio.to_thread(mt5.symbol_info, position.symbol)
        one_pip = pip_size(info) if info is not None else 0.0
        if one_pip > 0 and int(position.type) == mt5.POSITION_TYPE_BUY:
            profit_pips = (
                float(position.price_current) - float(position.price_open)
            ) / one_pip
        elif one_pip > 0:
            profit_pips = (
                float(position.price_open) - float(position.price_current)
            ) / one_pip
        else:
            profit_pips = 0.0

        initial_risk_pips = float(
            snapshot.get("initial_risk_pips", 0.0) or 0.0
        )
        account = dict(self._active_account_identity or {})
        if info is not None and one_pip > 0 and account:
            refreshed_risk, baseline_ok = await self._initial_risk_for_position(
                account,
                position,
                info,
            )
            if baseline_ok and refreshed_risk > 0:
                initial_risk_pips = refreshed_risk

        peak_pips, peak_usd = await self._restore_and_update_position_peak(
            account,
            ticket=ticket,
            profit_pips=profit_pips,
            observed_profit_usd=float(position.profit),
        )
        try:
            configured_cost = (
                configured_execution_cost_usd(
                    position.symbol,
                    info,
                    float(position.volume),
                )
                if info is not None
                else 0.0
            )
        except (TypeError, ValueError, OverflowError):
            configured_cost = 0.0

        refreshed = dict(snapshot)
        refreshed.update(
            symbol=str(position.symbol),
            type=int(position.type),
            volume=float(position.volume),
            price_open=float(position.price_open),
            price_current=float(position.price_current),
            sl=float(position.sl or 0.0),
            tp=float(position.tp or 0.0),
            profit=float(position.profit),
            estimated_net_profit_usd=(
                float(position.profit)
                + float(getattr(position, "swap", 0.0) or 0.0)
                - configured_cost
            ),
            profit_pips=profit_pips,
            peak_profit_pips=peak_pips,
            peak_profit_usd=peak_usd,
            initial_risk_pips=initial_risk_pips,
            initial_risk_usd=getattr(self, "_initial_risk_usd", {}).get(
                ticket, 0.0
            ),
            bot_owned=(
                int(getattr(position, "magic", 0))
                == settings.strategy_magic
            ),
        )
        return refreshed

    async def _update_profit_retention(self, position: Dict[str, Any]) -> RetentionState:
        """Persist the desired net floor separately from an accepted broker SL."""
        if not (settings.profit_lock_enabled and settings.profit_retention_enabled):
            return RetentionState()
        for name in ("_profit_retention_states", "_profit_retention_persisted"):
            if not hasattr(self, name):
                setattr(self, name, {})
        if not hasattr(self, "_profit_retention_loaded"):
            self._profit_retention_loaded = set()
        ticket = int(position["ticket"])
        symbol = str(position["symbol"])
        previous = self._profit_retention_states.get(ticket, RetentionState())
        key = self._position_peak_cache_key(self._active_account_identity, ticket) + ":retention"
        inputs = dict(
            net_profit_usd=position.get("estimated_net_profit_usd", position["profit"]),
            volume=position.get("volume", 0.0),
            initial_risk_usd=position.get("initial_risk_usd", 0.0),
            policy=settings,
        )
        state = advance_retention(previous, **inputs)
        if previous.volume > 0 and state.volume != previous.volume:
            # The same broker SL now represents a different cash amount.
            getattr(self, "_profit_lock_levels", {}).pop(ticket, None)
            getattr(self, "_profit_lock_tickets", set()).discard(ticket)
        if ticket not in self._profit_retention_loaded:
            try:
                stored = RetentionState.from_dict(await self.db.get_cache(key))
                restored = advance_retention(stored, **inputs)
                # Merge observations made while storage was temporarily unavailable.
                if restored.volume == state.volume:
                    state = RetentionState(
                        max(state.peak_net_usd, restored.peak_net_usd),
                        max(state.floor_usd, restored.floor_usd),
                        state.volume,
                        max(state.reference_volume, restored.reference_volume),
                    )
                self._profit_retention_loaded.add(ticket)
                self._profit_retention_persisted[ticket] = stored
            except Exception as exc:
                self._log_protection_failure(ticket, symbol, "Profit retention restore", exc)
        self._profit_retention_states[ticket] = state
        persisted = self._profit_retention_persisted.get(ticket, RetentionState())
        changed = (
            state.floor_usd != persisted.floor_usd
            or state.volume != persisted.volume
            or state.reference_volume != persisted.reference_volume
            or abs(state.peak_net_usd - persisted.peak_net_usd) >= 0.01 - 1e-9
        )
        if ticket in self._profit_retention_loaded and changed:
            try:
                if not await self.db.set_cache(key, state.to_dict()):
                    raise RuntimeError("cache write was not confirmed")
                self._profit_retention_persisted[ticket] = state
            except Exception as exc:
                self._log_protection_failure(ticket, symbol, "Profit retention persist", exc)
        return state

    async def _apply_protections(
        self,
        open_positions: List[Dict[str, Any]],
        *,
        refresh_from_broker: bool = False,
    ) -> None:
        """Applies target profit, target loss, trailing stops and break-even rules to open tickets."""
        if not hasattr(self, "_profit_lock_tickets"):
            self._profit_lock_tickets = set()
        if not hasattr(self, "_profit_lock_levels"):
            self._profit_lock_levels = {}
        for candidate in open_positions:
            p = candidate
            if refresh_from_broker:
                refreshed = await self._refresh_protection_snapshot(candidate)
                if refreshed is None:
                    continue
                p = refreshed
            ticket = p["ticket"]
            profit_usd = p["profit"]
            estimated_net_profit_usd = float(
                p.get("estimated_net_profit_usd", profit_usd) or 0.0
            )
            profit_pips = p["profit_pips"]
            symbol = p["symbol"]
            risk_pips = float(p.get("initial_risk_pips", 0.0) or 0.0)
            initial_risk_usd = float(
                p.get("initial_risk_usd", 0.0) or 0.0
            )
            peak_profit_usd = max(
                0.0, float(p.get("peak_profit_usd", 0.0) or 0.0)
            )

            # Update peak pips profit in memory
            peak = max(self._peak_profits.get(ticket, profit_pips), profit_pips)
            self._peak_profits[ticket] = peak
            peak_r = peak / risk_pips if risk_pips > 0 else 0.0
            live_r = profit_pips / risk_pips if risk_pips > 0 else 0.0

            # ── 1. Target Profit Auto-Close ────────────────────────────
            if settings.auto_close_profit_enabled and profit_usd >= settings.auto_close_profit_usd:
                self.log(
                    f"Target Profit Auto-Close triggered for ticket {ticket} ({symbol}): "
                    f"Floating profit ${profit_usd:.2f} >= Target ${settings.auto_close_profit_usd:.2f}. "
                    f"Executing exit..."
                )
                res = await self.executor.close_position(
                    ticket, expected_account=self._active_account_identity
                )
                if res.success:
                    await self._remember_strategy_close_reason(
                        ticket, "TARGET_PROFIT"
                    )
                    self._forget_position_runtime_state(ticket)
                    await self._update_replay_outcome(
                        ticket,
                        profit_usd,
                        close_reason="TARGET_PROFIT",
                    )
                    await self.db.log_trade({
                        "ticket": ticket,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol,
                        "action": "CLOSE",
                        "lot_size": p.get("volume", 0.0),
                        "price": res.price or p.get("price_current", 0.0),
                        "sl": p.get("sl", 0.0),
                        "tp": p.get("tp", 0.0),
                        "profit": profit_usd,
                        "reasoning": f"Target Profit of ${settings.auto_close_profit_usd:.2f} reached",
                        "status": "CLOSED"
                    })
                    continue
                self._log_protection_failure(
                    ticket, symbol, "Target-profit close", res.error
                )

            # ── 2. Target Loss Auto-Close ──────────────────────────────
            # Profit will be a negative value (e.g. -1.50) representing a loss.
            if settings.auto_close_loss_enabled and profit_usd <= -settings.auto_close_loss_usd:
                self.log(
                    f"Target Loss Auto-Close triggered for ticket {ticket} ({symbol}): "
                    f"Floating loss ${profit_usd:.2f} <= Target -${settings.auto_close_loss_usd:.2f}. "
                    f"Executing exit..."
                )
                res = await self.executor.close_position(
                    ticket, expected_account=self._active_account_identity
                )
                if res.success:
                    await self._remember_strategy_close_reason(
                        ticket, "TARGET_LOSS"
                    )
                    self._forget_position_runtime_state(ticket)
                    await self._update_replay_outcome(
                        ticket,
                        profit_usd,
                        close_reason="TARGET_LOSS",
                    )
                    await self.db.log_trade({
                        "ticket": ticket,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol,
                        "action": "CLOSE",
                        "lot_size": p.get("volume", 0.0),
                        "price": res.price or p.get("price_current", 0.0),
                        "sl": p.get("sl", 0.0),
                        "tp": p.get("tp", 0.0),
                        "profit": profit_usd,
                        "reasoning": f"Target Loss of -${settings.auto_close_loss_usd:.2f} reached",
                        "status": "CLOSED"
                    })
                    continue
                self._log_protection_failure(
                    ticket, symbol, "Target-loss close", res.error
                )

            # Retire a setup that has consumed a full review window without
            # demonstrating even modest favorable excursion. This is expressed
            # in bars and R, not a fixed dollar amount, so it scales with the
            # original broker-verified stop risk.
            stagnation_minutes = float(
                getattr(settings, "position_stagnation_bars", 12)
            ) * 5.0
            if (
                bool(getattr(settings, "position_stagnation_exit_enabled", True))
                and risk_pips > 0
                and float(p.get("duration_min", 0.0) or 0.0)
                >= stagnation_minutes
                and peak_r
                < float(
                    getattr(settings, "position_stagnation_max_peak_r", 0.15)
                )
                and estimated_net_profit_usd <= 0.0
            ):
                self.log(
                    f"Stagnation exit triggered for ticket {ticket} ({symbol}): "
                    f"{float(p.get('duration_min', 0.0) or 0.0):.0f} minutes "
                    f"open with only {peak_r:.2f} R maximum progress."
                )
                stagnant_close = await self.executor.close_position(
                    ticket, expected_account=self._active_account_identity
                )
                if stagnant_close.success:
                    await self._remember_strategy_close_reason(
                        ticket, "STAGNATION_EXIT"
                    )
                    self._forget_position_runtime_state(ticket)
                    await self._update_replay_outcome(
                        ticket,
                        estimated_net_profit_usd,
                        close_reason="STAGNATION_EXIT",
                    )
                    await self.db.log_trade({
                        "ticket": ticket,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol,
                        "action": "CLOSE",
                        "lot_size": p.get("volume", 0.0),
                        "price": stagnant_close.price or p.get("price_current", 0.0),
                        "sl": p.get("sl", 0.0),
                        "tp": p.get("tp", 0.0),
                        "profit": estimated_net_profit_usd,
                        "reasoning": (
                            f"Stagnation exit after {stagnation_minutes:.0f} minutes; "
                            f"peak progress {peak_r:.2f} R"
                        ),
                        "status": "CLOSED",
                    })
                    continue
                self._log_protection_failure(
                    ticket, symbol, "Stagnation close", stagnant_close.error
                )

            # ── 3. Persisted-peak profit giveback guard ──────────────
            retention = await self._update_profit_retention(p)
            retention_breached = (
                retention.floor_usd > 0
                and math.isfinite(estimated_net_profit_usd)
                and estimated_net_profit_usd <= retention.floor_usd + 1e-9
            )
            giveback_level_r = peak_r * (
                1.0 - settings.profit_giveback_fraction
            )
            giveback_level_usd = peak_profit_usd * (
                1.0 - settings.profit_giveback_fraction
            )
            r_giveback_armed = (
                risk_pips > 0
                and peak_r >= settings.profit_giveback_trigger_r
            )
            usd_giveback_armed = (
                settings.micro_profit_protection_enabled
                and float(
                    getattr(settings, "profit_giveback_trigger_usd", 0.20)
                )
                > 0
                and peak_profit_usd
                >= float(
                    getattr(settings, "profit_giveback_trigger_usd", 0.20)
                )
            )
            # A sub-1R winner already has a broker-side profit floor installed
            # by the lock path below.  Do not also market-close it on an
            # ordinary giveback: that converted recoverable pullbacks (such as
            # the observed USDJPY continuation) into premature exits.  Once a
            # valid-risk trade has reached the mature threshold, the existing
            # R/USD giveback conditions may close it.  The USD fallback remains
            # available only when an R baseline could not be reconstructed.
            giveback_close_mature = (
                risk_pips <= 0
                or peak_r
                >= float(
                    getattr(settings, "profit_giveback_close_min_r", 1.0)
                )
            )
            if (
                settings.profit_giveback_enabled
                and (
                    retention_breached
                    or (giveback_close_mature and (
                    (
                        r_giveback_armed
                        and live_r <= giveback_level_r + 1e-9
                    )
                    or (
                        usd_giveback_armed
                        and profit_usd <= giveback_level_usd + 1e-9
                    )
                    ))
                )
            ):
                close_reason = "PROFIT_RETENTION" if retention_breached else "PROFIT_GIVEBACK"
                exit_detail = (
                    f"Net-profit retention: peak ${retention.peak_net_usd:.2f}, "
                    f"floor ${retention.floor_usd:.2f}, net ${estimated_net_profit_usd:.2f}"
                    if retention_breached else
                    f"R-based peak giveback {peak_r:.2f} R to {live_r:.2f} R"
                )
                self.log(
                    f"{close_reason} guard triggered for ticket {ticket} "
                    f"({symbol}): trade retraced from +{peak_r:.2f} R to "
                    f"{live_r:+.2f} R (broker ${profit_usd:.2f}, estimated "
                    f"net ${estimated_net_profit_usd:.2f}; peak "
                    f"${peak_profit_usd:.2f}). {exit_detail}. Executing exit..."
                )
                res = await self.executor.close_position(
                    ticket, expected_account=self._active_account_identity
                )
                if res.success:
                    await self._remember_strategy_close_reason(
                        ticket, close_reason
                    )
                    self._forget_position_runtime_state(ticket)
                    await self._update_replay_outcome(
                        ticket,
                        estimated_net_profit_usd,
                        close_reason=close_reason,
                    )
                    await self.db.log_trade({
                        "ticket": ticket,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol,
                        "action": "CLOSE",
                        "lot_size": p.get("volume", 0.0),
                        "price": res.price or p.get("price_current", 0.0),
                        "sl": p.get("sl", 0.0),
                        "tp": p.get("tp", 0.0),
                        "profit": estimated_net_profit_usd,
                        "reasoning": exit_detail,
                        "status": "CLOSED",
                    })
                    continue
                self._log_protection_failure(
                    ticket, symbol, "Profit-giveback close", res.error
                )

            # ── 4. Live-profit tiered broker protection ─────────────
            # The first two tiers require both live net USD and live R. The
            # early cash tier is capped by initial R. Mature net-peak retention
            # adds an uncapped cash milestone, then a percentage ratchet.
            # Per-ticket applied levels
            # let later tiers improve the stop without repeating the same MT5
            # modification every protection poll.
            lock_floor_usd, lock_tier = self._profit_lock_target(
                live_r=live_r,
                estimated_net_profit_usd=estimated_net_profit_usd,
                has_r_baseline=risk_pips > 0,
                initial_risk_usd=initial_risk_usd * (
                    retention.volume / retention.reference_volume
                    if retention.reference_volume > 0 else 1.0
                ),
            )
            if retention.floor_usd > lock_floor_usd:
                lock_floor_usd, lock_tier = retention.floor_usd, "peak retention"

            applied_floor_usd = float(
                self._profit_lock_levels.get(ticket, 0.0) or 0.0
            )
            if (
                settings.profit_lock_enabled
                and lock_floor_usd > applied_floor_usd + 1e-9
            ):
                result = await self.executor.lock_minimum_net_profit(
                    ticket,
                    floor_usd=lock_floor_usd,
                    expected_account=self._active_account_identity,
                )
                if result.success or "worsen" in str(result.error or "").lower():
                    self._profit_lock_tickets.add(ticket)
                    self._profit_lock_levels[ticket] = lock_floor_usd
                    if result.success:
                        self.log(
                            f"{lock_tier.capitalize()} profit floor armed for "
                            f"ticket {ticket} ({symbol}) at {live_r:+.2f} R / "
                            f"${estimated_net_profit_usd:.2f} estimated net; "
                            f"protected net floor ${lock_floor_usd:.2f}."
                        )
                else:
                    self._log_protection_failure(
                        ticket, symbol, "Profit floor", result.error
                    )

            # ── 5. Break-even at a configured R multiple ──────────────
            if not hasattr(self, "_breakeven_tickets"):
                self._breakeven_tickets = set()
            if (
                risk_pips > 0
                and profit_pips >= risk_pips * settings.breakeven_trigger_r
                and ticket not in self._breakeven_tickets
            ):
                breakeven_result = await self.executor.move_to_breakeven(
                    ticket,
                    buffer_pips=settings.breakeven_buffer_pips,
                    expected_account=self._active_account_identity,
                )
                breakeven_error = str(breakeven_result.error or "").lower()
                if breakeven_result.success or "worsen" in breakeven_error:
                    # A stronger profit/trailing stop already satisfies the
                    # break-even objective. Do not query and warn every 0.5s.
                    self._breakeven_tickets.add(ticket)
                else:
                    self._log_protection_failure(
                        ticket,
                        symbol,
                        "Break-even",
                        breakeven_result.error,
                    )

            # ── 6. Volatility-normalized trailing stop ────────────────
            if risk_pips > 0 and profit_pips >= risk_pips * settings.trailing_trigger_r:
                await self.executor.apply_trailing_stop(
                    ticket,
                    trail_pips=max(risk_pips * settings.trailing_distance_r, 0.1),
                    expected_account=self._active_account_identity,
                )

    @staticmethod
    def _profit_lock_target(
        *,
        live_r: float,
        estimated_net_profit_usd: float,
        has_r_baseline: bool,
        initial_risk_usd: float = 0.0,
    ) -> Tuple[float, str]:
        """Return the strongest currently eligible net-profit floor and tier."""
        try:
            net_profit = float(estimated_net_profit_usd)
            current_r = float(live_r)
            risk_usd = float(initial_risk_usd)
        except (TypeError, ValueError, OverflowError):
            return 0.0, ""
        if not math.isfinite(net_profit):
            return 0.0, ""
        if not has_r_baseline:
            if net_profit >= settings.profit_lock_final_trigger_usd:
                return float(settings.profit_lock_final_floor_usd), "final"
            return 0.0, ""
        if not math.isfinite(current_r):
            return 0.0, ""

        eligible: List[Tuple[float, str]] = []
        if (
            not settings.profit_lock_final_fallback_only
            and net_profit >= settings.profit_lock_final_trigger_usd
            and current_r >= settings.profit_lock_final_trigger_r
            and math.isfinite(risk_usd)
            and risk_usd > 0
        ):
            hybrid_floor = min(
                settings.profit_lock_final_floor_usd,
                risk_usd * settings.profit_lock_final_max_floor_r,
            )
            if hybrid_floor > 0 and hybrid_floor < net_profit:
                eligible.append((round(hybrid_floor + 1e-12, 2), "hybrid $1"))
        elif (
            not settings.profit_lock_final_fallback_only
            and net_profit >= settings.profit_lock_final_trigger_usd
            and (not math.isfinite(risk_usd) or risk_usd <= 0)
        ):
            # The pip baseline can occasionally survive a conversion metadata
            # outage without its dollar companion. Retain the original bounded
            # fallback instead of silently dropping mature protection.
            return float(settings.profit_lock_final_floor_usd), "final"
        if (
            current_r >= settings.profit_lock_mature_trigger_r
            and math.isfinite(risk_usd)
            and risk_usd > 0
        ):
            floor = max(
                settings.profit_lock_floor_usd,
                risk_usd * settings.profit_lock_mature_floor_r,
            )
            # A floor at or above current cost-adjusted profit has no room for
            # spread/slippage and cannot be placed safely. The mature trigger
            # normally supplies ample headroom; this cap protects unusual
            # commission or conversion cases without disabling the tier.
            floor = min(floor, net_profit * 0.80)
            if floor > 0:
                eligible.append((
                    round(floor + 1e-12, 2),
                    f"mature {settings.profit_lock_mature_floor_r:.2f}R",
                ))
        if (
            current_r >= settings.profit_lock_mid_trigger_r
            and net_profit >= settings.profit_lock_mid_trigger_usd
        ):
            floor = max(
                settings.profit_lock_floor_usd,
                net_profit * settings.profit_lock_mid_fraction,
            )
            eligible.append((round(floor + 1e-12, 2), "35%"))
        if (
            current_r >= settings.profit_lock_trigger_r
            and net_profit >= settings.profit_lock_trigger_usd
        ):
            eligible.append((float(settings.profit_lock_floor_usd), "first"))
        return max(eligible, key=lambda item: item[0]) if eligible else (0.0, "")

"""
ui/dashboard.py
FastAPI + WebSocket server for the monitoring dashboard.

Serves the static HTML UI and pushes live JSON over WebSocket every second.
Never blocks — all heavy work is delegated to the trading engine threads.
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Dict, Set, Tuple
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app_config.settings import settings
from ui.state import dashboard_state
from utils.system_monitor import get_system_metrics
from utils.instance_lock import InstanceLock

logger = logging.getLogger("TradingSystem.Dashboard")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
LOCAL_LLM_QUANTIZATION_PROFILES = ("AUTO", "Q4_K_M", "Q6_K", "Q8_0")


def _config_env_path() -> Path:
    return Path(__file__).parent.parent / ".env"


def _is_local_origin(value: str) -> bool:
    """Accept browser origins only when they resolve to this local terminal."""
    if not value:
        return True  # Non-browser local clients (for diagnostics/tests).
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and parsed.hostname in {
            "127.0.0.1", "localhost", "::1"
        }
    except ValueError:
        return False

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Own background tasks and trading dependencies in one lifecycle."""
    instance_lock = InstanceLock(Path(__file__).parent.parent)
    instance_lock.acquire()
    broadcast_task = asyncio.create_task(_broadcast_loop())
    logger.info("Dashboard WebSocket broadcaster started.")
    engine = getattr(app.state, "engine", None)
    database = getattr(app.state, "database", None)
    connection = getattr(app.state, "connection_manager", None)
    try:
        if database is not None:
            try:
                history = await database.get_trade_history(limit=50)
                dashboard_state.load_trade_history(history)
            except Exception as exc:
                logger.error("Failed to load trade history: %s", exc)

        if engine is not None and settings.auto_start_monitoring:
            if not await engine.start():
                logger.error("Automatic market monitoring failed to start.")
        elif connection is not None and await connection.initialize():
            account = await connection.get_account_info()
            if account:
                dashboard_state.update_account(account)
                if engine is not None:
                    await engine._reconcile_history(account)
        yield
    finally:
        if engine is not None:
            await engine.stop()
        if connection is not None:
            await connection.shutdown()
        broadcast_task.cancel()
        with suppress(asyncio.CancelledError):
            await broadcast_task
        instance_lock.release()


app = FastAPI(title="LLM Trading Dashboard", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def loopback_only(request: Request, call_next):
    """Trading controls are intentionally unavailable to LAN clients."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse({"error": "Local access only"}, status_code=403)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not _is_local_origin(
        request.headers.get("origin", "")
    ):
        return JSONResponse({"error": "Local browser origin required"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' ws: wss:; img-src 'self' data:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return response

# Serve static files (JS, CSS, fonts if any)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Connected WebSocket clients ────────────────────────────────────────
_ws_clients: Set[WebSocket] = set()


def _live_payload() -> Dict[str, Any]:
    """Return one compact quote sample per market for the fast UI stream."""
    with dashboard_state._lock:
        prices = {
            key: {
                "bid": value.bid,
                "ask": value.ask,
                "spread_pips": value.spread_pips,
                "spread_value": value.spread_value,
                "spread_unit": value.spread_unit,
                "asset_class": value.asset_class,
                "trend": value.trend,
                "adx": value.adx,
                "updated_at": value.updated_at,
            }
            for key, value in dashboard_state.prices.items()
        }
        return {
            "_type": "live",
            "system": {
                "cpu_pct": dashboard_state.system.cpu_pct,
                "ram_pct": dashboard_state.system.ram_pct,
                "ram_used_gb": dashboard_state.system.ram_used_gb,
                "ram_total_gb": dashboard_state.system.ram_total_gb,
                "gpu_util_pct": dashboard_state.system.gpu_util_pct,
                "gpu_mem_pct": dashboard_state.system.gpu_mem_pct,
                "gpu_mem_used_mb": dashboard_state.system.gpu_mem_used_mb,
                "gpu_mem_total_mb": dashboard_state.system.gpu_mem_total_mb,
                "gpu_temp_c": dashboard_state.system.gpu_temp_c,
            },
            "prices": prices,
            # Re-sending the entire 50-tick backlog three times a second made
            # the browser repeatedly parse and deduplicate old data.  Current
            # quote samples are sufficient because the client owns its bounded
            # sparkline history.
            "tick_stream": [
                {
                    "symbol": key,
                    "bid": value["bid"],
                    "ask": value["ask"],
                    "time": value["updated_at"],
                }
                for key, value in prices.items()
            ],
        }


async def _send_bounded(
    websocket: WebSocket, payload: str, timeout: float = 0.75
) -> Tuple[WebSocket, bool]:
    """Prevent one stalled browser from blocking the trading event loop."""
    try:
        await asyncio.wait_for(websocket.send_text(payload), timeout=timeout)
        return websocket, True
    except Exception:
        return websocket, False


async def _broadcast_loop():
    """Unified broadcast loop: pushes fast live metrics every 333ms, full state every 1s."""
    global _ws_clients
    counter = 0
    last_metrics_error_log = 0.0
    last_payload_error_log = 0.0
    while True:
        await asyncio.sleep(0.333)
        if not _ws_clients:
            continue
            
        counter += 1
        try:
            sys_metrics = get_system_metrics()   # instant — no thread needed
            dashboard_state.update_system(sys_metrics)
        except Exception as exc:
            now_mono = asyncio.get_running_loop().time()
            if now_mono - last_metrics_error_log >= 30.0:
                logger.warning("System metrics update failed: %s", exc)
                last_metrics_error_log = now_mono

        # Build payload. Serialize after releasing the state lock so dashboard
        # rendering cannot delay broker/position state writers.
        try:
            snapshot = (
                dashboard_state.to_dict(
                    compact=True,
                    include_tick_stream=False,
                )
                if counter % 3 == 0
                else _live_payload()
            )
            payload = json.dumps(snapshot, separators=(",", ":"))
        except Exception as exc:
            now_mono = asyncio.get_running_loop().time()
            if now_mono - last_payload_error_log >= 30.0:
                logger.error("Dashboard payload serialization failed: %s", exc)
                last_payload_error_log = now_mono
            continue

        clients = list(_ws_clients)
        results = await asyncio.gather(
            *(_send_bounded(ws, payload) for ws in clients),
            return_exceptions=False,
        )
        dead = {ws for ws, sent in results if not sent}
        if dead:
            _ws_clients -= dead
            await asyncio.gather(
                *(
                    ws.close(code=1011, reason="Dashboard stream stalled")
                    for ws in dead
                ),
                return_exceptions=True,
            )


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page dashboard HTML."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found. Place index.html in ui/static/</h1>")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host if websocket.client else ""
    if (
        client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}
        or not _is_local_origin(websocket.headers.get("origin", ""))
    ):
        await websocket.close(code=1008, reason="Local access only")
        return
    await websocket.accept()
    try:
        # Send the initial snapshot before the broadcaster owns all outbound
        # writes. The endpoint then only receives, avoiding concurrent writes
        # to one Starlette WebSocket.
        await asyncio.wait_for(
            websocket.send_text(
                json.dumps(
                    dashboard_state.to_dict(
                        compact=True,
                        include_tick_stream=False,
                    )
                )
            ),
            timeout=2.0,
        )
        _ws_clients.add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(_ws_clients)}")
        while True:
            await websocket.receive()
    except (WebSocketDisconnect, Exception) as e:
        logger.debug(f"WebSocket connection closed: {e}")
    finally:
        _ws_clients.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(_ws_clients)}")


# ── REST API ────────────────────────────────────────────────────────────

@app.get("/api/state")
async def get_state():
    return JSONResponse(
        dashboard_state.to_dict(compact=True, include_tick_stream=False)
    )


@app.get("/api/live")
async def get_live():
    """Lightweight fast-poll endpoint: returns only system metrics and prices."""
    sys_metrics = get_system_metrics()
    dashboard_state.update_system(sys_metrics)
    with dashboard_state._lock:
        return JSONResponse({
            "system": {
                "cpu_pct":         dashboard_state.system.cpu_pct,
                "ram_pct":         dashboard_state.system.ram_pct,
                "ram_used_gb":     dashboard_state.system.ram_used_gb,
                "ram_total_gb":    dashboard_state.system.ram_total_gb,
                "gpu_util_pct":    dashboard_state.system.gpu_util_pct,
                "gpu_mem_pct":     dashboard_state.system.gpu_mem_pct,
                "gpu_mem_used_mb": dashboard_state.system.gpu_mem_used_mb,
                "gpu_mem_total_mb":dashboard_state.system.gpu_mem_total_mb,
                "gpu_temp_c":      dashboard_state.system.gpu_temp_c,
            },
            "prices": {k: {"bid": v.bid, "ask": v.ask,
                           "spread_pips": v.spread_pips,
                           "spread_value": v.spread_value,
                           "spread_unit": v.spread_unit,
                           "asset_class": v.asset_class,
                           "trend": v.trend, "adx": v.adx}
                       for k, v in dashboard_state.prices.items()},
        })


@app.get("/api/config")
async def get_config():
    return {
        "config_fingerprint": settings.config_fingerprint,
        "symbols": settings.trading_symbols,
        "dynamic_market_selection_enabled": settings.dynamic_market_selection_enabled,
        "market_candidate_symbols": settings.market_candidate_symbols,
        "dynamic_market_max_symbols": settings.dynamic_market_max_symbols,
        "market_selection_refresh_seconds": settings.market_selection_refresh_seconds,
        "shadow_symbols": settings.shadow_symbols,
        "entry_min_adx": settings.entry_min_adx,
        "entry_require_adx_rising": settings.entry_require_adx_rising,
        "entry_min_opposing_distance_atr": settings.entry_min_opposing_distance_atr,
        "entry_unconfirmed_bos_min_opposing_distance_atr": (
            settings.entry_unconfirmed_bos_min_opposing_distance_atr
        ),
        "same_thesis_reentry_min_bars": settings.same_thesis_reentry_min_bars,
        "retest_continuation_enabled": settings.retest_continuation_enabled,
        "retest_min_resumption_atr": settings.retest_min_resumption_atr,
        "timeframe": settings.trading_timeframe,
        "scan_timeframe": "M5",
        "confirmation_timeframes": ["M15", "H1", "H4"],
        "analysis_interval_seconds": settings.analysis_interval_seconds,
        "decision_poll_seconds": settings.decision_poll_seconds,
        "analysis_history_bars": settings.analysis_history_bars,
        "max_tick_age_seconds": settings.max_tick_age_seconds,
        "auto_start_monitoring": settings.auto_start_monitoring,
        "autonomous_health_interval_seconds": settings.autonomous_health_interval_seconds,
        "engine_heartbeat_stale_seconds": settings.engine_heartbeat_stale_seconds,
        "max_entry_decision_age_seconds": settings.max_entry_decision_age_seconds,
        "max_entry_bar_age_seconds": settings.max_entry_bar_age_seconds,
        "entry_max_execution_drift_atr": (
            settings.entry_max_execution_drift_atr
        ),
        "entry_strong_alignment_max_execution_drift_atr": (
            settings.entry_strong_alignment_max_execution_drift_atr
        ),
        "entry_strong_alignment_min_confidence": (
            settings.entry_strong_alignment_min_confidence
        ),
        "entry_strong_alignment_chase_min_confidence": (
            settings.entry_strong_alignment_chase_min_confidence
        ),
        "entry_strong_alignment_chase_max_extension_atr": (
            settings.entry_strong_alignment_chase_max_extension_atr
        ),
        "entry_max_executable_premium_atr": (
            settings.entry_max_executable_premium_atr
        ),
        "breakout_min_quality_score": settings.breakout_min_quality_score,
        "breakout_macro_min_adx": settings.breakout_macro_min_adx,
        "breakout_strong_lower_adx": settings.breakout_strong_lower_adx,
        "breakout_exhaustion_continuation_enabled": (
            settings.breakout_exhaustion_continuation_enabled
        ),
        "breakout_exhaustion_max_extension_atr": (
            settings.breakout_exhaustion_max_extension_atr
        ),
        "overextension_min_band_overshoot_atr": (
            settings.overextension_min_band_overshoot_atr
        ),
        "decision_provider_failure_threshold": settings.decision_provider_failure_threshold,
        "risk_percent": settings.risk_percent,
        "manual_override_max_risk_pct": settings.manual_override_max_risk_pct,
        "max_open_positions": settings.max_open_positions,
        "default_lot_size": settings.default_lot_size,
        "confidence_threshold": settings.confidence_threshold,
        "failed_thesis_reversal_enabled": settings.failed_thesis_reversal_enabled,
        "failed_thesis_reversal_min_confidence": (
            settings.failed_thesis_reversal_min_confidence
        ),
        "failed_thesis_reversal_max_age_bars": (
            settings.failed_thesis_reversal_max_age_bars
        ),
        "failed_thesis_reversal_min_m5_adx": (
            settings.failed_thesis_reversal_min_m5_adx
        ),
        "failed_thesis_reversal_min_m15_adx": (
            settings.failed_thesis_reversal_min_m15_adx
        ),
        "llm_provider": settings.llm_provider,
        "decision_model": settings.decision_model,
        "local_llm_model": settings.local_llm_model,
        "local_llm_required_quantization": settings.local_llm_required_quantization,
        "local_llm_quantization_options": list(
            LOCAL_LLM_QUANTIZATION_PROFILES
        ),
        "local_llm_url": settings.local_llm_url,
        "local_llm_temperature": settings.local_llm_temperature,
        "local_llm_top_p": settings.local_llm_top_p,
        "local_llm_seed": settings.local_llm_seed,
        "local_llm_context_size": settings.local_llm_context_size,
        "local_llm_max_tokens": settings.local_llm_max_tokens,
        "local_llm_timeout": settings.local_llm_timeout,
        "local_llm_max_retries": settings.local_llm_max_retries,
        "local_llm_structured_output": settings.local_llm_structured_output,
        "openai_model": settings.openai_model,
        "openai_reasoning_effort": settings.openai_reasoning_effort,
        "openai_configured": bool(settings.openai_api_key),
        "llm_max_concurrency": settings.llm_max_concurrency,
        "llm_entry_candidates_per_bar": settings.llm_entry_candidates_per_bar,
        "deterministic_entry_fast_path_enabled": (
            settings.deterministic_entry_fast_path_enabled
        ),
        "max_spread_pips": settings.max_spread_pips,
        "max_crypto_spread_bps": settings.max_crypto_spread_bps,
        "max_spread_to_stop_pct": settings.max_spread_to_stop_pct,
        "max_order_deviation_points": settings.max_order_deviation_points,
        "momentum_deterioration_exit_enabled": (
            settings.momentum_deterioration_exit_enabled
        ),
        "momentum_deterioration_confirm_bars": (
            settings.momentum_deterioration_confirm_bars
        ),
        "momentum_deterioration_min_loss_r": (
            settings.momentum_deterioration_min_loss_r
        ),
        "momentum_deterioration_min_body_atr": (
            settings.momentum_deterioration_min_body_atr
        ),
        "fx_round_turn_cost_usd_per_lot": settings.fx_round_turn_cost_usd_per_lot,
        "crypto_round_turn_cost_usd_per_lot": settings.crypto_round_turn_cost_usd_per_lot,
        "cfd_round_turn_cost_usd_per_lot": settings.cfd_round_turn_cost_usd_per_lot,
        "fixed_execution_cost_usd": settings.fixed_execution_cost_usd,
        "news_lockout_minutes": settings.news_lockout_minutes,
        "require_news_calendar": settings.require_news_calendar,
        "max_daily_loss_usd": settings.max_daily_loss_usd,
        "max_drawdown_pct": settings.max_drawdown_pct,
        "drawdown_entry_lock_enabled": settings.drawdown_entry_lock_enabled,
        "min_risk_reward_ratio": settings.min_risk_reward_ratio,
        "screenshots_enabled": settings.screenshots_enabled,
        "dry_run": settings.dry_run,
        "session_filter_enabled": settings.session_filter_enabled,
        "allowed_sessions_utc": settings.allowed_sessions_utc,
        "weekend_trading_enabled": settings.weekend_trading_enabled,
        "weekend_symbols": settings.weekend_symbols,
        "crypto_only_on_weekend": settings.crypto_only_on_weekend,
        "max_portfolio_risk_pct": settings.max_portfolio_risk_pct,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "max_margin_usage_pct": settings.max_margin_usage_pct,
        "min_margin_level_pct": settings.min_margin_level_pct,
        "loss_cooldown_minutes": settings.loss_cooldown_minutes,
        "loss_streak_pause_hours": settings.loss_streak_pause_hours,
        "auto_close_profit_enabled": settings.auto_close_profit_enabled,
        "auto_close_profit_usd": settings.auto_close_profit_usd,
        "auto_close_loss_enabled": settings.auto_close_loss_enabled,
        "auto_close_loss_usd": settings.auto_close_loss_usd,
        "fast_exit_review_enabled": settings.fast_exit_review_enabled,
        "exit_model_confirmation_enabled": (
            settings.exit_model_confirmation_enabled
        ),
        "position_exit_review_timeframe": settings.position_exit_review_timeframe,
        "micro_profit_protection_enabled": (
            settings.micro_profit_protection_enabled
        ),
        "profit_lock_enabled": settings.profit_lock_enabled,
        "profit_lock_trigger_r": settings.profit_lock_trigger_r,
        "profit_lock_trigger_usd": settings.profit_lock_trigger_usd,
        "profit_lock_floor_usd": settings.profit_lock_floor_usd,
        "profit_lock_mid_trigger_r": settings.profit_lock_mid_trigger_r,
        "profit_lock_mid_trigger_usd": settings.profit_lock_mid_trigger_usd,
        "profit_lock_mid_fraction": settings.profit_lock_mid_fraction,
        "profit_lock_final_trigger_usd": (
            settings.profit_lock_final_trigger_usd
        ),
        "profit_lock_final_floor_usd": settings.profit_lock_final_floor_usd,
        "profit_giveback_enabled": settings.profit_giveback_enabled,
        "profit_giveback_trigger_r": settings.profit_giveback_trigger_r,
        "profit_giveback_close_min_r": (
            settings.profit_giveback_close_min_r
        ),
        "profit_giveback_trigger_usd": settings.profit_giveback_trigger_usd,
        "profit_giveback_fraction": settings.profit_giveback_fraction,
        "breakeven_trigger_r": settings.breakeven_trigger_r,
        "breakeven_buffer_pips": settings.breakeven_buffer_pips,
        "trailing_trigger_r": settings.trailing_trigger_r,
        "trailing_distance_r": settings.trailing_distance_r,
        "position_stagnation_exit_enabled": settings.position_stagnation_exit_enabled,
        "position_stagnation_bars": settings.position_stagnation_bars,
        "position_stagnation_max_peak_r": settings.position_stagnation_max_peak_r,
        "shadow_outcomes_enabled": settings.shadow_outcomes_enabled,
        "shadow_outcome_horizon_minutes": settings.shadow_outcome_horizon_minutes,
        "strategy_magic": settings.strategy_magic,
        "account_source": "ACTIVE_MT5_ACCOUNT",
        "plan_stop_atr": settings.plan_stop_atr,
        "plan_target_rr": settings.plan_target_rr,
        "plan_max_cost_target_extension_r": (
            settings.plan_max_cost_target_extension_r
        ),
        "plan_target_buffer_atr": settings.plan_target_buffer_atr,
        "require_technical_target": settings.require_technical_target,
        "entry_max_candle_range_atr": settings.entry_max_candle_range_atr,
        "breakout_min_displacement_atr": (
            settings.breakout_min_displacement_atr
        ),
        "entry_adx_decline_tolerance": settings.entry_adx_decline_tolerance,
        "entry_aligned_adx_decline_tolerance": (
            settings.entry_aligned_adx_decline_tolerance
        ),
        "entry_min_h4_adx": settings.entry_min_h4_adx,
        "overextension_rsi_high": settings.overextension_rsi_high,
        "overextension_rsi_low": settings.overextension_rsi_low,
        "adaptive_reversal_enabled": settings.adaptive_reversal_enabled,
        "adaptive_reversal_min_adx": settings.adaptive_reversal_min_adx,
        "market_shock_range_atr": settings.market_shock_range_atr,
        "market_shock_gap_atr": settings.market_shock_gap_atr,
    }


@app.get("/api/logs")
async def get_logs():
    return {"logs": dashboard_state.logs[-200:]}


@app.get("/api/positions")
async def get_positions():
    from dataclasses import asdict
    return {"positions": [asdict(p) for p in dashboard_state.positions]}


@app.get("/api/history")
async def get_history(limit: int = 50):
    from dataclasses import asdict
    return {"history": [asdict(t) for t in dashboard_state.closed_trades[:limit]]}


@app.post("/api/config")
async def save_config(body: dict):
    """
    Persist configuration changes back to the .env file.
    Changes take effect on the next engine restart.
    """
    env_path = _config_env_path()
    try:
        allowed = {
            "TRADING_SYMBOLS", "MARKET_CANDIDATE_SYMBOLS", "RISK_PERCENT",
            "DYNAMIC_MARKET_SELECTION_ENABLED", "DYNAMIC_MARKET_MAX_SYMBOLS",
            "LLM_ENTRY_CANDIDATES_PER_BAR",
            "SHADOW_SYMBOLS",
            "MARKET_SELECTION_REFRESH_SECONDS", "ADAPTIVE_REVERSAL_ENABLED",
            "ADAPTIVE_REVERSAL_MIN_ADX", "MARKET_SHOCK_RANGE_ATR",
            "MARKET_SHOCK_GAP_ATR",
            "AUTO_START_MONITORING", "ANALYSIS_INTERVAL_SECONDS",
            "DECISION_POLL_SECONDS",
            "MAX_OPEN_POSITIONS", "DEFAULT_LOT_SIZE", "CONFIDENCE_THRESHOLD",
            "FAILED_THESIS_REVERSAL_ENABLED",
            "FAILED_THESIS_REVERSAL_MIN_CONFIDENCE",
            "FAILED_THESIS_REVERSAL_MAX_AGE_BARS",
            "FAILED_THESIS_REVERSAL_MIN_M5_ADX",
            "FAILED_THESIS_REVERSAL_MIN_M15_ADX",
            "MAX_SPREAD_PIPS", "MAX_CRYPTO_SPREAD_BPS", "MAX_SPREAD_TO_STOP_PCT",
            "MAX_ORDER_DEVIATION_POINTS", "FX_ROUND_TURN_COST_USD_PER_LOT",
            "CRYPTO_ROUND_TURN_COST_USD_PER_LOT", "CFD_ROUND_TURN_COST_USD_PER_LOT",
            "FIXED_EXECUTION_COST_USD",
            "MIN_RISK_REWARD_RATIO", "MAX_DAILY_LOSS_USD",
            "MAX_DAILY_LOSS_PCT", "MAX_PORTFOLIO_RISK_PCT", "MAX_MARGIN_USAGE_PCT",
            "MIN_MARGIN_LEVEL_PCT", "LOSS_COOLDOWN_MINUTES", "NEWS_LOCKOUT_MINUTES",
            "LLM_PROVIDER", "LLM_MAX_CONCURRENCY",
            "LOCAL_LLM_MODEL", "LOCAL_LLM_REQUIRED_QUANTIZATION",
            "LOCAL_LLM_URL", "LOCAL_LLM_CONTEXT_SIZE",
            "LOCAL_LLM_TEMPERATURE", "LOCAL_LLM_TOP_P", "LOCAL_LLM_TIMEOUT",
            "OPENAI_MODEL", "OPENAI_REASONING_EFFORT",
            "LOCAL_LLM_STRUCTURED_OUTPUT", "SCREENSHOTS_ENABLED", "DRY_RUN",
            "SESSION_FILTER_ENABLED", "ALLOWED_SESSIONS_UTC", "WEEKEND_TRADING_ENABLED",
            "CRYPTO_ONLY_ON_WEEKEND", "WEEKEND_SYMBOLS", "AUTO_CLOSE_TARGET_PROFIT_ENABLED",
            "AUTO_CLOSE_TARGET_PROFIT_USD", "AUTO_CLOSE_TARGET_LOSS_ENABLED",
            "AUTO_CLOSE_TARGET_LOSS_USD", "REQUIRE_NEWS_CALENDAR",
            "FAST_EXIT_REVIEW_ENABLED", "POSITION_EXIT_REVIEW_TIMEFRAME",
            "PROFIT_LOCK_ENABLED", "PROFIT_LOCK_TRIGGER_R",
            "PROFIT_LOCK_TRIGGER_USD", "PROFIT_LOCK_FLOOR_USD",
            "PROFIT_LOCK_MID_TRIGGER_R", "PROFIT_LOCK_MID_TRIGGER_USD",
            "PROFIT_LOCK_MID_FRACTION", "PROFIT_LOCK_FINAL_TRIGGER_USD",
            "PROFIT_LOCK_FINAL_FLOOR_USD",
            "PROFIT_GIVEBACK_ENABLED", "PROFIT_GIVEBACK_TRIGGER_R",
            "PROFIT_GIVEBACK_CLOSE_MIN_R",
            "PROFIT_GIVEBACK_TRIGGER_USD", "PROFIT_GIVEBACK_FRACTION",
            "BREAKEVEN_TRIGGER_R", "BREAKEVEN_BUFFER_PIPS",
            "TRAILING_TRIGGER_R", "TRAILING_DISTANCE_R",
            "PLAN_STOP_ATR", "PLAN_TARGET_RR",
            "PLAN_MAX_COST_TARGET_EXTENSION_R", "PLAN_TARGET_BUFFER_ATR",
            "REQUIRE_TECHNICAL_TARGET",
            "ENTRY_MIN_ADX", "ENTRY_ADX_DECLINE_TOLERANCE",
            "ENTRY_ALIGNED_ADX_DECLINE_TOLERANCE",
            "ENTRY_MIN_OPPOSING_DISTANCE_ATR",
            "ENTRY_UNCONFIRMED_BOS_MIN_OPPOSING_DISTANCE_ATR",
            "ENTRY_MAX_CANDLE_RANGE_ATR",
            "ENTRY_STRONG_ALIGNMENT_CHASE_MIN_CONFIDENCE",
            "ENTRY_STRONG_ALIGNMENT_CHASE_MAX_EXTENSION_ATR",
            "BREAKOUT_MIN_DISPLACEMENT_ATR",
            "ENTRY_MIN_H4_ADX",
            "OVEREXTENSION_RSI_HIGH", "OVEREXTENSION_RSI_LOW",
            "ENTRY_REQUIRE_ADX_RISING", "SAME_THESIS_REENTRY_MIN_BARS",
            "RETEST_CONTINUATION_ENABLED", "RETEST_MIN_RESUMPTION_ATR",
            "ANALYSIS_HISTORY_BARS", "MAX_TICK_AGE_SECONDS",
            "LOSS_STREAK_PAUSE_HOURS",
            "MICRO_PROFIT_PROTECTION_ENABLED",
        }
        updates = {str(k).upper(): str(v).strip() for k, v in body.items() if str(k).upper() in allowed}
        if not updates:
            return JSONResponse({"error": "No permitted configuration keys supplied"}, status_code=400)
        if any("\n" in value or "\r" in value for value in updates.values()):
            return JSONResponse({"error": "Configuration values cannot contain newlines"}, status_code=422)

        numeric_ranges = {
            "RISK_PERCENT": (0.05, 20.0), "MAX_OPEN_POSITIONS": (1, 10),
            "DEFAULT_LOT_SIZE": (0.01, 100.0), "CONFIDENCE_THRESHOLD": (0.5, 0.99),
            "FAILED_THESIS_REVERSAL_MIN_CONFIDENCE": (0.60, 0.99),
            "FAILED_THESIS_REVERSAL_MAX_AGE_BARS": (1, 36),
            "FAILED_THESIS_REVERSAL_MIN_M5_ADX": (0.0, 60.0),
            "FAILED_THESIS_REVERSAL_MIN_M15_ADX": (0.0, 60.0),
            "MAX_SPREAD_PIPS": (0.1, 1000.0), "MIN_RISK_REWARD_RATIO": (1.0, 10.0),
            "MAX_CRYPTO_SPREAD_BPS": (0.1, 1000.0),
            "MAX_SPREAD_TO_STOP_PCT": (1.0, 100.0),
            "MAX_ORDER_DEVIATION_POINTS": (0, 1000),
            "FX_ROUND_TURN_COST_USD_PER_LOT": (0, 1000),
            "CRYPTO_ROUND_TURN_COST_USD_PER_LOT": (0, 1000),
            "CFD_ROUND_TURN_COST_USD_PER_LOT": (0, 1000),
            "FIXED_EXECUTION_COST_USD": (0, 1000),
            "PLAN_STOP_ATR": (0.5, 10.0), "PLAN_TARGET_RR": (1.0, 10.0),
            "PLAN_MAX_COST_TARGET_EXTENSION_R": (0.0, 2.0),
            "PLAN_TARGET_BUFFER_ATR": (0.0, 1.0),
            "ENTRY_MIN_ADX": (15.0, 60.0),
            "ENTRY_ADX_DECLINE_TOLERANCE": (0.0, 5.0),
            "ENTRY_ALIGNED_ADX_DECLINE_TOLERANCE": (0.0, 5.0),
            "ENTRY_MIN_OPPOSING_DISTANCE_ATR": (0.0, 5.0),
            "ENTRY_UNCONFIRMED_BOS_MIN_OPPOSING_DISTANCE_ATR": (0.0, 5.0),
            "ENTRY_MAX_CANDLE_RANGE_ATR": (0.5, 5.0),
            "ENTRY_STRONG_ALIGNMENT_CHASE_MIN_CONFIDENCE": (0.5, 1.0),
            "ENTRY_STRONG_ALIGNMENT_CHASE_MAX_EXTENSION_ATR": (0.5, 5.0),
            "BREAKOUT_MIN_DISPLACEMENT_ATR": (0.0, 1.0),
            "ENTRY_MIN_H4_ADX": (0.0, 60.0),
            "OVEREXTENSION_RSI_HIGH": (50.0, 100.0),
            "OVEREXTENSION_RSI_LOW": (0.0, 50.0),
            "SAME_THESIS_REENTRY_MIN_BARS": (1, 24),
            "RETEST_MIN_RESUMPTION_ATR": (0.01, 1.5),
            "MAX_DAILY_LOSS_USD": (0.01, 100000.0), "MAX_DAILY_LOSS_PCT": (0.0, 20.0),
            "MAX_PORTFOLIO_RISK_PCT": (0.1, 20.0), "MAX_MARGIN_USAGE_PCT": (5.0, 90.0),
            "MIN_MARGIN_LEVEL_PCT": (100.0, 5000.0), "LOSS_COOLDOWN_MINUTES": (0, 10080),
            "ANALYSIS_INTERVAL_SECONDS": (15, 300),
            "DECISION_POLL_SECONDS": (0.5, 5.0),
            "ANALYSIS_HISTORY_BARS": (210, 2000),
            "MAX_TICK_AGE_SECONDS": (1, 120),
            "LOSS_STREAK_PAUSE_HOURS": (0, 720),
            "AUTO_CLOSE_TARGET_PROFIT_USD": (0.01, 100000.0),
            "AUTO_CLOSE_TARGET_LOSS_USD": (0.01, 100000.0),
            "PROFIT_LOCK_TRIGGER_R": (0.1, 10.0),
            "PROFIT_LOCK_TRIGGER_USD": (0.01, 100000.0),
            "PROFIT_LOCK_FLOOR_USD": (0.0, 100000.0),
            "PROFIT_LOCK_MID_TRIGGER_R": (0.1, 10.0),
            "PROFIT_LOCK_MID_TRIGGER_USD": (0.01, 100000.0),
            "PROFIT_LOCK_MID_FRACTION": (0.05, 0.95),
            "PROFIT_LOCK_FINAL_TRIGGER_USD": (0.01, 100000.0),
            "PROFIT_LOCK_FINAL_FLOOR_USD": (0.0, 100000.0),
            "PROFIT_GIVEBACK_TRIGGER_R": (0.1, 10.0),
            "PROFIT_GIVEBACK_CLOSE_MIN_R": (0.1, 10.0),
            "PROFIT_GIVEBACK_TRIGGER_USD": (0.0, 100000.0),
            "PROFIT_GIVEBACK_FRACTION": (0.05, 0.95),
            "BREAKEVEN_TRIGGER_R": (0.1, 10.0),
            "BREAKEVEN_BUFFER_PIPS": (0.0, 1000.0),
            "TRAILING_TRIGGER_R": (0.1, 10.0),
            "TRAILING_DISTANCE_R": (0.05, 10.0),
            "LOCAL_LLM_CONTEXT_SIZE": (1024, 262144),
            "LLM_MAX_CONCURRENCY": (1, 6),
            "NEWS_LOCKOUT_MINUTES": (0, 1440),
            "DYNAMIC_MARKET_MAX_SYMBOLS": (1, 12),
            "MARKET_SELECTION_REFRESH_SECONDS": (30, 3600),
            "ADAPTIVE_REVERSAL_MIN_ADX": (18, 60),
            "MARKET_SHOCK_RANGE_ATR": (1.5, 10),
            "MARKET_SHOCK_GAP_ATR": (0.5, 10),
        }
        integer_keys = {
            "MAX_OPEN_POSITIONS", "ANALYSIS_INTERVAL_SECONDS", "ANALYSIS_HISTORY_BARS",
            "LOSS_COOLDOWN_MINUTES", "LOSS_STREAK_PAUSE_HOURS", "NEWS_LOCKOUT_MINUTES",
            "LOCAL_LLM_CONTEXT_SIZE", "LLM_MAX_CONCURRENCY", "MAX_ORDER_DEVIATION_POINTS",
            "DYNAMIC_MARKET_MAX_SYMBOLS",
            "SAME_THESIS_REENTRY_MIN_BARS",
            "FAILED_THESIS_REVERSAL_MAX_AGE_BARS",
        }
        for key, bounds in numeric_ranges.items():
            if key in updates:
                try:
                    value = float(updates[key])
                except (TypeError, ValueError):
                    return JSONResponse({"error": f"{key} must be numeric"}, status_code=422)
                if not bounds[0] <= value <= bounds[1]:
                    return JSONResponse(
                        {"error": f"{key} must be between {bounds[0]} and {bounds[1]}"},
                        status_code=422,
                    )
                if key in integer_keys and not value.is_integer():
                    return JSONResponse({"error": f"{key} must be a whole number"}, status_code=422)
        first_trigger_r = float(
            updates.get("PROFIT_LOCK_TRIGGER_R", settings.profit_lock_trigger_r)
        )
        first_trigger_usd = float(
            updates.get(
                "PROFIT_LOCK_TRIGGER_USD", settings.profit_lock_trigger_usd
            )
        )
        first_floor_usd = float(
            updates.get("PROFIT_LOCK_FLOOR_USD", settings.profit_lock_floor_usd)
        )
        mid_trigger_r = float(
            updates.get(
                "PROFIT_LOCK_MID_TRIGGER_R",
                settings.profit_lock_mid_trigger_r,
            )
        )
        mid_trigger_usd = float(
            updates.get(
                "PROFIT_LOCK_MID_TRIGGER_USD",
                settings.profit_lock_mid_trigger_usd,
            )
        )
        final_trigger_usd = float(
            updates.get(
                "PROFIT_LOCK_FINAL_TRIGGER_USD",
                settings.profit_lock_final_trigger_usd,
            )
        )
        final_floor_usd = float(
            updates.get(
                "PROFIT_LOCK_FINAL_FLOOR_USD",
                settings.profit_lock_final_floor_usd,
            )
        )
        if first_floor_usd >= first_trigger_usd:
            return JSONResponse(
                {
                    "error": (
                        "PROFIT_LOCK_FLOOR_USD must be lower than "
                        "PROFIT_LOCK_TRIGGER_USD to preserve headroom"
                    )
                },
                status_code=422,
            )
        if mid_trigger_r < first_trigger_r or mid_trigger_usd < first_trigger_usd:
            return JSONResponse(
                {
                    "error": (
                        "Mid profit-lock triggers cannot be lower than the "
                        "first-tier triggers"
                    )
                },
                status_code=422,
            )
        if (
            final_trigger_usd < mid_trigger_usd
            or final_floor_usd >= final_trigger_usd
        ):
            return JSONResponse(
                {
                    "error": (
                        "Final profit-lock trigger must follow the mid tier "
                        "and remain above its protected floor"
                    )
                },
                status_code=422,
            )
        if "LOCAL_LLM_URL" in updates:
            try:
                parsed = urlparse(updates["LOCAL_LLM_URL"])
                local_url = (
                    parsed.scheme in {"http", "https"}
                    and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                    and bool(parsed.port)
                )
            except ValueError:
                local_url = False
            if not local_url:
                return JSONResponse({"error": "Only a local LLM URL with an explicit port is allowed"}, status_code=422)

        if (
            "LLM_PROVIDER" in updates
            and updates["LLM_PROVIDER"].lower()
            not in {"deterministic", "local", "openai"}
        ):
            return JSONResponse(
                {"error": "LLM_PROVIDER must be deterministic, local, or openai"},
                status_code=422,
            )
        if (
            "OPENAI_REASONING_EFFORT" in updates
            and updates["OPENAI_REASONING_EFFORT"].lower()
            not in {"none", "low", "medium", "high", "xhigh", "max"}
        ):
            return JSONResponse(
                {"error": "OPENAI_REASONING_EFFORT is invalid"}, status_code=422
            )
        if (
            "LOCAL_LLM_REQUIRED_QUANTIZATION" in updates
            and updates["LOCAL_LLM_REQUIRED_QUANTIZATION"].upper()
            not in LOCAL_LLM_QUANTIZATION_PROFILES
        ):
            return JSONResponse(
                {
                    "error": (
                        "LOCAL_LLM_REQUIRED_QUANTIZATION must be "
                        "AUTO, Q4_K_M, Q6_K, or Q8_0"
                    )
                },
                status_code=422,
            )

        boolean_keys = {
            "AUTO_START_MONITORING", "LOCAL_LLM_STRUCTURED_OUTPUT",
            "SCREENSHOTS_ENABLED", "DRY_RUN", "SESSION_FILTER_ENABLED",
            "WEEKEND_TRADING_ENABLED", "CRYPTO_ONLY_ON_WEEKEND",
            "AUTO_CLOSE_TARGET_PROFIT_ENABLED", "AUTO_CLOSE_TARGET_LOSS_ENABLED",
            "REQUIRE_NEWS_CALENDAR",
            "DYNAMIC_MARKET_SELECTION_ENABLED", "ADAPTIVE_REVERSAL_ENABLED",
            "FAILED_THESIS_REVERSAL_ENABLED",
            "ENTRY_REQUIRE_ADX_RISING",
            "RETEST_CONTINUATION_ENABLED",
            "FAST_EXIT_REVIEW_ENABLED",
            "MICRO_PROFIT_PROTECTION_ENABLED", "PROFIT_LOCK_ENABLED",
            "PROFIT_GIVEBACK_ENABLED",
        }
        invalid_boolean = next(
            (key for key in boolean_keys if key in updates and updates[key].lower() not in {"true", "false"}),
            None,
        )
        if invalid_boolean:
            return JSONResponse({"error": f"{invalid_boolean} must be true or false"}, status_code=422)
        reversal_enabled = updates.get(
            "FAILED_THESIS_REVERSAL_ENABLED",
            str(settings.failed_thesis_reversal_enabled),
        ).lower() == "true"
        reversal_min_confidence = float(
            updates.get(
                "FAILED_THESIS_REVERSAL_MIN_CONFIDENCE",
                settings.failed_thesis_reversal_min_confidence,
            )
        )
        global_confidence = float(
            updates.get("CONFIDENCE_THRESHOLD", settings.confidence_threshold)
        )
        if reversal_enabled and reversal_min_confidence >= global_confidence:
            return JSONResponse(
                {
                    "error": (
                        "FAILED_THESIS_REVERSAL_MIN_CONFIDENCE must be lower "
                        "than CONFIDENCE_THRESHOLD when the exception is enabled"
                    )
                },
                status_code=422,
            )
        chase_confidence = float(
            updates.get(
                "ENTRY_STRONG_ALIGNMENT_CHASE_MIN_CONFIDENCE",
                settings.entry_strong_alignment_chase_min_confidence,
            )
        )
        if chase_confidence < global_confidence:
            return JSONResponse(
                {
                    "error": (
                        "ENTRY_STRONG_ALIGNMENT_CHASE_MIN_CONFIDENCE must be "
                        "at least CONFIDENCE_THRESHOLD"
                    )
                },
                status_code=422,
            )
        normal_chase_limit = float(
            updates.get(
                "ENTRY_MAX_CANDLE_RANGE_ATR",
                settings.entry_max_candle_range_atr,
            )
        )
        aligned_chase_limit = float(
            updates.get(
                "ENTRY_STRONG_ALIGNMENT_CHASE_MAX_EXTENSION_ATR",
                settings.entry_strong_alignment_chase_max_extension_atr,
            )
        )
        if aligned_chase_limit < normal_chase_limit:
            return JSONResponse(
                {
                    "error": (
                        "ENTRY_STRONG_ALIGNMENT_CHASE_MAX_EXTENSION_ATR must "
                        "be at least ENTRY_MAX_CANDLE_RANGE_ATR"
                    )
                },
                status_code=422,
            )
        if (
            "POSITION_EXIT_REVIEW_TIMEFRAME" in updates
            and updates["POSITION_EXIT_REVIEW_TIMEFRAME"].upper()
            not in {"M1", "M5"}
        ):
            return JSONResponse(
                {"error": "POSITION_EXIT_REVIEW_TIMEFRAME must be M1 or M5"},
                status_code=422,
            )
        # Preserve comments, credentials and ordering while replacing only
        # explicitly allowed values.
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        seen = set()
        new_lines = []
        for line in lines:
            key = line.partition("=")[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                new_lines.append(line)
        for key, value in updates.items():
            if key not in seen:
                new_lines.append(f"{key}={value}")
        temp_path = env_path.with_suffix(env_path.suffix + ".tmp")
        temp_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        temp_path.replace(env_path)
        hot_applied = []
        if "MICRO_PROFIT_PROTECTION_ENABLED" in updates:
            # This isolated boolean is safe to change while the engine is
            # running: the position-protection loop reads it atomically on
            # every poll. Hot-applying it also keeps GET /api/config and the
            # refreshed settings form consistent with the persisted value.
            object.__setattr__(
                settings,
                "micro_profit_protection_enabled",
                updates["MICRO_PROFIT_PROTECTION_ENABLED"].lower() == "true",
            )
            hot_applied.append("MICRO_PROFIT_PROTECTION_ENABLED")
        logger.info("Configuration saved (%s keys).", len(updates))
        return {
            "status": "saved",
            "keys_updated": len(updates),
            "hot_applied": hot_applied,
            "restart_required": any(key not in hot_applied for key in updates),
        }
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

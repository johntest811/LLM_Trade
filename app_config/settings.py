"""Environment-backed application configuration.

Trading is intentionally conservative by default: the engine may monitor and
manage positions, but live autonomy requires explicit account-bound
authorization. Credentials remain in ``.env`` and are never exposed by the
API.
"""

from __future__ import annotations

import os
import hashlib
import json
import re
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from app_config.paths import ENV_PATH


_ENV_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_FINGERPRINT_EXCLUDED_FIELDS = frozenset(
    {
        "mt5_account",
        "mt5_password",
        "mt5_path",
        "openai_api_key",
        "openai_organization",
        "openai_project",
    }
)


def duplicate_env_keys(path: Path = ENV_PATH) -> Dict[str, List[int]]:
    """Return duplicated dotenv keys and their line numbers without values."""
    occurrences: Dict[str, List[int]] = {}
    if not path.exists():
        return {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = _ENV_KEY_PATTERN.match(line)
        if match:
            occurrences.setdefault(match.group(1), []).append(line_number)
    return {
        key: line_numbers
        for key, line_numbers in occurrences.items()
        if len(line_numbers) > 1
    }


_duplicate_keys = duplicate_env_keys()
if _duplicate_keys:
    details = ", ".join(
        f"{key} (lines {','.join(str(value) for value in line_numbers)})"
        for key, line_numbers in sorted(_duplicate_keys.items())
    )
    raise RuntimeError(f"Duplicate .env configuration keys are not allowed: {details}")

load_dotenv(ENV_PATH, override=False)


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> List[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    return value if value in choices else default


@dataclass(frozen=True)
class AppConfig:
    # LLM / decision service
    llm_provider: str = _choice(
        "LLM_PROVIDER", "deterministic", {"deterministic", "local", "openai"}
    )
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "qwen/qwen3.5-9b")
    local_llm_required_quantization: str = os.getenv(
        "LOCAL_LLM_REQUIRED_QUANTIZATION", ""
    ).strip().upper()
    local_llm_url: str = os.getenv(
        "LOCAL_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions"
    )
    local_llm_temperature: float = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.0"))
    local_llm_top_p: float = float(os.getenv("LOCAL_LLM_TOP_P", "0.8"))
    local_llm_seed: int = int(os.getenv("LOCAL_LLM_SEED", "42"))
    local_llm_context_size: int = int(os.getenv("LOCAL_LLM_CONTEXT_SIZE", "8192"))
    local_llm_timeout: float = float(os.getenv("LOCAL_LLM_TIMEOUT", "45.0"))
    local_llm_max_retries: int = int(os.getenv("LOCAL_LLM_MAX_RETRIES", "2"))
    local_llm_structured_output: bool = _bool("LOCAL_LLM_STRUCTURED_OUTPUT", True)
    openai_api_key: str = field(
        default=os.getenv("OPENAI_API_KEY", ""), repr=False
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.6-terra").strip()
    openai_base_url: str = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    ).rstrip("/")
    openai_reasoning_effort: str = _choice(
        "OPENAI_REASONING_EFFORT",
        "low",
        {"none", "low", "medium", "high", "xhigh", "max"},
    )
    openai_timeout: float = float(os.getenv("OPENAI_TIMEOUT", "20.0"))
    openai_max_retries: int = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "1")))
    openai_organization: str = field(
        default=os.getenv("OPENAI_ORGANIZATION", "").strip(), repr=False
    )
    openai_project: str = field(
        default=os.getenv("OPENAI_PROJECT", "").strip(), repr=False
    )
    llm_max_concurrency: int = max(
        1,
        min(
            6,
            int(
                os.getenv(
                    "LLM_MAX_CONCURRENCY",
                    "3" if os.getenv("LLM_PROVIDER", "local").strip().lower() == "openai" else "1",
                )
            ),
        ),
    )
    # Analyze every configured market deterministically, but reserve the
    # comparatively expensive model lane for the strongest current setups.
    # This prevents a serial local model queue from making later decisions
    # stale before execution.
    llm_entry_candidates_per_bar: int = max(
        1, min(12, int(os.getenv("LLM_ENTRY_CANDIDATES_PER_BAR", "3")))
    )

    @property
    def decision_model(self) -> str:
        if self.llm_provider == "deterministic":
            return "rules-v2-adaptive"
        return self.openai_model if self.llm_provider == "openai" else self.local_llm_model

    def sanitized_snapshot(self) -> Dict[str, object]:
        """Return the effective configuration without credentials or account IDs."""
        snapshot: Dict[str, object] = {}
        for definition in dataclass_fields(self):
            if definition.name in _FINGERPRINT_EXCLUDED_FIELDS:
                continue
            value = getattr(self, definition.name)
            snapshot[definition.name] = list(value) if isinstance(value, list) else value
        return snapshot

    @property
    def config_fingerprint(self) -> str:
        """Stable ID for the exact non-secret configuration used by a run."""
        payload = json.dumps(
            self.sanitized_snapshot(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    # MT5 account / terminal
    mt5_account: int | None = int(os.getenv("MT5_ACCOUNT")) if os.getenv("MT5_ACCOUNT") else None
    mt5_password: str = os.getenv("MT5_PASSWORD", "")
    mt5_server: str = os.getenv("MT5_SERVER", "")
    mt5_path: str = os.getenv("MT5_PATH", "")
    expected_broker: str = os.getenv("EXPECTED_BROKER", "Pepperstone")
    strategy_magic: int = int(os.getenv("STRATEGY_MAGIC", "202600"))
    order_comment: str = os.getenv("ORDER_COMMENT", "LLM-Terminal")[:31]

    # Simulation is independent of the broker account type. DRY_RUN=False
    # sends orders to whichever Pepperstone account is active in MT5.
    dry_run: bool = _bool("DRY_RUN", True)
    allow_new_trades_on_start: bool = _bool("ALLOW_NEW_TRADES_ON_START", False)
    manage_external_positions: bool = _bool("MANAGE_EXTERNAL_POSITIONS", False)
    require_sl_tp: bool = _bool("REQUIRE_SL_TP", True)

    # Markets and cadence
    trading_timeframe: str = os.getenv("TRADING_TIMEFRAME", "M5").upper()
    trading_symbols: List[str] = field(
        default_factory=lambda: _csv("TRADING_SYMBOLS", "USDJPY,USDCAD,CADJPY")
    )
    dynamic_market_selection_enabled: bool = _bool(
        "DYNAMIC_MARKET_SELECTION_ENABLED", True
    )
    market_candidate_symbols: List[str] = field(
        default_factory=lambda: _csv(
            "MARKET_CANDIDATE_SYMBOLS",
            "USDJPY,USDCAD,CADJPY,EURUSD,GBPUSD,AUDUSD,NZDUSD,USDCHF,EURJPY,GBPJPY,EURGBP,AUDCAD",
        )
    )
    dynamic_market_max_symbols: int = max(
        1, min(12, int(os.getenv("DYNAMIC_MARKET_MAX_SYMBOLS", "6")))
    )
    market_selection_refresh_seconds: float = max(
        30.0, float(os.getenv("MARKET_SELECTION_REFRESH_SECONDS", "60"))
    )
    market_performance_lookback: int = max(
        3, int(os.getenv("MARKET_PERFORMANCE_LOOKBACK", "8"))
    )
    market_performance_min_trades: int = max(
        3, int(os.getenv("MARKET_PERFORMANCE_MIN_TRADES", "4"))
    )
    market_performance_block_expectancy_r: float = float(
        os.getenv("MARKET_PERFORMANCE_BLOCK_EXPECTANCY_R", "-0.35")
    )
    market_performance_block_profit_factor: float = max(
        0.0, float(os.getenv("MARKET_PERFORMANCE_BLOCK_PROFIT_FACTOR", "0.60"))
    )
    auto_start_monitoring: bool = _bool("AUTO_START_MONITORING", True)
    autonomous_health_interval_seconds: float = max(
        5.0, float(os.getenv("AUTONOMOUS_HEALTH_INTERVAL_SECONDS", "15"))
    )
    engine_heartbeat_stale_seconds: float = max(
        15.0, float(os.getenv("ENGINE_HEARTBEAT_STALE_SECONDS", "30"))
    )
    max_entry_decision_age_seconds: float = max(
        10.0,
        min(
            240.0, float(os.getenv("MAX_ENTRY_DECISION_AGE_SECONDS", "45"))
        ),
    )
    max_entry_bar_age_seconds: float = max(
        10.0,
        min(240.0, float(os.getenv("MAX_ENTRY_BAR_AGE_SECONDS", "120"))),
    )
    decision_provider_failure_threshold: int = max(
        1,
        min(
            10, int(os.getenv("DECISION_PROVIDER_FAILURE_THRESHOLD", "2"))
        ),
    )
    analysis_interval_seconds: int = int(os.getenv("ANALYSIS_INTERVAL_SECONDS", "60"))
    decision_poll_seconds: float = max(
        0.5, float(os.getenv("DECISION_POLL_SECONDS", "1.0"))
    )
    analysis_history_bars: int = int(os.getenv("ANALYSIS_HISTORY_BARS", "260"))
    max_tick_age_seconds: float = float(os.getenv("MAX_TICK_AGE_SECONDS", "10.0"))
    reconcile_interval_seconds: int = int(os.getenv("RECONCILE_INTERVAL_SECONDS", "30"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    allow_strong_countertrend_entries: bool = _bool(
        "ALLOW_STRONG_COUNTERTREND_ENTRIES", False
    )
    countertrend_min_confidence: float = float(
        os.getenv("COUNTERTREND_MIN_CONFIDENCE", "0.85")
    )
    countertrend_min_adx: float = float(os.getenv("COUNTERTREND_MIN_ADX", "34.5"))
    countertrend_min_confluence: float = float(
        os.getenv("COUNTERTREND_MIN_CONFLUENCE", "50.0")
    )
    countertrend_sell_min_rsi: float = float(
        os.getenv("COUNTERTREND_SELL_MIN_RSI", "15.0")
    )
    countertrend_buy_max_rsi: float = float(
        os.getenv("COUNTERTREND_BUY_MAX_RSI", "85.0")
    )
    breakout_min_adx: float = float(os.getenv("BREAKOUT_MIN_ADX", "25.0"))
    entry_min_adx: float = float(os.getenv("ENTRY_MIN_ADX", "25.0"))
    entry_require_adx_rising: bool = _bool("ENTRY_REQUIRE_ADX_RISING", True)
    entry_adx_decline_tolerance: float = max(
        0.0, float(os.getenv("ENTRY_ADX_DECLINE_TOLERANCE", "0.50"))
    )
    entry_min_opposing_distance_atr: float = float(
        os.getenv("ENTRY_MIN_OPPOSING_DISTANCE_ATR", "0.75")
    )
    entry_max_candle_range_atr: float = max(
        0.0, float(os.getenv("ENTRY_MAX_CANDLE_RANGE_ATR", "1.50"))
    )
    entry_max_execution_drift_atr: float = max(
        0.0, float(os.getenv("ENTRY_MAX_EXECUTION_DRIFT_ATR", "0.25"))
    )
    breakout_min_displacement_atr: float = max(
        0.0, float(os.getenv("BREAKOUT_MIN_DISPLACEMENT_ATR", "0.10"))
    )
    # Breakout-only entries have no BOS, CHoCH, or verified retest to anchor
    # the entry.  Give that weaker trigger its own quality and macro-strength
    # requirements instead of raising the floor for every strategy.
    breakout_min_quality_score: float = min(
        100.0,
        max(0.0, float(os.getenv("BREAKOUT_MIN_QUALITY_SCORE", "65.0"))),
    )
    breakout_macro_min_adx: float = max(
        0.0, float(os.getenv("BREAKOUT_MACRO_MIN_ADX", "20.0"))
    )
    breakout_strong_lower_adx: float = max(
        0.0, float(os.getenv("BREAKOUT_STRONG_LOWER_ADX", "30.0"))
    )
    overextension_rsi_high: float = float(
        os.getenv("OVEREXTENSION_RSI_HIGH", "65.0")
    )
    overextension_rsi_low: float = float(
        os.getenv("OVEREXTENSION_RSI_LOW", "35.0")
    )
    same_thesis_reentry_min_bars: int = max(
        1, int(os.getenv("SAME_THESIS_REENTRY_MIN_BARS", "2"))
    )
    retest_continuation_enabled: bool = _bool(
        "RETEST_CONTINUATION_ENABLED", True
    )
    retest_min_resumption_atr: float = max(
        0.01, float(os.getenv("RETEST_MIN_RESUMPTION_ATR", "0.10"))
    )
    shadow_symbols: List[str] = field(
        default_factory=lambda: _csv("SHADOW_SYMBOLS", "")
    )
    confirmation_min_adx: float = float(
        os.getenv("CONFIRMATION_MIN_ADX", "18.0")
    )
    entry_min_h4_adx: float = max(
        0.0, float(os.getenv("ENTRY_MIN_H4_ADX", "18.0"))
    )
    overextension_stoch_high: float = float(
        os.getenv("OVEREXTENSION_STOCH_HIGH", "90.0")
    )
    overextension_stoch_low: float = float(
        os.getenv("OVEREXTENSION_STOCH_LOW", "10.0")
    )
    overextension_min_band_overshoot_atr: float = max(
        0.0,
        float(os.getenv("OVEREXTENSION_MIN_BAND_OVERSHOOT_ATR", "0.15")),
    )
    adaptive_reversal_enabled: bool = _bool("ADAPTIVE_REVERSAL_ENABLED", True)
    adaptive_reversal_min_adx: float = float(
        os.getenv("ADAPTIVE_REVERSAL_MIN_ADX", "25.0")
    )
    range_reversion_enabled: bool = _bool("RANGE_REVERSION_ENABLED", True)
    range_band_tolerance_atr: float = max(
        0.0, float(os.getenv("RANGE_BAND_TOLERANCE_ATR", "0.10"))
    )
    range_max_band_overshoot_atr: float = max(
        0.0, float(os.getenv("RANGE_MAX_BAND_OVERSHOOT_ATR", "0.35"))
    )
    range_buy_max_rsi: float = float(os.getenv("RANGE_BUY_MAX_RSI", "35.0"))
    range_sell_min_rsi: float = float(os.getenv("RANGE_SELL_MIN_RSI", "65.0"))
    range_buy_max_stoch: float = float(
        os.getenv("RANGE_BUY_MAX_STOCH", "25.0")
    )
    range_sell_min_stoch: float = float(
        os.getenv("RANGE_SELL_MIN_STOCH", "75.0")
    )
    range_min_reversal_body_atr: float = max(
        0.0, float(os.getenv("RANGE_MIN_REVERSAL_BODY_ATR", "0.05"))
    )
    range_min_target_distance_atr: float = max(
        0.0, float(os.getenv("RANGE_MIN_TARGET_DISTANCE_ATR", "0.75"))
    )
    range_confirmation_max_adx: float = max(
        0.0, float(os.getenv("RANGE_CONFIRMATION_MAX_ADX", "24.0"))
    )
    range_h4_opposition_min_adx: float = max(
        0.0, float(os.getenv("RANGE_H4_OPPOSITION_MIN_ADX", "25.0"))
    )
    range_invalidation_atr: float = max(
        0.0, float(os.getenv("RANGE_INVALIDATION_ATR", "0.25"))
    )
    market_shock_range_atr: float = float(
        os.getenv("MARKET_SHOCK_RANGE_ATR", "2.75")
    )
    market_shock_gap_atr: float = float(
        os.getenv("MARKET_SHOCK_GAP_ATR", "1.25")
    )

    # Position sizing and portfolio safety
    # Legacy compatibility fallback only. The live path must use broker-native
    # risk sizing and may never substitute this value for failed validation.
    default_lot_size: float = float(os.getenv("DEFAULT_LOT_SIZE", "0.01"))
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "2"))
    risk_percent: float = float(os.getenv("RISK_PERCENT", "1.0"))
    manual_override_max_risk_pct: float = float(
        os.getenv("MANUAL_OVERRIDE_MAX_RISK_PCT", "7.0")
    )
    max_portfolio_risk_pct: float = float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "3.0"))
    max_margin_usage_pct: float = float(os.getenv("MAX_MARGIN_USAGE_PCT", "35.0"))
    max_daily_loss_usd: float = float(os.getenv("MAX_DAILY_LOSS_USD", "1.00"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
    drawdown_entry_lock_enabled: bool = _bool("DRAWDOWN_ENTRY_LOCK_ENABLED", True)
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "8.0"))
    max_spread_pips: float = float(os.getenv("MAX_SPREAD_PIPS", "3.0"))
    max_crypto_spread_bps: float = float(os.getenv("MAX_CRYPTO_SPREAD_BPS", "30.0"))
    max_spread_to_stop_pct: float = float(os.getenv("MAX_SPREAD_TO_STOP_PCT", "20.0"))
    # The stop-risk budget includes adverse fill within the submitted MT5
    # deviation and an explicit round-turn fee estimate.  FX defaults to a
    # conservative Razor-style commission; other CFD classes stay at zero
    # until their account-specific fee schedule is configured.
    max_order_deviation_points: int = int(os.getenv("MAX_ORDER_DEVIATION_POINTS", "20"))
    fx_round_turn_cost_usd_per_lot: float = float(
        os.getenv("FX_ROUND_TURN_COST_USD_PER_LOT", "7.0")
    )
    crypto_round_turn_cost_usd_per_lot: float = float(
        os.getenv("CRYPTO_ROUND_TURN_COST_USD_PER_LOT", "0.0")
    )
    cfd_round_turn_cost_usd_per_lot: float = float(
        os.getenv("CFD_ROUND_TURN_COST_USD_PER_LOT", "0.0")
    )
    fixed_execution_cost_usd: float = float(os.getenv("FIXED_EXECUTION_COST_USD", "0.0"))
    min_risk_reward_ratio: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.47"))
    min_free_margin_usd: float = float(os.getenv("MIN_FREE_MARGIN_USD", "5.0"))
    min_margin_level_pct: float = float(os.getenv("MIN_MARGIN_LEVEL_PCT", "250.0"))
    loss_cooldown_minutes: int = int(os.getenv("LOSS_COOLDOWN_MINUTES", "60"))
    # One-, two-, and three-loss adaptive penalties all expire after this
    # review window.  Keeping every level bounded prevents a micro account
    # from becoming permanently unable to place the broker minimum volume.
    loss_streak_pause_hours: int = int(os.getenv("LOSS_STREAK_PAUSE_HOURS", "24"))
    news_lockout_minutes: int = int(os.getenv("NEWS_LOCKOUT_MINUTES", "30"))
    require_news_calendar: bool = _bool("REQUIRE_NEWS_CALENDAR", False)

    # Deterministic order plan. The LLM selects direction; code owns levels.
    plan_stop_atr: float = float(os.getenv("PLAN_STOP_ATR", "2.0"))
    plan_target_rr: float = float(os.getenv("PLAN_TARGET_RR", "1.8"))
    plan_net_rr_buffer: float = float(os.getenv("PLAN_NET_RR_BUFFER", "0.05"))
    plan_max_cost_target_extension_r: float = max(
        0.0, float(os.getenv("PLAN_MAX_COST_TARGET_EXTENSION_R", "0.50"))
    )
    plan_target_buffer_atr: float = max(
        0.0, float(os.getenv("PLAN_TARGET_BUFFER_ATR", "0.10"))
    )
    require_technical_target: bool = _bool("REQUIRE_TECHNICAL_TARGET", True)

    # Exit policy. Dollar targets are optional caps/filters. A target-profit
    # requirement never increases risk or stretches a technical target.
    auto_close_profit_enabled: bool = _bool("AUTO_CLOSE_TARGET_PROFIT_ENABLED", False)
    auto_close_profit_usd: float = float(os.getenv("AUTO_CLOSE_TARGET_PROFIT_USD", "0.50"))
    auto_close_loss_enabled: bool = _bool("AUTO_CLOSE_TARGET_LOSS_ENABLED", False)
    auto_close_loss_usd: float = float(os.getenv("AUTO_CLOSE_TARGET_LOSS_USD", "0.25"))
    fast_exit_review_enabled: bool = _bool("FAST_EXIT_REVIEW_ENABLED", True)
    position_exit_review_timeframe: str = os.getenv(
        "POSITION_EXIT_REVIEW_TIMEFRAME", "M1"
    ).strip().upper()
    # Master switch for the optional small-dollar peak/floor behavior. Hard
    # stops and structure-based exits remain active when this is disabled.
    micro_profit_protection_enabled: bool = _bool(
        "MICRO_PROFIT_PROTECTION_ENABLED", False
    )
    early_profit_lock_enabled: bool = _bool("EARLY_PROFIT_LOCK_ENABLED", True)
    early_profit_lock_trigger_usd: float = max(
        0.01, float(os.getenv("EARLY_PROFIT_LOCK_TRIGGER_USD", "0.12"))
    )
    early_profit_lock_floor_usd: float = min(
        max(0.0, early_profit_lock_trigger_usd - 0.01),
        max(0.0, float(os.getenv("EARLY_PROFIT_LOCK_FLOOR_USD", "0.03"))),
    )
    profit_lock_enabled: bool = _bool("PROFIT_LOCK_ENABLED", True)
    profit_lock_trigger_r: float = max(
        0.1, float(os.getenv("PROFIT_LOCK_TRIGGER_R", "0.75"))
    )
    profit_lock_min_live_fraction: float = min(
        1.0,
        max(0.1, float(os.getenv("PROFIT_LOCK_MIN_LIVE_FRACTION", "0.75"))),
    )
    profit_lock_floor_usd: float = max(
        0.0, float(os.getenv("PROFIT_LOCK_FLOOR_USD", "0.03"))
    )
    profit_lock_trigger_usd: float = max(
        0.0, float(os.getenv("PROFIT_LOCK_TRIGGER_USD", "0.20"))
    )
    profit_giveback_enabled: bool = _bool("PROFIT_GIVEBACK_ENABLED", True)
    profit_giveback_trigger_r: float = max(
        0.1, float(os.getenv("PROFIT_GIVEBACK_TRIGGER_R", "0.50"))
    )
    profit_giveback_trigger_usd: float = max(
        0.0, float(os.getenv("PROFIT_GIVEBACK_TRIGGER_USD", "0.20"))
    )
    profit_giveback_fraction: float = min(
        0.95, max(0.05, float(os.getenv("PROFIT_GIVEBACK_FRACTION", "0.50")))
    )
    profit_peak_persist_delta_usd: float = max(
        0.01, float(os.getenv("PROFIT_PEAK_PERSIST_DELTA_USD", "0.01"))
    )
    breakeven_trigger_r: float = float(os.getenv("BREAKEVEN_TRIGGER_R", "1.0"))
    breakeven_buffer_pips: float = float(os.getenv("BREAKEVEN_BUFFER_PIPS", "0.2"))
    trailing_trigger_r: float = float(os.getenv("TRAILING_TRIGGER_R", "1.5"))
    trailing_distance_r: float = float(os.getenv("TRAILING_DISTANCE_R", "0.75"))
    position_stagnation_exit_enabled: bool = _bool(
        "POSITION_STAGNATION_EXIT_ENABLED", True
    )
    position_stagnation_bars: int = max(
        3, int(os.getenv("POSITION_STAGNATION_BARS", "12"))
    )
    position_stagnation_max_peak_r: float = max(
        0.0, float(os.getenv("POSITION_STAGNATION_MAX_PEAK_R", "0.15"))
    )
    # A structure event is intentionally not the only possible exit fact.
    # Two completed M1 bars with adverse trend and momentum may cut a losing
    # position before its full broker stop, while a single noisy pullback or a
    # position that still has headroom cannot trigger this path.
    momentum_deterioration_exit_enabled: bool = _bool(
        "MOMENTUM_DETERIORATION_EXIT_ENABLED", True
    )
    momentum_deterioration_confirm_bars: int = max(
        2,
        min(
            5,
            int(os.getenv("MOMENTUM_DETERIORATION_CONFIRM_BARS", "2")),
        ),
    )
    momentum_deterioration_min_loss_r: float = min(
        0.95,
        max(
            0.10,
            float(os.getenv("MOMENTUM_DETERIORATION_MIN_LOSS_R", "0.35")),
        ),
    )
    momentum_deterioration_min_body_atr: float = max(
        0.0,
        float(os.getenv("MOMENTUM_DETERIORATION_MIN_BODY_ATR", "0.15")),
    )

    # Rejected-signal counterfactual evaluation. These observations never
    # place orders or relax a gate; they provide broker-price evidence for
    # later configuration changes.
    shadow_outcomes_enabled: bool = _bool("SHADOW_OUTCOMES_ENABLED", True)
    shadow_outcome_horizon_minutes: int = max(
        15, min(1440, int(os.getenv("SHADOW_OUTCOME_HORIZON_MINUTES", "60")))
    )
    shadow_outcome_poll_seconds: float = max(
        10.0, float(os.getenv("SHADOW_OUTCOME_POLL_SECONDS", "30"))
    )

    # Time/session behavior
    session_filter_enabled: bool = _bool("SESSION_FILTER_ENABLED", False)
    allowed_sessions_utc: List[str] = field(
        default_factory=lambda: _csv("ALLOWED_SESSIONS_UTC", "00:00-23:59")
    )
    weekend_trading_enabled: bool = _bool("WEEKEND_TRADING_ENABLED", False)
    crypto_only_on_weekend: bool = _bool("CRYPTO_ONLY_ON_WEEKEND", True)
    weekend_symbols: List[str] = field(
        default_factory=lambda: _csv("WEEKEND_SYMBOLS", "BTCUSD,ETHUSD")
    )
    screenshots_enabled: bool = _bool("SCREENSHOTS_ENABLED", False)


settings = AppConfig()

"""
risk/manager.py — Complete Risk Management Engine

Validates every LLM decision before it reaches the execution layer.
Each check is independent; all failures are logged with a clear rejection reason.
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app_config.settings import settings
from risk.execution_costs import estimate_execution_risk
from risk.instruments import downside_risk_usd, is_crypto_symbol, validate_spread

from mt5.safe_api import mt5

logger = logging.getLogger("TradingSystem.RiskManager")


def _parse_utc_datetime(value: Any) -> Optional[datetime]:
    """Parse broker/database timestamps into an aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _effective_losing_streak(
    trade_history: Optional[List[Dict[str, Any]]],
    now_utc: datetime,
    reset_after_utc: Any = None,
) -> int:
    """Return recent consecutive losses within the configured penalty window.

    The old implementation expired only streaks of three or more.  A one-loss
    or two-loss sizing penalty could therefore remain active indefinitely and,
    on a micro account, make the broker minimum lot impossible to trade.  All
    adaptive streak effects now share the same bounded review window.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)
    reset_after = _parse_utc_datetime(reset_after_utc)

    closed_trades = []
    for sequence, trade in enumerate(trade_history or []):
        status = str(trade.get("status", "")).upper()
        if status != "CLOSED" and "net_profit" not in trade:
            continue
        try:
            profit = float(trade.get("net_profit", trade.get("profit", 0.0)) or 0.0)
        except (TypeError, ValueError):
            logger.warning("Ignoring closed trade with invalid profit while computing loss streak.")
            continue
        if not math.isfinite(profit):
            logger.warning("Ignoring closed trade with non-finite profit while computing loss streak.")
            continue
        closed_at = _parse_utc_datetime(trade.get("close_time", trade.get("time")))
        # An operator reset starts a new streak window without deleting or
        # rewriting broker history. Undated legacy rows cannot safely be
        # classified as post-reset, so they remain outside the new window.
        if reset_after is not None and (
            closed_at is None or closed_at <= reset_after
        ):
            continue
        # Broker-reconciled rows have valid ISO timestamps.  Preserve input
        # ordering as a deterministic fallback for legacy rows that do not.
        sort_time = closed_at or datetime.min.replace(tzinfo=timezone.utc)
        closed_trades.append((sort_time, -sequence, closed_at, profit))

    closed_trades.sort(key=lambda item: (item[0], item[1]), reverse=True)
    streak = 0
    latest_closed_at: Optional[datetime] = None
    for _, _, closed_at, profit in closed_trades:
        if latest_closed_at is None:
            latest_closed_at = closed_at
        if profit < 0:
            streak += 1
        else:
            # A win or scratch trade ends a consecutive losing sequence.
            break

    if streak > 0 and latest_closed_at is not None:
        age = now_utc - latest_closed_at
        penalty_window = timedelta(hours=max(0, settings.loss_streak_pause_hours))
        if age >= penalty_window:
            logger.info(
                "Adaptive Risk: Expired %d-loss penalty after %.1f hours.",
                streak,
                max(0.0, age.total_seconds() / 3600.0),
            )
            return 0
    return streak


# ---------------------------------------------------------------------------
# Result dataclass returned to the caller
# ---------------------------------------------------------------------------
@dataclass
class RiskValidationResult:
    approved: bool
    reason: str
    adjusted_lot: Optional[float] = None  # populated when approved
    quality_score: float = 0.0
    confluence_score: float = 0.0
    estimated_risk_usd: float = 0.0
    estimated_reward_usd: float = 0.0
    risk_percent_balance: float = 0.0
    planned_rr: float = 0.0
    risk_budget_usd: float = 0.0


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------
class RiskManager:
    """
    Validates every LLM decision against 14 independent risk rules before
    execution.  Returns a structured RiskValidationResult so the caller knows
    exactly why a trade was rejected.

    Stateful attributes (daily loss, last loss time) are kept in-process.
    They reset automatically when a new UTC trading day begins.
    """

    def __init__(self) -> None:
        self._reset_daily_state()

    @staticmethod
    def _trend_direction(analysis: Optional[Dict[str, Any]]) -> str:
        """Return the canonical current direction with a legacy fallback."""
        structure = (analysis or {}).get("market_structure", {}) or {}
        value = (
            structure.get("trend_state_direction")
            or structure.get("trend")
            or "NEUTRAL"
        )
        normalized = str(value).upper()
        return normalized if normalized in {"BULLISH", "BEARISH"} else "NEUTRAL"

    @classmethod
    def _resolve_strategy_mode(
        cls,
        action: str,
        llm_decision: Dict[str, Any],
        m5_analysis: Optional[Dict[str, Any]],
        m15_analysis: Optional[Dict[str, Any]],
        h1_analysis: Optional[Dict[str, Any]],
        h4_analysis: Optional[Dict[str, Any]],
    ) -> str:
        """Deterministically label model directions for downstream risk rules."""
        supplied = str(
            (llm_decision.get("_strategy") or {}).get("mode", "")
        ).upper()
        allowed = {
            "TREND_CONTINUATION",
            "PULLBACK_RESUMPTION",
            "CONFIRMED_REVERSAL",
        }
        if supplied in allowed:
            return supplied

        expected = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(
            str(action).upper()
        )
        if expected is None:
            return ""
        m5_structure = (m5_analysis or {}).get("market_structure", {}) or {}
        range_setup = m5_structure.get("range_reversion", {}) or {}
        if (
            isinstance(range_setup, dict)
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper()
            == str(action).upper()
        ):
            return "RANGE_REVERSION"
        m15_structure = (m15_analysis or {}).get("market_structure", {}) or {}
        m5_events = [
            event
            for event in m5_structure.get("structure_events", []) or []
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected
        ]
        m15_events = [
            event
            for event in m15_structure.get("structure_events", []) or []
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected
            and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
        ]
        has_choch = any(
            str(event.get("type", "")).upper() == "CHOCH"
            for event in m5_events
        )
        has_bos = any(
            str(event.get("type", "")).upper() == "BOS"
            for event in m5_events
        )
        breakout = str(m5_structure.get("breakout_status", "")).upper()
        has_breakout = expected in breakout and "BREAKOUT" in breakout
        retest = m5_structure.get("retest_continuation")
        has_retest = bool(
            isinstance(retest, dict)
            and str(retest.get("direction", "")).upper() == expected
        )
        m5_state = str(m5_structure.get("trend_state", "")).upper()
        if (
            settings.adaptive_reversal_enabled
            and
            m5_state == f"EARLY_{expected}_REVERSAL"
            and cls._trend_direction(m15_analysis) == expected
            and has_choch
            and (has_bos or m15_events or has_breakout)
        ):
            return "CONFIRMED_REVERSAL"

        if (
            has_retest
            and all(
                cls._trend_direction(analysis) == expected
                for analysis in (m5_analysis, m15_analysis, h1_analysis)
            )
        ):
            return "PULLBACK_RESUMPTION"

        if all(
            cls._trend_direction(analysis) == expected
            for analysis in (m5_analysis, m15_analysis, h1_analysis)
        ):
            return "TREND_CONTINUATION"
        return ""

    @classmethod
    def _requires_strong_countertrend_exception(
        cls,
        strategy_mode: str,
        h1_analysis: Optional[Dict[str, Any]],
        h4_analysis: Optional[Dict[str, Any]],
    ) -> bool:
        """Keep a fully validated reversal out of the continuation-only gate."""
        if str(strategy_mode).upper() in {
            "CONFIRMED_REVERSAL",
            "RANGE_REVERSION",
        }:
            return False
        t_h1 = cls._trend_direction(h1_analysis)
        t_h4 = cls._trend_direction(h4_analysis)
        return (
            (t_h1 == "BULLISH" and t_h4 == "BEARISH")
            or (t_h1 == "BEARISH" and t_h4 == "BULLISH")
        )

    @staticmethod
    def _strong_countertrend_exception(
        action: str,
        llm_decision: Dict[str, Any],
        m5_analysis: Optional[Dict[str, Any]],
        m15_analysis: Optional[Dict[str, Any]],
        h1_analysis: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Allow only high-conviction lower-timeframe alignment against H4."""
        expected_trend = {
            "BUY": "BULLISH",
            "SELL": "BEARISH",
        }.get(str(action).upper())
        if expected_trend is None:
            return False, "entry direction is invalid"

        analyses = {
            "M5": m5_analysis,
            "M15": m15_analysis,
            "H1": h1_analysis,
        }
        trends = {
            timeframe: RiskManager._trend_direction(analysis)
            for timeframe, analysis in analyses.items()
        }
        aligned = all(trend == expected_trend for trend in trends.values())
        try:
            confidence = float(llm_decision.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 1.0:
            confidence /= 100.0
        try:
            adx = float((m5_analysis or {}).get("indicators", {}).get("adx_14", 0.0) or 0.0)
        except (TypeError, ValueError):
            adx = 0.0
        try:
            rsi = float((m5_analysis or {}).get("indicators", {}).get("rsi_14"))
        except (TypeError, ValueError):
            rsi = math.nan

        if expected_trend == "BEARISH":
            rsi_safe = math.isfinite(rsi) and rsi >= settings.countertrend_sell_min_rsi
            rsi_limit = f">={settings.countertrend_sell_min_rsi:.1f}"
        else:
            rsi_safe = math.isfinite(rsi) and rsi <= settings.countertrend_buy_max_rsi
            rsi_limit = f"<={settings.countertrend_buy_max_rsi:.1f}"

        allowed = (
            settings.allow_strong_countertrend_entries
            and aligned
            and confidence >= settings.countertrend_min_confidence
            and adx >= settings.countertrend_min_adx
            and rsi_safe
        )
        detail = (
            f"M5/M15/H1={trends['M5']}/{trends['M15']}/{trends['H1']} "
            f"expected={expected_trend}, confidence={confidence:.0%}/"
            f"{settings.countertrend_min_confidence:.0%}, "
            f"M5 ADX={adx:.1f}/{settings.countertrend_min_adx:.1f}, "
            f"M5 RSI={rsi:.1f}/{rsi_limit}"
        )
        return allowed, detail

    @staticmethod
    def _check_market_shock(
        m5_analysis: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Pause new entries after an abnormal completed M5 candle or gap."""
        indicators = (m5_analysis or {}).get("indicators", {}) or {}
        try:
            range_atr = float(indicators.get("candle_range_atr", 0.0) or 0.0)
            gap_atr = float(indicators.get("opening_gap_atr", 0.0) or 0.0)
        except (TypeError, ValueError):
            range_atr = gap_atr = math.inf

        if (
            not math.isfinite(range_atr)
            or not math.isfinite(gap_atr)
            or range_atr >= settings.market_shock_range_atr
            or gap_atr >= settings.market_shock_gap_atr
        ):
            return False, (
                "REJECTED [Market Shock]: Completed M5 candle is abnormally "
                f"large/gapped ({range_atr:.2f} ATR range, {gap_atr:.2f} ATR gap); "
                "wait for post-shock structure and spreads to stabilize."
            )
        return True, "Market-shock gate passed."

    @staticmethod
    def _check_entry_structure(
        action: str,
        m5_analysis: Optional[Dict[str, Any]],
        m15_analysis: Optional[Dict[str, Any]],
        h1_analysis: Optional[Dict[str, Any]],
        h4_analysis: Optional[Dict[str, Any]],
        confluence_factors: Optional[Dict[str, bool]] = None,
        strategy_mode: str = "",
        market_snapshot: Any = None,
    ) -> Tuple[bool, str]:
        """Reject weak or overextended entries using deterministic evidence.

        LLM prose is deliberately not trusted for BOS/CHoCH claims. The gate
        reads the computed structure events and indicators that were supplied
        to the model, preventing a confident but contradictory explanation
        from authorizing a live order.
        """
        expected = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(str(action).upper())
        if expected is None or not all((m5_analysis, m15_analysis, h1_analysis, h4_analysis)):
            return False, "REJECTED [Structure Gate]: Complete M5/M15/H1/H4 analysis is required."

        def structure(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return (analysis or {}).get("market_structure", {})

        def direction(analysis: Optional[Dict[str, Any]]) -> str:
            data = structure(analysis)
            return str(
                data.get("trend_state_direction") or data.get("trend") or "NEUTRAL"
            ).upper()

        def adx(analysis: Optional[Dict[str, Any]]) -> float:
            try:
                value = float((analysis or {}).get("indicators", {}).get("adx_14", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        directions = {
            "M5": direction(m5_analysis),
            "M15": direction(m15_analysis),
            "H1": direction(h1_analysis),
            "H4": direction(h4_analysis),
        }
        range_mode = str(strategy_mode).upper() == "RANGE_REVERSION"
        m5_structure = structure(m5_analysis)
        range_setup = m5_structure.get("range_reversion", {}) or {}
        if range_mode and not (
            isinstance(range_setup, dict)
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper()
            == str(action).upper()
        ):
            return (
                False,
                "REJECTED [Range Setup]: The deterministic completed-candle "
                "range qualification is absent or points the other way.",
            )

        # M15 is deliberately slower than the entry chart.  Do not reject a
        # fresh M5 BOS merely because the M15 direction label has not yet
        # advanced from NEUTRAL when both macro timeframes already agree and
        # M5/M15 momentum is independently strong.  An opposing M15 direction
        # is never bridged by this exception.
        early_m5_events = [
            event
            for event in m5_structure.get("structure_events", []) or []
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected
        ]
        early_has_bos = any(
            str(event.get("type", "")).upper() == "BOS"
            for event in early_m5_events
        )
        neutral_m15_bridge = (
            not range_mode
            and directions["M15"] == "NEUTRAL"
            and directions["M5"] == expected
            and directions["H1"] == expected
            and directions["H4"] == expected
            and early_has_bos
            and adx(m5_analysis) >= settings.entry_min_adx
            and adx(m15_analysis) >= settings.breakout_min_adx
        )
        for timeframe in (() if range_mode else ("M5", "M15", "H1")):
            if (
                timeframe == "H1"
                and str(strategy_mode).upper() == "CONFIRMED_REVERSAL"
            ):
                continue
            if timeframe == "M15" and neutral_m15_bridge:
                continue
            if directions[timeframe] != expected:
                return (
                    False,
                    f"REJECTED [Structure Gate]: {timeframe} direction "
                    f"{directions[timeframe]} does not confirm {expected}.",
                )

        m5_structure = structure(m5_analysis)
        events = [
            event for event in m5_structure.get("structure_events", [])
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected
        ]
        has_bos = any(str(event.get("type", "")).upper() == "BOS" for event in events)
        has_choch = any(str(event.get("type", "")).upper() == "CHOCH" for event in events)
        breakout = str(m5_structure.get("breakout_status", "")).upper()
        has_breakout = expected in breakout and "BREAKOUT" in breakout
        retest = m5_structure.get("retest_continuation")
        has_retest = bool(
            isinstance(retest, dict)
            and str(retest.get("direction", "")).upper() == expected
        )
        m15_events = [
            event
            for event in structure(m15_analysis).get("structure_events", [])
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected
            and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
        ]
        m5_adx = adx(m5_analysis)
        m15_adx = adx(m15_analysis)
        h1_adx = adx(h1_analysis)
        h4_adx = adx(h4_analysis)
        lower_timeframe_momentum_confirms = (
            (has_retest or has_bos)
            and m5_adx >= settings.breakout_min_adx
            and m15_adx >= settings.confirmation_min_adx
        )
        if (
            str(strategy_mode).upper() not in {
                "CONFIRMED_REVERSAL",
                "RANGE_REVERSION",
            }
            and directions["H4"] == expected
            and h4_adx < settings.entry_min_h4_adx
            and not lower_timeframe_momentum_confirms
        ):
            return (
                False,
                f"REJECTED [Macro Momentum]: H4 direction agrees but ADX "
                f"{h4_adx:.1f} is below {settings.entry_min_h4_adx:.1f}; "
                "the continuation lacks higher-timeframe strength. A verified "
                "pullback or fresh strong M5 BOS is required while H4 momentum "
                "is weak.",
            )
        if settings.entry_require_adx_rising and not range_mode:
            try:
                adx_delta = float(
                    (m5_analysis or {}).get("indicators", {}).get("adx_delta")
                )
            except (TypeError, ValueError):
                adx_delta = math.nan
            decline_limit = -settings.entry_adx_decline_tolerance
            if math.isfinite(adx_delta) and adx_delta < decline_limit:
                return (
                    False,
                    f"REJECTED [ADX Direction]: M5 ADX is falling "
                    f"({adx_delta:+.2f}); allowed noise is "
                    f"{settings.entry_adx_decline_tolerance:.2f}. Wait for "
                    "momentum to stabilize.",
                )

        if str(strategy_mode).upper() == "CONFIRMED_REVERSAL":
            if not has_choch or not (has_bos or m15_events or has_breakout):
                return (
                    False,
                    "REJECTED [Reversal Confirmation]: A deterministic reversal "
                    "requires M5 CHoCH plus M5 BOS, M15 structure, or breakout confirmation.",
                )
            if (
                m5_adx < settings.adaptive_reversal_min_adx
                or m15_adx < settings.confirmation_min_adx
            ):
                return (
                    False,
                    "REJECTED [Reversal Confirmation]: Reversal ADX confirmation "
                    f"is insufficient ({m5_adx:.1f}/{m15_adx:.1f}).",
                )

        # Broad trend alignment is context, not an entry trigger. Every live
        # order requires a fresh event on the completed M5 candle so a model
        # cannot repeatedly trade an old multi-timeframe thesis.
        has_confirmed_choch = bool(has_choch and (m15_events or has_breakout))
        has_range_trigger = bool(
            range_mode
            and isinstance(range_setup, dict)
            and range_setup.get("eligible")
        )
        if not (
            has_bos
            or has_confirmed_choch
            or has_breakout
            or has_retest
            or has_range_trigger
        ):
            return (
                False,
                "REJECTED [Fresh Structure]: No directional M5 BOS, confirmed "
                "CHoCH, breakout, or verified pullback retest exists on the "
                "completed entry candle.",
            )
        if (
            not range_mode
            and has_choch
            and not has_confirmed_choch
            and not has_bos
            and not has_retest
        ):
            return (
                False,
                "REJECTED [Reversal Confirmation]: M5 CHoCH is early evidence "
                "and requires matching M15 structure or breakout confirmation.",
            )
        # A valid BOS or confirmed CHoCH is already a fresh structural trigger.
        # Apply the additional M15 breakout momentum rule only when breakout is
        # the sole qualifying trigger; otherwise a harmless breakout annotation
        # must not invalidate an independently valid setup.
        breakout_only = (
            has_breakout
            and not has_bos
            and not has_confirmed_choch
            and not has_retest
        )
        breakout_momentum_confirms = (
            m5_adx >= settings.entry_min_adx
            and m15_adx >= settings.confirmation_min_adx
            and max(m5_adx, m15_adx) >= settings.breakout_min_adx
        )
        if breakout_only and not breakout_momentum_confirms:
            return (
                False,
                "REJECTED [Breakout Confirmation]: Breakout entry requires "
                f"M5/M15 ADX >= {settings.entry_min_adx:.1f}/"
                f"{settings.confirmation_min_adx:.1f} and at least one >= "
                f"{settings.breakout_min_adx:.1f}; got "
                f"{m5_adx:.1f}/{m15_adx:.1f}.",
            )
        # A raw breakout is materially weaker than a BOS or verified retest.
        # Direction labels alone must not make weak H1/H4 regimes look like
        # strong macro confirmation. Exceptionally strong M5+M15 momentum can
        # still authorize a timely breakout so this does not become a blanket
        # retest-only strategy.
        macro_breakout_weak = (
            h1_adx < settings.breakout_macro_min_adx
            and h4_adx < settings.breakout_macro_min_adx
        )
        strong_lower_breakout = (
            m5_adx >= settings.breakout_strong_lower_adx
            and m15_adx >= settings.breakout_strong_lower_adx
        )
        if breakout_only and macro_breakout_weak and not strong_lower_breakout:
            return (
                False,
                "REJECTED [Breakout Macro Strength]: Breakout is the only "
                "fresh trigger, both H1/H4 ADX values are below "
                f"{settings.breakout_macro_min_adx:.1f} "
                f"({h1_adx:.1f}/{h4_adx:.1f}), and M5/M15 are not both >= "
                f"{settings.breakout_strong_lower_adx:.1f}. Wait for a "
                "verified retest, fresh BOS, or stronger momentum.",
            )
        indicators = (m5_analysis or {}).get("indicators", {})
        try:
            candle_range_atr = float(indicators.get("candle_range_atr"))
        except (TypeError, ValueError):
            candle_range_atr = math.nan
        directional_metrics = []
        direction_multiplier = 1.0 if expected == "BULLISH" else -1.0
        for key in ("candle_body_atr_signed", "candle_return_atr"):
            try:
                value = float(indicators.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                directional_metrics.append(direction_multiplier * value)
        # Old/replayed analyses may not contain directional candle metrics. In
        # that case retain the conservative legacy range behavior. Live
        # analyses distinguish a genuine impulse from a long-wick candle.
        directional_extension_atr = (
            max(directional_metrics)
            if directional_metrics
            else candle_range_atr
        )
        if (
            settings.entry_max_candle_range_atr > 0
            and math.isfinite(candle_range_atr)
            and candle_range_atr > settings.entry_max_candle_range_atr
            and math.isfinite(directional_extension_atr)
            and directional_extension_atr > settings.entry_max_candle_range_atr
        ):
            return (
                False,
                f"REJECTED [Entry Chase]: Completed M5 candle extends "
                f"{directional_extension_atr:.2f} ATR in the entry direction "
                f"({candle_range_atr:.2f} ATR full range); maximum is "
                f"{settings.entry_max_candle_range_atr:.2f} ATR. Wait for a "
                "retest instead of entering after an extended impulse.",
            )
        try:
            current_price = float(indicators.get("current_price"))
            atr = float(indicators.get("atr_14"))
        except (TypeError, ValueError):
            current_price = atr = math.nan
        # MT5 chart candles are bid-based.  Compare the completed bid close to
        # the live bid so the BUY-side spread is not misclassified as price
        # movement.  Spread and execution cost are validated separately.
        market_price = current_price
        try:
            metrics = getattr(market_snapshot, "metrics")
            live_bid = float(getattr(metrics, "bid"))
            if math.isfinite(live_bid) and live_bid > 0:
                market_price = live_bid
        except (AttributeError, TypeError, ValueError):
            pass
        if (
            math.isfinite(current_price)
            and math.isfinite(market_price)
            and math.isfinite(atr)
            and atr > 0
            and settings.entry_max_execution_drift_atr > 0
        ):
            directional_drift = (
                market_price - current_price
                if expected == "BULLISH"
                else current_price - market_price
            )
            if directional_drift > (
                atr * settings.entry_max_execution_drift_atr + 1e-12
            ):
                return (
                    False,
                    f"REJECTED [Execution Drift]: Live bid moved "
                    f"{directional_drift / atr:.2f} ATR beyond the analyzed "
                    f"M5 close; maximum is "
                    f"{settings.entry_max_execution_drift_atr:.2f} ATR. "
                    "Wait for a fresh candle or retest.",
                )
        if (
            math.isfinite(market_price)
            and math.isfinite(atr)
            and atr > 0
            and settings.entry_min_opposing_distance_atr > 0
        ):
            # Decide whether a level was consumed from the completed M5 close,
            # not from a later live quote. A small post-close retest must not
            # resurrect the resistance/supply (or support/demand) that the
            # completed structural trigger already broke. Live-price movement
            # remains bounded independently by the execution-drift gate above.
            structure_reference_price = (
                current_price if math.isfinite(current_price) else market_price
            )
            opposing_distances = []
            if expected == "BEARISH":
                try:
                    support = float(m5_structure.get("support"))
                    if support <= structure_reference_price:
                        opposing_distances.append(max(0.0, market_price - support))
                except (TypeError, ValueError):
                    pass
                for zone in m5_structure.get("demand_zones", []) or []:
                    try:
                        low, high = sorted((float(zone["bottom"]), float(zone["top"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if low <= structure_reference_price <= high:
                        opposing_distances.append(0.0)
                    elif high < structure_reference_price:
                        opposing_distances.append(max(0.0, market_price - high))
                opposing_label = "support/demand"
            else:
                try:
                    resistance = float(m5_structure.get("resistance"))
                    if resistance >= structure_reference_price:
                        opposing_distances.append(max(0.0, resistance - market_price))
                except (TypeError, ValueError):
                    pass
                for zone in m5_structure.get("supply_zones", []) or []:
                    try:
                        low, high = sorted((float(zone["bottom"]), float(zone["top"])))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if low <= structure_reference_price <= high:
                        opposing_distances.append(0.0)
                    elif low > structure_reference_price:
                        opposing_distances.append(max(0.0, low - market_price))
                opposing_label = "resistance/supply"
            nearest = min(opposing_distances) if opposing_distances else math.inf
            minimum = atr * settings.entry_min_opposing_distance_atr
            if nearest < minimum:
                return (
                    False,
                    f"REJECTED [Opposing Zone]: {action} is only "
                    f"{nearest / atr:.2f} ATR from {opposing_label}; minimum is "
                    f"{settings.entry_min_opposing_distance_atr:.2f} ATR.",
                )

        stochastic = indicators.get("stochastic", {}) or {}
        bands = indicators.get("bollinger_bands", {}) or {}
        try:
            stoch_k = float(stochastic.get("k"))
            current = float(market_price)
            outer_band = float(
                bands.get("upper") if expected == "BULLISH" else bands.get("lower")
            )
        except (TypeError, ValueError):
            stoch_k = current = outer_band = math.nan
        try:
            rsi = float(indicators.get("rsi_14"))
        except (TypeError, ValueError):
            rsi = math.nan
        # RSI/stochastic exhaustion is meaningful even just inside the outer
        # Bollinger band when the only entry trigger is an unretested breakout.
        # BOS and retest setups retain the existing, less restrictive
        # band-overshoot rule below.
        patterns = [
            str(pattern).upper()
            for pattern in m5_structure.get("candlestick_patterns", []) or []
        ]
        has_directional_pattern = any(expected in pattern for pattern in patterns)
        exhausted_momentum = (
            math.isfinite(stoch_k)
            and math.isfinite(rsi)
            and (
                (
                    expected == "BULLISH"
                    and stoch_k >= settings.overextension_stoch_high
                    and rsi >= settings.overextension_rsi_high
                )
                or (
                    expected == "BEARISH"
                    and stoch_k <= settings.overextension_stoch_low
                    and rsi <= settings.overextension_rsi_low
                )
            )
        )

        # A same-candle M5 BOS plus breakout can still be the final push of an
        # exhausted move. Trend labels alone did not protect the historical
        # GBPUSD/NZDUSD/USDJPY examples from immediate failure. Preserve timely
        # winners when a directional candle pattern, verified retest, or an
        # actual H1/H4 structure event confirms the continuation; otherwise
        # wait for the retest instead of treating the BOS label as sufficient.
        def _macro_structure_confirms(analysis: Optional[Dict[str, Any]]) -> bool:
            macro = structure(analysis)
            macro_events = macro.get("structure_events", []) or []
            event_confirms = any(
                isinstance(event, dict)
                and str(event.get("direction", "")).upper() == expected
                and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
                for event in macro_events
            )
            macro_breakout = str(macro.get("breakout_status", "")).upper()
            return event_confirms or (
                expected in macro_breakout and "BREAKOUT" in macro_breakout
            )

        exhausted_bos_breakout = (
            has_bos
            and has_breakout
            and not has_retest
            and not has_directional_pattern
            and not _macro_structure_confirms(h1_analysis)
            and not _macro_structure_confirms(h4_analysis)
            and exhausted_momentum
        )
        if exhausted_bos_breakout:
            return (
                False,
                f"REJECTED [Overextension - BOS Breakout Exhaustion]: "
                f"{action} BOS/breakout is unretested while stochastic "
                f"{stoch_k:.1f} and RSI {rsi:.1f} are both extended, with no "
                "directional candle pattern or H1/H4 structure event. Wait "
                "for a verified retest or fresh macro confirmation.",
            )
        breakout_exhausted = (
            breakout_only
            and not has_directional_pattern
            and exhausted_momentum
        )
        if breakout_exhausted:
            return (
                False,
                f"REJECTED [Overextension - Breakout Exhaustion]: {action} "
                "breakout is "
                f"unretested while stochastic {stoch_k:.1f} and RSI "
                f"{rsi:.1f} are both extended. Wait for a verified retest, "
                "fresh BOS, or directional reversal candle.",
            )
        if all(
            math.isfinite(value)
            for value in (stoch_k, current, outer_band, rsi, atr)
        ) and atr > 0:
            band_overshoot = (
                current - outer_band
                if expected == "BULLISH"
                else outer_band - current
            )
            band_overshoot_atr = max(0.0, band_overshoot) / atr
            overextended_buy = (
                expected == "BULLISH"
                and band_overshoot_atr
                >= settings.overextension_min_band_overshoot_atr
                and stoch_k >= settings.overextension_stoch_high
                and rsi >= settings.overextension_rsi_high
            )
            overextended_sell = (
                expected == "BEARISH"
                and band_overshoot_atr
                >= settings.overextension_min_band_overshoot_atr
                and stoch_k <= settings.overextension_stoch_low
                and rsi <= settings.overextension_rsi_low
            )
            if overextended_buy or overextended_sell:
                rsi_label = f"{rsi:.1f}" if math.isfinite(rsi) else "unavailable"
                return (
                    False,
                    f"REJECTED [Overextension]: {action} is beyond the outer "
                    f"Bollinger Band by {band_overshoot_atr:.2f} ATR with "
                    f"stochastic {stoch_k:.1f} and RSI {rsi_label}; wait for a "
                    "retest instead of chasing the breakout.",
                )

        return True, ""

    @staticmethod
    def _check_same_thesis_reentry(
        action: str,
        m5_analysis: Optional[Dict[str, Any]],
        trade_history: Optional[List[Dict[str, Any]]],
    ) -> Tuple[bool, str]:
        """Require genuinely new structure after any same-direction close.

        A profitable close does not make the old thesis new again.  Requiring
        post-close structure prevents an autonomous strategy from repeatedly
        reopening the same signal while price and the evidence set are
        unchanged.
        """
        ordered = []
        for trade in trade_history or []:
            closed_at = _parse_utc_datetime(
                trade.get("close_time") or trade.get("time")
            )
            try:
                pnl = float(
                    trade.get("net_profit", trade.get("profit", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                continue
            direction = str(
                trade.get("direction", trade.get("action", ""))
            ).upper()
            if closed_at is not None and math.isfinite(pnl):
                ordered.append((closed_at, pnl, direction))
        if not ordered:
            return True, ""
        normalized_action = str(action).upper()
        same_direction = [
            item for item in ordered if item[2] == normalized_action
        ]
        if not same_direction:
            return True, ""
        closed_at, pnl, _ = max(same_direction, key=lambda item: item[0])

        analysis_time = _parse_utc_datetime((m5_analysis or {}).get("timestamp"))
        minimum_ready = closed_at + timedelta(
            minutes=5 * settings.same_thesis_reentry_min_bars
        )
        structure = (m5_analysis or {}).get("market_structure", {}) or {}
        expected = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(normalized_action, "")
        new_events = []
        for event in structure.get("structure_events", []) or []:
            if (
                isinstance(event, dict)
                and str(event.get("direction", "")).upper() == expected
                and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
            ):
                event_time = _parse_utc_datetime(event.get("time"))
                if event_time is not None and event_time > closed_at:
                    new_events.append(event)
        retest = structure.get("retest_continuation")
        retest_time = (
            _parse_utc_datetime(retest.get("time"))
            if isinstance(retest, dict)
            and str(retest.get("direction", "")).upper() == expected
            else None
        )
        has_new_retest = bool(retest_time is not None and retest_time > closed_at)
        range_setup = structure.get("range_reversion", {}) or {}
        range_time = (
            analysis_time
            if isinstance(range_setup, dict)
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper()
            == normalized_action
            else None
        )
        has_new_range = bool(range_time is not None and range_time > closed_at)

        if (
            analysis_time is None
            or analysis_time < minimum_ready
            or not (new_events or has_new_retest or has_new_range)
        ):
            bars = settings.same_thesis_reentry_min_bars
            outcome = "profit" if pnl > 0 else "loss" if pnl < 0 else "flat"
            return (
                False,
                "REJECTED [Same-Thesis Re-entry]: The latest trade in this "
                f"direction closed with a {outcome} at "
                f"{closed_at.strftime('%H:%M:%S')} UTC. Wait "
                f"for at least {bars} completed M5 bars and a new directional "
                "BOS/CHoCH, verified pullback resumption, or a newly qualified "
                "range reversal formed after that close.",
            )
        return True, ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        symbol: str,
        action: str,                         # "BUY" | "SELL" | "HOLD"
        llm_decision: Dict[str, Any],        # full JSON from LLM client
        account_info: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        market_snapshot,                     # snapshot-like object built by the engine
        calendar_events: Optional[List[Dict[str, Any]]],
        trade_history: List[Dict[str, Any]],
        m5_analysis: Optional[Dict[str, Any]] = None,
        m15_analysis: Optional[Dict[str, Any]] = None,
        h1_analysis: Optional[Dict[str, Any]] = None,
        h4_analysis: Optional[Dict[str, Any]] = None,
    ) -> RiskValidationResult:
        """
        Runs standard and dynamic risk rules in sequence.
        Returns RiskValidationResult containing approval status and dynamic scoring.
        """
        now_utc = datetime.now(timezone.utc)
        self._maybe_reset_daily(now_utc)

        # HOLD never needs further validation
        if action == "HOLD":
            return RiskValidationResult(approved=True, reason="HOLD — no execution needed.")

        # ── 0. Adaptive Risk Streak Logic ─────────────────────────────
        losing_streak = _effective_losing_streak(
            trade_history, now_utc, self._loss_streak_reset_after_utc
        )

        # A weekend is not a reason to lower quality requirements.
        is_crypto = any(c in symbol.upper() for c in ["ETH", "LTC", "XRP", "BTC"])
        is_weekend_crypto = is_crypto and now_utc.weekday() >= 5
        
        if is_weekend_crypto:
            min_confluence = 55.0
            min_quality = 55.0
        else:
            min_confluence = 55.0
            min_quality = 55.0

        confluence_streak_surcharge = 0.0
        # A zero pause is an explicit operator choice to disable every
        # losing-streak penalty. Previously the entry lock, quality surcharge
        # and sizing reduction remained active even while the UI reported a
        # zero-hour pause.
        streak_pause_hours = max(
            0, int(getattr(settings, "loss_streak_pause_hours", 24))
        )
        streak_controls_enabled = streak_pause_hours > 0
        if streak_controls_enabled:
            if losing_streak == 1:
                confluence_streak_surcharge = 10.0
                min_confluence += 10.0
                min_quality += 10.0
                logger.info(
                    "Adaptive Risk: Streak = 1 loss. "
                    "Quality/Confluence floor raised to %.1f%%.",
                    min_quality,
                )
            elif losing_streak == 2:
                confluence_streak_surcharge = 20.0
                min_confluence += 20.0
                min_quality += 20.0
                logger.info(
                    "Adaptive Risk: Streak = 2 losses. "
                    "Quality/Confluence floor raised to %.1f%%.",
                    min_quality,
                )
            elif losing_streak >= 3:
                logger.warning(
                    "Adaptive Risk: Losing streak is %s. "
                    "Enforcing a %s-hour cooling lock.",
                    losing_streak,
                    streak_pause_hours,
                )
                return RiskValidationResult(
                    approved=False,
                    reason=(
                        f"REJECTED [Losing Streak Pause]: {losing_streak} "
                        f"consecutive losses; pause lasts "
                        f"{streak_pause_hours} hours from the "
                        "latest close."
                    ),
                )

        # Compute Confluence & Quality
        confluence_score = 0.0
        quality_score = 0.0
        confluence_factors: Dict[str, bool] = {}
        strategy_mode = self._resolve_strategy_mode(
            action,
            llm_decision,
            m5_analysis,
            m15_analysis,
            h1_analysis,
            h4_analysis,
        )
        if strategy_mode and not str(
            (llm_decision.get("_strategy") or {}).get("mode", "")
        ).strip():
            llm_decision = dict(llm_decision)
            llm_decision["_strategy"] = {
                "mode": strategy_mode,
                "source": "DETERMINISTIC_RISK_CLASSIFIER",
            }
        if m5_analysis:
            from core.scoring import DecisionScoringEngine
            c_score, factors = DecisionScoringEngine.calculate_confluence_score(
                action=action,
                m5_analysis=m5_analysis,
                m15_analysis=m15_analysis,
                h1_analysis=h1_analysis,
                h4_analysis=h4_analysis,
                session_info=(
                    (llm_decision.get("_forex_context") or {}).get(
                        "active_sessions_utc", "UNKNOWN"
                    )
                ),
                strategy_mode=strategy_mode,
            )
            confluence_score = c_score
            confluence_factors = factors

            q_data = DecisionScoringEngine.calculate_trade_quality_score(
                action=action,
                llm_decision=llm_decision,
                m5_analysis=m5_analysis,
                m15_analysis=m15_analysis,
                h1_analysis=h1_analysis,
                h4_analysis=h4_analysis,
                calendar_events=calendar_events
            )
            quality_score = q_data["overall_score"]

        def reject(reason: str) -> RiskValidationResult:
            """Preserve computed audit scores on every downstream rejection."""
            return RiskValidationResult(
                approved=False,
                reason=reason,
                quality_score=quality_score,
                confluence_score=confluence_score,
            )

        # ── 1. Weekend filter ──────────────────────────────────────────
        ok, msg = self._check_weekend(symbol, now_utc)
        if not ok:
            return reject(msg)

        # ── 2. Session filter ──────────────────────────────────────────
        ok, msg = self._check_session(now_utc)
        if not ok:
            return reject(msg)

        # ── 3. News / economic calendar filter ────────────────────────
        ok, msg = self._check_news(symbol, calendar_events, now_utc)
        if not ok:
            return reject(msg)

        # ── 4. Maximum spread filter ───────────────────────────────────
        ok, msg = self._check_spread(symbol, market_snapshot, llm_decision)
        if not ok:
            return reject(msg)

        # ── 5. Maximum simultaneous trades ────────────────────────────
        ok, msg = self._check_max_positions(open_positions)
        if not ok:
            return reject(msg)

        # ── 6. Duplicate trade prevention ─────────────────────────────
        ok, msg = self._check_duplicate(symbol, action, open_positions)
        if not ok:
            return reject(msg)

        # ── 7. Maximum daily loss ──────────────────────────────────────
        ok, msg = self._check_daily_loss(account_info)
        if not ok:
            return reject(msg)

        # ── 8. Maximum drawdown ────────────────────────────────────────
        ok, msg = self._check_drawdown(account_info)
        if not ok:
            return reject(msg)

        # ── 9. Cooldown after losses ───────────────────────────────────
        ok, msg = self._check_loss_cooldown(now_utc, trade_history)
        if not ok:
            return reject(msg)

        # ── 10. Free margin validation ─────────────────────────────────
        ok, msg = self._check_free_margin(account_info)
        if not ok:
            return reject(msg)

        # ── 11. Margin level validation ────────────────────────────────
        ok, msg = self._check_margin_level(account_info)
        if not ok:
            return reject(msg)

        ok, msg = self._check_margin_usage(account_info)
        if not ok:
            return reject(msg)

        # ── 12. Minimum Risk-to-Reward ratio ──────────────────────────
        ok, msg = self._check_risk_reward(llm_decision, market_snapshot, symbol)
        if not ok:
            return reject(msg)

        # ── 12B. Trend Agreement (H1 vs H4) ───────────────────────────
        strong_countertrend_active = False
        if h1_analysis and h4_analysis:
            t_h1 = self._trend_direction(h1_analysis)
            t_h4 = self._trend_direction(h4_analysis)
            if self._requires_strong_countertrend_exception(
                strategy_mode, h1_analysis, h4_analysis
            ):
                countertrend_allowed, countertrend_detail = self._strong_countertrend_exception(
                    action,
                    llm_decision,
                    m5_analysis,
                    m15_analysis,
                    h1_analysis,
                )
                if not countertrend_allowed:
                    return reject(
                        (
                            f"REJECTED [Trend Disagreement]: H1 Trend ({t_h1}) "
                            f"disagrees with H4 Trend ({t_h4}); strong-countertrend "
                            f"exception not met ({countertrend_detail})."
                        )
                    )
                logger.warning(
                    "Allowing controlled counter-H4 %s on %s: %s.",
                    action,
                    symbol,
                    countertrend_detail,
                )
                strong_countertrend_active = True

        # ── 12C. ATR Minimum Volatility ───────────────────────────────
        if m5_analysis:
            atr = m5_analysis.get("indicators", {}).get("atr_14_pips", 0.0)
            is_crypto = any(c in symbol.upper() for c in ["ETH", "LTC", "XRP", "BTC"])
            if is_crypto:
                price = m5_analysis.get("indicators", {}).get("current_price", 1.0)
                raw_atr = atr / 100.0 if "XRP" in symbol.upper() else atr
                min_raw_atr = price * 0.0005  # 0.05% of price
                if raw_atr < min_raw_atr:
                    return reject(
                        f"REJECTED [Low Volatility]: Crypto ATR of {raw_atr:.5f} is below minimum 0.05% price threshold ({min_raw_atr:.5f})."
                    )
            else:
                if atr < 1.0:
                    return reject(
                        f"REJECTED [Low Volatility]: ATR of {atr:.2f} pips is below minimum 1.0 pip threshold."
                    )

        # ── 12D. Ranging Market Filter ────────────────────────────────
        if m5_analysis:
            indicators = m5_analysis.get("indicators", {})
            shock_ok, shock_reason = self._check_market_shock(m5_analysis)
            if not shock_ok:
                return reject(shock_reason)
            adx = indicators.get("adx_14", 0.0)
            is_crypto = any(c in symbol.upper() for c in ["ETH", "LTC", "XRP", "BTC"])
            min_adx = settings.entry_min_adx
            if adx < min_adx and strategy_mode != "RANGE_REVERSION":
                return reject(
                    f"REJECTED [ADX Filter]: Weak ranging market (ADX {adx:.1f} < {min_adx}). Trend strategy requires momentum."
                )

        ok, msg = self._check_entry_structure(
            action,
            m5_analysis,
            m15_analysis,
            h1_analysis,
            h4_analysis,
            confluence_factors,
            strategy_mode,
            market_snapshot,
        )
        if not ok:
            return reject(msg)

        # Keep the base floor permissive enough for verified BOS/retest/range
        # setups, but require more evidence when a raw breakout is the only
        # completed-candle trigger.
        m5_structure = (m5_analysis or {}).get("market_structure", {}) or {}
        expected_direction = {
            "BUY": "BULLISH",
            "SELL": "BEARISH",
        }.get(str(action).upper(), "")
        matching_events = [
            event
            for event in m5_structure.get("structure_events", []) or []
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected_direction
            and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
        ]
        matching_retest = m5_structure.get("retest_continuation")
        has_matching_retest = bool(
            isinstance(matching_retest, dict)
            and str(matching_retest.get("direction", "")).upper()
            == expected_direction
        )
        breakout_text = str(
            m5_structure.get("breakout_status", "")
        ).upper()
        breakout_only_setup = bool(
            expected_direction
            and expected_direction in breakout_text
            and "BREAKOUT" in breakout_text
            and not matching_events
            and not has_matching_retest
        )
        if breakout_only_setup:
            min_quality = max(
                min_quality, settings.breakout_min_quality_score
            )

        forex_context = llm_decision.get("_forex_context", {}) or {}
        context_bias = str(
            forex_context.get("bias", "NEUTRAL") or "NEUTRAL"
        ).upper()
        action_bias = {"BUY": "BULLISH", "SELL": "BEARISH"}.get(action)
        if (
            forex_context.get("reliable")
            and context_bias in {"BULLISH", "BEARISH"}
            and context_bias != action_bias
        ):
            return reject(
                "REJECTED [Currency Strength]: Broker-derived cross-pair "
                f"context is {context_bias}, which conflicts with {action}."
            )

        ok, msg = self._check_same_thesis_reentry(
            action, m5_analysis, trade_history
        )
        if not ok:
            return reject(msg)

        # ── 12E. Confluence & Trade Quality Score Checks ───────────────
        if strong_countertrend_active:
            min_confluence = (
                settings.countertrend_min_confluence + confluence_streak_surcharge
            )
            logger.warning(
                "Applying strong-countertrend confluence floor %.1f%% on %s; "
                "the normal quality floor and all execution protections remain.",
                min_confluence,
                symbol,
            )

        if confluence_score < min_confluence:
            return reject(
                f"REJECTED [Confluence Score]: Calculated confluence {confluence_score:.1f}% is below required {min_confluence:.1f}%."
            )

        if quality_score < min_quality:
            return reject(
                f"REJECTED [Quality Score]: Overall trade quality {quality_score:.1f} is below required {min_quality:.1f}."
            )

        # ── 13. Dynamic lot sizing & risk per trade ────────────────────
        lot, msg, sizing = self._compute_lot_size(
            llm_decision,
            account_info,
            market_snapshot,
            symbol,
            trade_history,
            losing_streak=losing_streak,
        )
        if lot is None:
            return reject(msg)
        if sizing["rr"] + 1e-9 < settings.min_risk_reward_ratio:
            return reject(
                (
                    f"REJECTED [Net R:R Ratio]: Execution-adjusted R:R of "
                    f"{sizing['rr']:.2f} is below {settings.min_risk_reward_ratio:.2f}."
                )
            )

        ok, msg = self._check_portfolio_risk(account_info, open_positions, sizing["risk_usd"])
        if not ok:
            return reject(msg)

        # ── 14. Final margin sanity on computed lot ────────────────────
        ok, msg = self._check_margin_for_lot(account_info, lot, symbol, action)
        if not ok:
            return reject(msg)

        logger.info(
            f"[RISK APPROVED] {symbol} {action} | Lot={lot:.2f} | "
            f"Quality Score={quality_score:.1f} | Confluence={confluence_score:.1f}%"
        )
        return RiskValidationResult(
            approved=True,
            reason="All risk checks passed.",
            adjusted_lot=lot,
            quality_score=quality_score,
            confluence_score=confluence_score,
            estimated_risk_usd=sizing["risk_usd"],
            estimated_reward_usd=sizing["reward_usd"],
            risk_percent_balance=sizing["risk_pct"],
            planned_rr=sizing["rr"],
            risk_budget_usd=sizing["risk_budget_usd"],
        )

    def validate_manual_override(
        self,
        *,
        symbol: str,
        action: str,
        decision: Dict[str, Any],
        account_info: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        market_snapshot: Any,
        trade_history: List[Dict[str, Any]],
    ) -> RiskValidationResult:
        """Size an explicitly confirmed operator override at broker minimum.

        This path deliberately bypasses project strategy and risk-policy gates.
        The engine and executor still enforce current decision/account identity,
        valid broker volume and SL/TP, quote freshness, available broker margin,
        and MT5 ``order_check`` immediately before submission. Automatic entry
        validation is unchanged.
        """
        action = str(action).upper()

        def reject(reason: str) -> RiskValidationResult:
            return RiskValidationResult(approved=False, reason=reason)

        if action not in {"BUY", "SELL"}:
            return reject("Manual override requires a BUY or SELL candidate")

        lot, reason, sizing = self._compute_lot_size(
            decision,
            account_info,
            market_snapshot,
            symbol,
            trade_history,
            losing_streak=0,
            apply_streak_scaling=False,
            force_minimum_lot=True,
            enforce_profit_objective=False,
        )
        if lot is None:
            return reject(reason)

        return RiskValidationResult(
            approved=True,
            reason="Operator override sized at the broker minimum volume.",
            adjusted_lot=lot,
            estimated_risk_usd=sizing["risk_usd"],
            estimated_reward_usd=sizing["reward_usd"],
            risk_percent_balance=sizing["risk_pct"],
            planned_rr=sizing["rr"],
            risk_budget_usd=sizing["risk_budget_usd"],
        )

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------

    def _reset_daily_state(self) -> None:
        self._daily_loss_usd: float = 0.0
        self._daily_reset_date: Optional[str] = None
        self._daily_loss_reset_after_utc: Optional[datetime] = None
        self._loss_cooldown_reset_after_utc: Optional[datetime] = None
        self._loss_streak_reset_after_utc: Optional[datetime] = None
        self._peak_balance: float = 0.0          # set on first call
        self._last_loss_time: Optional[datetime] = None

    def _maybe_reset_daily(self, now_utc: datetime) -> None:
        today = now_utc.strftime("%Y-%m-%d")
        if self._daily_reset_date != today:
            self._daily_loss_usd = 0.0
            self._daily_reset_date = today
            self._daily_loss_reset_after_utc = None
            self._loss_cooldown_reset_after_utc = None
            logger.info(f"Daily risk counters reset for {today}.")

    def restore_daily_loss_reset(self, reset_at: Any) -> bool:
        """Restore an operator reset marker when it belongs to the current UTC day."""
        now = datetime.now(timezone.utc)
        self._maybe_reset_daily(now)
        parsed = _parse_utc_datetime(reset_at)
        if parsed is None or parsed.date() != now.date():
            return False
        self._daily_loss_reset_after_utc = parsed
        self._loss_cooldown_reset_after_utc = parsed
        self._daily_loss_usd = 0.0
        self._last_loss_time = None
        return True

    def restore_loss_cooldown_reset(self, reset_at: Any) -> bool:
        """Restore today's operator cooldown-only reset marker."""
        now = datetime.now(timezone.utc)
        self._maybe_reset_daily(now)
        parsed = _parse_utc_datetime(reset_at)
        if parsed is None or parsed.date() != now.date():
            return False
        current = self._loss_cooldown_reset_after_utc
        effective = max(parsed, current) if current is not None else parsed
        self._loss_cooldown_reset_after_utc = effective
        if self._last_loss_time is None or self._last_loss_time <= effective:
            self._last_loss_time = None
        return True

    def reset_loss_cooldown(
        self, reset_at: Optional[datetime] = None
    ) -> datetime:
        """Clear the current cooldown without altering UTC daily-loss totals."""
        marker = reset_at or datetime.now(timezone.utc)
        if marker.tzinfo is None:
            marker = marker.replace(tzinfo=timezone.utc)
        marker = marker.astimezone(timezone.utc)
        self._maybe_reset_daily(marker)
        self._loss_cooldown_reset_after_utc = marker
        self._last_loss_time = None
        logger.warning(
            "Operator reset the global loss cooldown at %s; daily loss was unchanged.",
            marker.isoformat(),
        )
        return marker

    def restore_losing_streak_reset(self, reset_at: Any) -> bool:
        """Restore an account-scoped operator streak reset marker."""
        parsed = _parse_utc_datetime(reset_at)
        if parsed is None or parsed > datetime.now(timezone.utc) + timedelta(minutes=1):
            return False
        self._loss_streak_reset_after_utc = parsed
        return True

    def reset_losing_streak(
        self, reset_at: Optional[datetime] = None
    ) -> datetime:
        """Start loss-streak accounting again without altering trade history."""
        marker = reset_at or datetime.now(timezone.utc)
        if marker.tzinfo is None:
            marker = marker.replace(tzinfo=timezone.utc)
        marker = marker.astimezone(timezone.utc)
        self._loss_streak_reset_after_utc = marker
        logger.warning(
            "Operator reset the account loss-streak window at %s; broker history "
            "and daily-loss accounting were unchanged.",
            marker.isoformat(),
        )
        return marker

    def reset_daily_loss_for_today(
        self, reset_at: Optional[datetime] = None
    ) -> datetime:
        """Start today's strategy gross-loss accounting again from zero."""
        marker = reset_at or datetime.now(timezone.utc)
        if marker.tzinfo is None:
            marker = marker.replace(tzinfo=timezone.utc)
        marker = marker.astimezone(timezone.utc)
        self._maybe_reset_daily(marker)
        self._daily_loss_reset_after_utc = marker
        self._loss_cooldown_reset_after_utc = marker
        self._daily_loss_usd = 0.0
        self._last_loss_time = None
        logger.warning(
            "Operator reset UTC daily-loss accounting and global loss cooldown at %s.",
            marker.isoformat(),
        )
        return marker

    def record_trade_closed(
        self,
        profit_usd: float,
        balance: float,
        *,
        strategy_owned: bool = True,
    ) -> None:
        """
        Track a strategy-owned closure for daily loss and cooldown state.

        External/manual MT5 positions are intentionally excluded so their
        outcomes cannot pause autonomous entries or alter the strategy's loss
        streak. Account equity and free-margin gates still include them while
        they are open.
        """
        if not strategy_owned:
            return
        if profit_usd < 0:
            self._daily_loss_usd += abs(profit_usd)
            self._last_loss_time = datetime.now(timezone.utc)
            logger.info(
                f"Loss recorded: ${profit_usd:.2f} | "
                f"Daily loss total: ${self._daily_loss_usd:.2f}"
            )
        # Track peak balance for drawdown calculation
        if balance > self._peak_balance:
            self._peak_balance = balance

    def synchronize_closed_trades(self, trades: List[Dict[str, Any]], balance: float) -> None:
        """Rebuild loss/cooldown state from broker-confirmed closed positions."""
        now = datetime.now(timezone.utc)
        self._maybe_reset_daily(now)
        today = now.date()
        todays = []
        for trade in trades:
            closed_at = _parse_utc_datetime(trade.get("close_time") or trade.get("time"))
            if closed_at is None:
                continue
            try:
                pnl = float(trade.get("net_profit", trade.get("profit", 0.0)) or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pnl):
                continue
            if closed_at.date() == today:
                todays.append((closed_at, pnl))

        broker_daily_loss = sum(
            abs(pnl)
            for closed_at, pnl in todays
            if pnl < 0
            and (
                self._daily_loss_reset_after_utc is None
                or closed_at > self._daily_loss_reset_after_utc
            )
        )
        # A range-based broker sync can lag a just-observed closure. Never let
        # an incomplete snapshot erase losses already recorded by the position
        # monitor during the same UTC day.
        self._daily_loss_usd = max(self._daily_loss_usd, broker_daily_loss)
        cooldown_losses = [
            closed_at
            for closed_at, pnl in todays
            if pnl < 0
            and (
                self._loss_cooldown_reset_after_utc is None
                or closed_at > self._loss_cooldown_reset_after_utc
            )
        ]
        broker_last_loss = max(cooldown_losses) if cooldown_losses else None
        if broker_last_loss is not None and (
            self._last_loss_time is None or broker_last_loss > self._last_loss_time
        ):
            self._last_loss_time = broker_last_loss
        self.update_peak_balance(balance)

    def update_peak_balance(self, balance: float) -> None:
        """Call at startup with the current account balance."""
        if balance > self._peak_balance:
            self._peak_balance = balance

    def reset_for_account(self) -> None:
        """Clear risk counters before loading a different MT5 account."""
        self._reset_daily_state()

    @property
    def peak_balance(self) -> float:
        return self._peak_balance

    @property
    def daily_loss_usd(self) -> float:
        return self._daily_loss_usd

    def daily_loss_status(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Return the current UTC-day loss gate and a dashboard-safe summary."""
        self._maybe_reset_daily(datetime.now(timezone.utc))
        balance = float(account_info.get("balance", 0.0) or 0.0)
        percent_limit = balance * settings.max_daily_loss_pct / 100.0
        limits = [
            limit
            for limit in (settings.max_daily_loss_usd, percent_limit)
            if limit > 0
        ]
        effective_limit = min(limits) if limits else 0.0
        if effective_limit and self._daily_loss_usd >= effective_limit:
            return (
                False,
                f"Gross losses ${self._daily_loss_usd:.2f} / ${effective_limit:.2f}; "
                "new entries resume after the UTC daily reset",
            )
        if effective_limit:
            return True, f"${self._daily_loss_usd:.2f} / ${effective_limit:.2f}"
        return True, "No daily loss limit configured"

    # ------------------------------------------------------------------
    # Individual rule implementations
    # ------------------------------------------------------------------

    def _check_weekend(self, symbol: str, now_utc: datetime) -> Tuple[bool, str]:
        """Rule 1 — Weekend filter."""
        day = now_utc.weekday()
        forex_closed = (
            (day == 4 and now_utc.hour >= 22)
            or day == 5
            or (day == 6 and now_utc.hour < 22)
        )
        if not forex_closed:
            return True, ""
        if settings.weekend_trading_enabled and symbol.upper() in [
            s.upper() for s in settings.weekend_symbols
        ]:
            return True, ""
        return (
            False,
            f"REJECTED [Weekend Filter]: {symbol} trading is disabled on weekends. "
            f"Weekday={now_utc.weekday()}."
        )

    def _check_session(self, now_utc: datetime) -> Tuple[bool, str]:
        """Rule 2 — Trading session filter (UTC)."""
        if not settings.session_filter_enabled:
            return True, ""

        current_time_str = now_utc.strftime("%H:%M")
        for session_range in settings.allowed_sessions_utc:
            try:
                start_str, end_str = session_range.split("-")
                start = datetime.strptime(start_str.strip(), "%H:%M").time()
                end = datetime.strptime(end_str.strip(), "%H:%M").time()
                current_t = now_utc.time().replace(second=0, microsecond=0)
                if start <= current_t <= end:
                    return True, ""
            except ValueError:
                continue

        return (
            False,
            f"REJECTED [Session Filter]: Current UTC time {current_time_str} is outside "
            f"allowed trading sessions: {settings.allowed_sessions_utc}."
        )

    def _check_news(
        self,
        symbol: str,
        calendar_events: Optional[List[Dict[str, Any]]],
        now_utc: datetime,
    ) -> Tuple[bool, str]:
        """Rule 3 — High-impact news lockout window."""
        if calendar_events is None:
            if settings.require_news_calendar and not is_crypto_symbol(symbol):
                return False, "REJECTED [News Filter]: Economic calendar is unavailable."
            return True, ""
        if not calendar_events:
            return True, ""

        lockout_minutes = settings.news_lockout_minutes
        if lockout_minutes <= 0:
            return True, ""

        base_curr = symbol[:3].upper()
        quote_curr = symbol[3:].upper()

        for event in calendar_events:
            if event.get("impact", "").upper() != "HIGH":
                continue
            event_curr = event.get("currency", "").upper()
            if event_curr not in (base_curr, quote_curr):
                continue

            event_time_str = event.get("time", "")
            try:
                event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                event_time = event_time.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            delta_minutes = abs((event_time - now_utc).total_seconds()) / 60
            if delta_minutes <= lockout_minutes:
                return (
                    False,
                    f"REJECTED [News Filter]: HIGH impact {event_curr} event "
                    f"'{event.get('event_name')}' within {lockout_minutes}-minute lockout window "
                    f"({delta_minutes:.1f} min away)."
                )
        return True, ""

    def _check_spread(
        self,
        symbol: str,
        market_snapshot,
        llm_decision: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Rule 4: asset-aware and stop-relative transaction-cost filter."""
        if market_snapshot is None:
            return False, "REJECTED [Spread Filter]: Live spread is unavailable."
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return False, "REJECTED [Spread Filter]: Broker quote metadata is unavailable."
        action = str((llm_decision or {}).get("action", "")).upper()
        entry = tick.ask if action == "BUY" else tick.bid if action == "SELL" else None
        stop_loss = (llm_decision or {}).get("stop_loss")
        ok, reason, _ = validate_spread(
            symbol, info, tick, entry=entry, stop_loss=stop_loss
        )
        return (True, "") if ok else (False, f"REJECTED [Spread Filter]: {reason}.")

    def _check_max_positions(self, open_positions: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Rule 5 — Maximum simultaneous trades."""
        count = len(open_positions)
        if count >= settings.max_open_positions:
            return (
                False,
                f"REJECTED [Max Positions]: Already {count} open trade(s). "
                f"Maximum allowed: {settings.max_open_positions}."
            )
        return True, ""

    def _check_duplicate(
        self, symbol: str, action: str, open_positions: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Rule 6 — Duplicate trade prevention (same symbol + same direction)."""
        pos_type_code = 0 if action == "BUY" else 1  # MT5: 0=BUY, 1=SELL
        for pos in open_positions:
            if (
                pos.get("symbol", "").upper() == symbol.upper()
                and pos.get("type") == pos_type_code
            ):
                return (
                    False,
                    f"REJECTED [Duplicate Trade]: An open {action} position on "
                    f"{symbol} (ticket {pos.get('ticket')}) already exists."
                )
        return True, ""

    def _check_daily_loss(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Rule 7 — Maximum daily loss in USD."""
        balance = float(account_info.get("balance", 0.0) or 0.0)
        percent_limit = balance * settings.max_daily_loss_pct / 100.0
        limits = [limit for limit in (settings.max_daily_loss_usd, percent_limit) if limit > 0]
        effective_limit = min(limits) if limits else 0.0
        if effective_limit and self._daily_loss_usd >= effective_limit:
            return (
                False,
                f"REJECTED [Daily Loss Limit]: Daily loss of ${self._daily_loss_usd:.2f} "
                f"has reached the effective maximum ${effective_limit:.2f}. "
                f"No new trades until tomorrow (UTC)."
            )
        return True, ""

    def _check_drawdown(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Rule 8 — Maximum drawdown from peak balance."""
        if not settings.drawdown_entry_lock_enabled or settings.max_drawdown_pct <= 0:
            return True, ""
        if self._peak_balance <= 0:
            # Not yet initialised; allow trade
            return True, ""
        equity = account_info.get("equity", 0.0)
        drawdown_pct = ((self._peak_balance - equity) / self._peak_balance) * 100.0
        if drawdown_pct >= settings.max_drawdown_pct:
            return (
                False,
                f"REJECTED [Drawdown Limit]: Current drawdown {drawdown_pct:.2f}% from peak "
                f"${self._peak_balance:.2f} exceeds max allowed {settings.max_drawdown_pct}%."
            )
        return True, ""

    def _check_loss_cooldown(
        self,
        now_utc: datetime,
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
        """Rule 9 — short, symbol-local cooldown after a losing trade.

        ``validate`` supplies broker-reconciled history for the symbol being
        evaluated.  The legacy in-memory timestamp remains a fallback for
        direct callers and for the brief interval before reconciliation.
        """
        if settings.loss_cooldown_minutes <= 0:
            return True, ""

        last_loss_time: Optional[datetime] = None
        if trade_history is not None:
            losses = []
            for trade in trade_history:
                closed_at = _parse_utc_datetime(
                    trade.get("close_time") or trade.get("time")
                )
                try:
                    pnl = float(
                        trade.get("net_profit", trade.get("profit", 0.0))
                        or 0.0
                    )
                except (TypeError, ValueError):
                    continue
                if (
                    closed_at is not None
                    and math.isfinite(pnl)
                    and pnl < 0
                    and (
                        self._loss_cooldown_reset_after_utc is None
                        or closed_at > self._loss_cooldown_reset_after_utc
                    )
                ):
                    losses.append(closed_at)
            last_loss_time = max(losses) if losses else None
        else:
            last_loss_time = self._last_loss_time

        if last_loss_time is None:
            return True, ""
        elapsed = (now_utc - last_loss_time).total_seconds() / 60.0
        if elapsed < settings.loss_cooldown_minutes:
            remaining = settings.loss_cooldown_minutes - elapsed
            return (
                False,
                f"REJECTED [Symbol Loss Cooldown]: This market's last loss was "
                f"{elapsed:.1f} min ago. "
                f"Cooldown period is {settings.loss_cooldown_minutes} min. "
                f"Wait {remaining:.1f} more minutes."
            )
        return True, ""

    def _check_free_margin(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Rule 10 — Minimum free margin in USD."""
        free_margin = account_info.get("margin_free", 0.0)
        if free_margin < settings.min_free_margin_usd:
            return (
                False,
                f"REJECTED [Free Margin]: Free margin ${free_margin:.2f} is below "
                f"minimum required ${settings.min_free_margin_usd:.2f}."
            )
        return True, ""

    def _check_margin_level(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Rule 11 — Minimum margin level percentage."""
        margin_level = account_info.get("margin_level", 0.0)
        # margin_level of 0 typically means no open positions (infinite margin)
        if margin_level == 0.0:
            return True, ""
        if margin_level < settings.min_margin_level_pct:
            return (
                False,
                f"REJECTED [Margin Level]: Margin level {margin_level:.1f}% is below "
                f"minimum required {settings.min_margin_level_pct:.1f}%."
            )
        return True, ""

    def _check_margin_usage(self, account_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Reject new exposure when existing margin use is already too high."""
        equity = float(account_info.get("equity", 0.0) or 0.0)
        margin = float(account_info.get("margin", 0.0) or 0.0)
        if equity <= 0:
            return False, "REJECTED [Margin Usage]: Account equity is unavailable."
        usage = margin / equity * 100.0
        if usage >= settings.max_margin_usage_pct:
            return (
                False,
                f"REJECTED [Margin Usage]: {usage:.1f}% is already in use; "
                f"limit is {settings.max_margin_usage_pct:.1f}%.",
            )
        return True, ""

    def _check_risk_reward(
        self,
        llm_decision: Dict[str, Any],
        market_snapshot,
        symbol: str,
    ) -> Tuple[bool, str]:
        """Rule 12 — Minimum Risk-to-Reward ratio from absolute SL/TP prices."""
        stop_loss = llm_decision.get("stop_loss")
        take_profit = llm_decision.get("take_profit")
        action = str(llm_decision.get("action", "")).upper()
        tick = mt5.symbol_info_tick(symbol)

        if stop_loss is None or take_profit is None or tick is None:
            return False, "REJECTED [R:R Ratio]: Live quote, SL and TP are required."

        try:
            entry = float(tick.ask if action == "BUY" else tick.bid)
            sl = float(stop_loss)
            tp = float(take_profit)
        except (TypeError, ValueError):
            return False, "REJECTED [R:R Ratio]: Could not parse entry/SL/TP from LLM decision."

        if action == "BUY" and not (sl < entry < tp):
            return False, "REJECTED [Price Levels]: BUY requires SL < live ask < TP."
        if action == "SELL" and not (tp < entry < sl):
            return False, "REJECTED [Price Levels]: SELL requires TP < live bid < SL."

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False, "REJECTED [R:R Ratio]: Stop loss equals entry price — zero risk distance."

        rr_ratio = reward / risk
        # Price arithmetic can produce values such as 1.469999999999 even
        # when the submitted levels represent exactly 1.47 R. Keep this raw
        # gate consistent with the execution-adjusted gate above.
        if rr_ratio + 1e-9 < settings.min_risk_reward_ratio:
            return (
                False,
                f"REJECTED [R:R Ratio]: Calculated R:R of {rr_ratio:.2f} is below the "
                f"minimum required {settings.min_risk_reward_ratio:.2f}."
            )
        return True, ""

    def _compute_lot_size(
        self,
        llm_decision: Dict[str, Any],
        account_info: Dict[str, Any],
        market_snapshot,
        symbol: str,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        *,
        losing_streak: Optional[int] = None,
        apply_streak_scaling: bool = True,
        risk_pct_override: Optional[float] = None,
        force_minimum_lot: bool = False,
        enforce_profit_objective: bool = True,
    ) -> Tuple[Optional[float], str, Dict[str, float]]:
        """
        Size from MT5's account-currency loss calculation and round down.

        If the broker's minimum volume is already above the risk budget, the
        setup is rejected. It is never rounded up into an oversized trade.
        """
        empty = {
            "risk_usd": 0.0,
            "stop_risk_usd": 0.0,
            "slippage_reserve_usd": 0.0,
            "execution_cost_usd": 0.0,
            "reward_usd": 0.0,
            "risk_pct": 0.0,
            "rr": 0.0,
            "risk_budget_usd": 0.0,
        }
        balance = float(account_info.get("balance", 0.0) or 0.0)
        if balance <= 0:
            return None, "REJECTED [Lot Sizing]: Account balance is zero.", empty

        # Validation passes its already-computed value so the quality and
        # sizing gates cannot disagree at the exact expiry boundary.  Direct
        # callers still get the same history-derived behavior.
        if losing_streak is None:
            losing_streak = _effective_losing_streak(
                trade_history,
                datetime.now(timezone.utc),
                self._loss_streak_reset_after_utc,
            )

        # The LLM decides direction only. Position risk is owned by the user's
        # deterministic configuration so a schema example cannot silently
        # reduce (or increase) the live budget.
        risk_pct = (
            float(risk_pct_override)
            if risk_pct_override is not None
            else settings.risk_percent
        )
        if not math.isfinite(risk_pct) or risk_pct <= 0:
            return None, "REJECTED [Lot Sizing]: Risk percentage is invalid.", empty
        
        # Automatic entries scale down after losses. A manually confirmed
        # rejected trade instead uses the configured per-trade ceiling, so a
        # broker-minimum lot that still fits that ceiling remains actionable.
        streak_scaling_enabled = (
            apply_streak_scaling
            and int(getattr(settings, "loss_streak_pause_hours", 24)) > 0
        )
        if streak_scaling_enabled and losing_streak == 1:
            risk_pct = max(0.1, risk_pct * 0.5)
            logger.info(f"Adaptive Risk: Scaling risk % to {risk_pct:.2f}% due to 1 loss.")
        elif streak_scaling_enabled and losing_streak >= 2:
            risk_pct = 0.1
            logger.info("Adaptive Risk: Streak >= 2. Clamping risk to minimum 0.1% for capital preservation.")

        risk_budget = balance * risk_pct / 100.0
        if settings.auto_close_loss_enabled and settings.auto_close_loss_usd > 0:
            risk_budget = min(risk_budget, settings.auto_close_loss_usd)
        if not math.isfinite(risk_budget) or risk_budget <= 0:
            return None, "REJECTED [Lot Sizing]: Risk budget is invalid.", empty

        action = str(llm_decision.get("action", "")).upper()
        sl = llm_decision.get("stop_loss")
        tp = llm_decision.get("take_profit")
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if action not in {"BUY", "SELL"} or sl is None or tp is None or info is None or tick is None:
            return None, "REJECTED [Lot Sizing]: Broker quote and valid SL/TP are required.", empty

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        entry = float(tick.ask if action == "BUY" else tick.bid)
        sl_f, tp_f = float(sl), float(tp)
        min_lot = float(info.volume_min)
        step = float(info.volume_step or min_lot)
        # Volume units differ by instrument. Risk and margin, rather than a
        # cross-asset numeric volume ceiling, cap the position.
        max_lot = float(info.volume_max)
        if (
            not all(math.isfinite(value) for value in (min_lot, step, max_lot))
            or min_lot <= 0
            or step <= 0
            or max_lot < min_lot
        ):
            return None, "REJECTED [Lot Sizing]: Broker volume constraints are invalid.", empty

        min_estimate = estimate_execution_risk(
            symbol, action, min_lot, entry, sl_f, info
        )
        if min_estimate is None:
            return None, "REJECTED [Lot Sizing]: MT5 could not calculate execution-adjusted stop risk.", empty
        if not force_minimum_lot and min_estimate.total_risk_usd > risk_budget + 1e-9:
            actual_pct = min_estimate.total_risk_usd / balance * 100.0
            return (
                None,
                f"REJECTED [Minimum Lot Risk]: Broker minimum {min_lot:g} lot has "
                f"${min_estimate.total_risk_usd:.2f} execution-adjusted risk "
                f"(${min_estimate.stop_risk_usd:.2f} stop + "
                f"${min_estimate.slippage_reserve_usd:.2f} slippage + "
                f"${min_estimate.configured_cost_usd:.2f} configured costs; "
                f"{actual_pct:.2f}% of balance), above the "
                f"${risk_budget:.2f} budget. This account is too small for this setup.",
                empty,
            )

        if force_minimum_lot:
            lot = min_lot
            final_estimate = min_estimate
            # The executor re-estimates at the final quote. Keep a small,
            # explicit buffer for a tick changing between preview and send.
            risk_budget = max(
                risk_budget,
                min_estimate.total_risk_usd
                + max(0.01, min_estimate.total_risk_usd * 0.05),
            )
        else:
            # Find the largest broker step whose worst allowed fill, stop loss and
            # configured round-turn costs all fit inside the budget. Binary search
            # avoids assuming that every CFD's fee model is linear in its volume.
            min_units = max(1, int(math.ceil((min_lot - 1e-12) / step)))
            max_units = int(math.floor((max_lot + 1e-12) / step))
            low, high = min_units, max_units
            lot = None
            final_estimate = None
            estimate_failed = False
            while low <= high:
                units = (low + high) // 2
                candidate = round(units * step, 8)
                estimate = estimate_execution_risk(
                    symbol, action, candidate, entry, sl_f, info
                )
                if estimate is None:
                    estimate_failed = True
                    break
                if estimate.total_risk_usd <= risk_budget + 1e-9:
                    lot = candidate
                    final_estimate = estimate
                    low = units + 1
                else:
                    high = units - 1
            if estimate_failed:
                return None, "REJECTED [Lot Sizing]: MT5 could not verify execution-adjusted risk.", empty
            if lot is None or final_estimate is None or lot < min_lot:
                return None, "REJECTED [Lot Sizing]: No broker-valid lot fits the risk budget.", empty

        reward_result = mt5.order_calc_profit(
            order_type, symbol, lot, final_estimate.worst_entry, tp_f
        )
        if reward_result is None or not math.isfinite(float(reward_result)):
            return None, "REJECTED [Lot Sizing]: MT5 could not verify target reward.", empty
        risk_usd = final_estimate.total_risk_usd
        reward_usd = max(0.0, float(reward_result) - final_estimate.configured_cost_usd)
        rr = reward_usd / risk_usd if risk_usd else 0.0
        actual_pct = risk_usd / balance * 100.0

        if (
            enforce_profit_objective
            and settings.auto_close_profit_enabled
            and reward_usd + 1e-9 < settings.auto_close_profit_usd
        ):
            return (
                None,
                f"REJECTED [Profit Objective]: Technical TP offers ${reward_usd:.2f}, below "
                f"the requested ${settings.auto_close_profit_usd:.2f}. The target will not be stretched.",
                empty,
            )

        sizing = {
            "risk_usd": round(risk_usd, 4),
            "stop_risk_usd": round(final_estimate.stop_risk_usd, 4),
            "slippage_reserve_usd": round(final_estimate.slippage_reserve_usd, 4),
            "execution_cost_usd": round(final_estimate.configured_cost_usd, 4),
            "reward_usd": round(reward_usd, 4),
            "risk_pct": round(actual_pct, 4),
            "rr": round(rr, 4),
            "risk_budget_usd": round(risk_budget, 4),
        }
        logger.info(
            "Broker sizing %s: lot=%s total_risk=$%.2f (stop=$%.2f, "
            "slippage=$%.2f, costs=$%.2f; %.2f%%) net_reward=$%.2f RR=%.2f",
            symbol, lot, risk_usd, final_estimate.stop_risk_usd,
            final_estimate.slippage_reserve_usd,
            final_estimate.configured_cost_usd, actual_pct, reward_usd, rr,
        )
        return lot, "", sizing

    def _check_portfolio_risk(
        self,
        account_info: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        proposed_risk_usd: float,
    ) -> Tuple[bool, str]:
        balance = float(account_info.get("balance", 0.0) or 0.0)
        if balance <= 0:
            return False, "REJECTED [Portfolio Risk]: Balance is unavailable."
        total_risk = proposed_risk_usd
        for position in open_positions:
            sl = float(position.get("sl", 0.0) or 0.0)
            if sl <= 0:
                return False, "REJECTED [Portfolio Risk]: An open position has no stop loss."
            order_type = mt5.ORDER_TYPE_BUY if int(position.get("type", 0)) == 0 else mt5.ORDER_TYPE_SELL
            risk = mt5.order_calc_profit(
                order_type,
                str(position.get("symbol", "")),
                float(position.get("volume", 0.0)),
                float(position.get("price_open", 0.0)),
                sl,
            )
            try:
                open_stop_risk = downside_risk_usd(risk)
            except ValueError:
                return False, (
                    "REJECTED [Portfolio Risk]: MT5 could not calculate an open "
                    "position's stop-loss risk."
                )
            total_risk += open_stop_risk
        pct = total_risk / balance * 100.0
        if pct > settings.max_portfolio_risk_pct + 1e-9:
            return (
                False,
                f"REJECTED [Portfolio Risk]: Stops would risk ${total_risk:.2f} "
                f"({pct:.2f}% of balance), above {settings.max_portfolio_risk_pct:.2f}%.",
            )
        return True, ""

    def _check_margin_for_lot(
        self, account_info: Dict[str, Any], lot: float, symbol: str = "", action: str = "BUY"
    ) -> Tuple[bool, str]:
        """Rule 14 — Sanity check that free margin can cover the computed lot."""
        free_margin = account_info.get("margin_free", 0.0)
        
        # 1. Try to compute using MT5 API (the most accurate way)
        symbol_info = mt5.symbol_info(symbol) if symbol else None
        if symbol_info:
            order_type = (
                mt5.ORDER_TYPE_SELL
                if str(action).upper() == "SELL"
                else mt5.ORDER_TYPE_BUY
            )
            current_price = (
                symbol_info.bid if order_type == mt5.ORDER_TYPE_SELL else symbol_info.ask
            )
            margin_calc = mt5.order_calc_margin(order_type, symbol, lot, current_price)
            if margin_calc is not None and margin_calc > 0:
                estimated_margin_needed = margin_calc
                required_with_buffer = estimated_margin_needed * 1.05
                if free_margin < required_with_buffer:
                    return (
                        False,
                        f"REJECTED [Margin for Lot]: MT5 margin needed (with buffer) "
                        f"${required_with_buffer:.2f} for {lot:.2f} lot of {symbol} exceeds "
                        f"free margin ${free_margin:.2f}."
                    )
                equity = float(account_info.get("equity", 0.0) or 0.0)
                current_margin = float(account_info.get("margin", 0.0) or 0.0)
                projected_usage = (
                    (current_margin + estimated_margin_needed) / equity * 100.0
                    if equity > 0 else 100.0
                )
                if projected_usage > settings.max_margin_usage_pct:
                    return (
                        False,
                        f"REJECTED [Projected Margin]: New trade would use "
                        f"{projected_usage:.1f}% of equity; limit is "
                        f"{settings.max_margin_usage_pct:.1f}%.",
                    )
                return True, ""

        return False, "REJECTED [Margin for Lot]: MT5 could not calculate broker margin."

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (used by older engine code)
    # ------------------------------------------------------------------

    @staticmethod
    def check_position_limits(active_positions: List[dict]) -> bool:
        return len(active_positions) < settings.max_open_positions

    @staticmethod
    def get_validated_lot_size(proposed_lot: float) -> float:
        if proposed_lot <= 0 or proposed_lot > settings.default_lot_size:
            return settings.default_lot_size
        return proposed_lot

    @staticmethod
    def has_sufficient_margin(account_info: dict, min_required: float = 4.0) -> bool:
        return account_info.get("margin_free", 0.0) >= min_required

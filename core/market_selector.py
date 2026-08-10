"""Broker-aware ranking for a small, explicitly bounded market universe."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from app_config.settings import settings


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


class AdaptiveMarketSelector:
    """Rank broker-open markets by evidence, independently of position sizing."""

    @staticmethod
    def summarize_performance(
        history: Iterable[Dict[str, Any]],
        *,
        strategy_magic: int,
    ) -> Dict[str, Any]:
        """Return a bounded, broker-confirmed expectancy summary."""
        manual_close_reasons = {"CLIENT", "MOBILE", "WEB"}
        rows = [
            row
            for row in history
            if int(row.get("magic", 0) or 0) == int(strategy_magic)
            and str(row.get("close_reason", "") or "").upper()
            not in manual_close_reasons
        ][: settings.market_performance_lookback]
        count = len(rows)
        wins = [
            _finite(row.get("net_profit"))
            for row in rows
            if _finite(row.get("net_profit")) > 0
        ]
        losses = [
            abs(_finite(row.get("net_profit")))
            for row in rows
            if _finite(row.get("net_profit")) < 0
        ]
        rr_values = [
            _finite(row.get("rr_achieved"))
            for row in rows
            if _finite(row.get("initial_risk_usd")) > 0
            and math.isfinite(_finite(row.get("rr_achieved"), math.nan))
        ]
        expectancy_usd = sum(
            _finite(row.get("net_profit")) for row in rows
        ) / count if count else 0.0
        expectancy_r = (
            sum(rr_values) / len(rr_values) if rr_values else 0.0
        )
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (3.0 if gross_profit > 0 else 0.0)
        )
        primary_count = len(rr_values) if rr_values else count
        evidence_weight = (
            primary_count / (primary_count + 8.0)
            if primary_count
            else 0.0
        )
        if rr_values:
            primary_return = expectancy_r
        else:
            average_absolute_outcome = (
                sum(abs(_finite(row.get("net_profit"))) for row in rows)
                / count
                if count
                else 0.0
            )
            primary_return = (
                expectancy_usd / max(average_absolute_outcome, 0.01)
                if count
                else 0.0
            )
        # Use one shrinkage-adjusted return metric. Profit factor remains a
        # diagnostic instead of double-counting the same P/L observations.
        raw_score = primary_return * 20.0
        performance_score = max(-20.0, min(20.0, raw_score)) * evidence_weight
        blocked = (
            count >= settings.market_performance_min_trades
            and (
                (
                    len(rr_values) >= settings.market_performance_min_trades
                    and expectancy_r
                    <= settings.market_performance_block_expectancy_r
                )
                or (
                    len(rr_values) < settings.market_performance_min_trades
                    and expectancy_usd < 0
                )
            )
            and profit_factor < settings.market_performance_block_profit_factor
        )
        return {
            "trade_count": count,
            "win_rate_pct": round(len(wins) / count * 100.0, 1) if count else 0.0,
            "expectancy_usd": round(expectancy_usd, 4),
            "expectancy_r": round(expectancy_r, 4),
            "r_sample_count": len(rr_values),
            "profit_factor": round(profit_factor, 3),
            "score": round(performance_score, 2),
            # This is now a soft probation flag, not a permanent exclusion.
            "blocked": blocked,
        }

    @staticmethod
    def score(
        symbol: str,
        analysis: Dict[str, Any],
        capital_fit: Dict[str, Any],
    ) -> Dict[str, Any]:
        indicators = analysis.get("indicators", {}) or {}
        structure = analysis.get("market_structure", {}) or {}
        broker_open = bool(capital_fit.get("broker_open", True))
        capital_fit_now = bool(capital_fit.get("capital_fit"))
        adx = max(0.0, _finite(indicators.get("adx_14")))
        state = str(structure.get("trend_state", "NEUTRAL") or "NEUTRAL").upper()
        spread = max(0.0, _finite(capital_fit.get("spread_value")))
        asset_class = str(capital_fit.get("asset_class", "FX/CFD")).upper()
        spread_limit = (
            settings.max_crypto_spread_bps
            if asset_class == "CRYPTO"
            else settings.max_spread_pips
        )
        spread_headroom = max(0.0, 1.0 - spread / max(spread_limit, 1e-9))
        try:
            adx_delta = _finite(indicators.get("adx_delta"))
        except (TypeError, ValueError, OverflowError):
            adx_delta = 0.0
        events = structure.get("structure_events", []) or []
        breakout = str(structure.get("breakout_status", "") or "").upper()
        range_setup = structure.get("range_reversion", {}) or {}
        has_range_candidate = bool(range_setup.get("candidate"))
        has_fresh_structure = any(
            isinstance(event, dict)
            and str(event.get("type", "")).upper() in {"BOS", "CHOCH"}
            for event in events
        ) or "BREAKOUT" in breakout or has_range_candidate
        performance = capital_fit.get("performance", {}) or {}
        performance_score = max(
            -20.0, min(20.0, _finite(performance.get("score")))
        )
        performance_probation = bool(performance.get("blocked", False))
        forex_context = capital_fit.get("forex_context", {}) or {}
        currency_context_score = 0.0
        context_bias = str(forex_context.get("bias", "NEUTRAL")).upper()
        if (
            forex_context.get("reliable")
            and context_bias in {"BULLISH", "BEARISH"}
        ):
            currency_context_score = (
                10.0 if forex_context.get("aligned_with_regime") else -10.0
            )
        state_points = 0.0
        if has_range_candidate:
            state_points = 15.0
        elif state.startswith("CONFIRMED_"):
            state_points = 20.0
        elif state.startswith("PULLBACK_"):
            state_points = 16.0
        elif state.startswith("EARLY_"):
            state_points = 10.0

        # Discovery must not disappear merely because the account cannot fund
        # the *current* technical stop at minimum volume.  Position sizing is a
        # final execution gate, not market evidence and not a ranking input.
        score = 0.0
        if broker_open:
            score = (
                20.0
                + (
                    20.0
                    if has_range_candidate
                    else min(25.0, adx / 1.5)
                )
                + state_points
                + spread_headroom * 15.0
                + (5.0 if adx_delta > 0 else 0.0)
                + (15.0 if has_fresh_structure else 0.0)
                + performance_score
                + (-12.0 if performance_probation else 0.0)
                + currency_context_score
            )
        return {
            "symbol": str(symbol).upper(),
            "selection_score": round(min(100.0, max(0.0, score)), 1),
            "selection_regime": state,
            "selection_adx": round(adx, 1),
            "selection_adx_delta": round(adx_delta, 2),
            "selection_has_structure": has_fresh_structure,
            "selection_performance": performance,
            "selection_forex_context": forex_context,
            "performance_blocked": False,
            "performance_probation": performance_probation,
            "selection_reason": (
                (
                    "Broker-open market ranked by trend/range setup, fresh structure, ADX, "
                    "spread, currency context, and broker-confirmed expectancy; "
                    + (
                        "the current technical plan is capital-fit"
                        if capital_fit_now
                        else "position sizing remains blocked until a technical plan fits the account"
                    )
                    + (
                        "; recent results apply a soft performance-probation penalty"
                        if performance_probation
                        else ""
                    )
                )
                if broker_open
                else (
                    str(
                        capital_fit.get(
                            "reason",
                            "Broker market is not open with a fresh tradable quote",
                        )
                    )
                )
            ),
            "selected": False,
        }

    @staticmethod
    def select(
        ranked: Iterable[Dict[str, Any]],
        maximum: int,
    ) -> List[str]:
        """Select active markets while retaining non-fit discovery fallbacks.

        Ranking remains evidence-based, but a market whose current minimum
        broker volume fits the account must not lose an active slot to a
        higher-scoring market that cannot currently be executed.  Non-fit
        markets remain eligible to fill any slots left after the capital-fit
        tier, so discovery stays broad.
        """
        eligible = [
            item
            for item in ranked
            if item.get("broker_open", True)
            and _finite(item.get("selection_score")) > 0.0
        ]
        eligible.sort(
            key=lambda item: (
                _finite(item.get("selection_score")),
                -_finite(item.get("spread_value"), math.inf),
                str(item.get("symbol", "")),
            ),
            reverse=True,
        )
        capital_fit = [item for item in eligible if item.get("capital_fit") is True]
        discovery = [item for item in eligible if item.get("capital_fit") is not True]
        prioritized = capital_fit + discovery
        return [
            str(item.get("symbol", "")).upper()
            for item in prioritized[: max(1, int(maximum))]
            if str(item.get("symbol", "")).strip()
        ]

import asyncio
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

from app_config.settings import settings
from core.engine import TradingEngine
from core.forex_context import build_currency_context
from core.market_selector import AdaptiveMarketSelector


def _analysis(adx=30.0, state="CONFIRMED_BULLISH"):
    return {
        "indicators": {"adx_14": adx},
        "market_structure": {"trend_state": state},
    }


def _fit(symbol, *, fit=True, spread=1.0, risk=0.5, budget=0.75, margin=20.0):
    return {
        "symbol": symbol,
        "capital_fit": fit,
        "reason": "fit" if fit else "unaffordable",
        "spread_value": spread,
        "spread_unit": "pips",
        "asset_class": "FX/CFD",
        "min_stop_risk_usd": risk,
        "risk_budget_usd": budget,
        "projected_margin_pct": margin,
    }


class AdaptiveMarketSelectorTests(unittest.TestCase):
    def test_unaffordable_market_is_still_discovered_without_budget_scoring(self):
        result = _fit("CADJPY", fit=False)
        result.update(AdaptiveMarketSelector.score("CADJPY", _analysis(), result))

        affordable = _fit("USDJPY", fit=True)
        affordable.update(
            AdaptiveMarketSelector.score("USDJPY", _analysis(), affordable)
        )
        self.assertGreater(result["selection_score"], 0.0)
        self.assertEqual(
            result["selection_score"], affordable["selection_score"]
        )
        self.assertEqual(AdaptiveMarketSelector.select([result], 6), ["CADJPY"])

    def test_stronger_lower_spread_market_ranks_first(self):
        strong = _fit("USDJPY", spread=0.8, risk=0.50, margin=15.0)
        weak = _fit("USDCAD", spread=2.8, risk=0.70, margin=60.0)
        strong.update(
            AdaptiveMarketSelector.score("USDJPY", _analysis(adx=35.0), strong)
        )
        weak.update(
            AdaptiveMarketSelector.score(
                "USDCAD", _analysis(adx=20.0, state="NEUTRAL"), weak
            )
        )

        self.assertGreater(strong["selection_score"], weak["selection_score"])
        self.assertEqual(
            AdaptiveMarketSelector.select([weak, strong], 2),
            ["USDJPY", "USDCAD"],
        )

    def test_maximum_active_market_limit_is_enforced(self):
        ranked = []
        for index, symbol in enumerate(("USDJPY", "USDCAD", "EURUSD")):
            item = _fit(symbol, spread=0.5 + index * 0.1)
            item.update(
                AdaptiveMarketSelector.score(
                    symbol, _analysis(adx=35.0 - index), item
                )
            )
            ranked.append(item)

        self.assertEqual(len(AdaptiveMarketSelector.select(ranked, 2)), 2)

    def test_capital_fit_markets_fill_active_slots_before_non_fit_discovery(self):
        high_score_non_fit = _fit("CADJPY", fit=False)
        high_score_non_fit["selection_score"] = 99.0
        first_fit = _fit("USDJPY", fit=True)
        first_fit["selection_score"] = 80.0
        second_fit = _fit("USDCAD", fit=True)
        second_fit["selection_score"] = 70.0

        self.assertEqual(
            AdaptiveMarketSelector.select(
                [high_score_non_fit, second_fit, first_fit], 2
            ),
            ["USDJPY", "USDCAD"],
        )

    def test_non_fit_discovery_fills_slots_after_fit_tier(self):
        fit = _fit("USDJPY", fit=True)
        fit["selection_score"] = 70.0
        best_discovery = _fit("CADJPY", fit=False)
        best_discovery["selection_score"] = 95.0
        other_discovery = _fit("EURUSD", fit=False)
        other_discovery["selection_score"] = 85.0

        self.assertEqual(
            AdaptiveMarketSelector.select(
                [other_discovery, fit, best_discovery], 2
            ),
            ["USDJPY", "CADJPY"],
        )

    def test_broker_closed_market_is_never_selected(self):
        closed = _fit("ETHUSD")
        closed["broker_open"] = False
        closed["reason"] = "Broker quote is stale"
        closed.update(
            AdaptiveMarketSelector.score("ETHUSD", _analysis(), closed)
        )

        self.assertEqual(closed["selection_score"], 0.0)
        self.assertEqual(AdaptiveMarketSelector.select([closed], 1), [])

    def test_unused_risk_budget_does_not_change_market_rank(self):
        first = _fit("EURUSD", risk=0.20, budget=5.0, margin=5.0)
        second = _fit("GBPUSD", risk=0.90, budget=0.91, margin=69.0)
        first.update(AdaptiveMarketSelector.score("EURUSD", _analysis(), first))
        second.update(AdaptiveMarketSelector.score("GBPUSD", _analysis(), second))

        self.assertEqual(first["selection_score"], second["selection_score"])

    def test_negative_broker_expectancy_applies_soft_probation(self):
        history = [
            {
                "magic": settings.strategy_magic,
                "net_profit": -0.60,
                "rr_achieved": -1.0,
                "initial_risk_usd": 0.60,
                "close_reason": "SL",
            }
            for _ in range(settings.market_performance_min_trades)
        ]
        performance = AdaptiveMarketSelector.summarize_performance(
            history, strategy_magic=settings.strategy_magic
        )
        result = _fit("EURUSD")
        result["performance"] = performance
        result.update(AdaptiveMarketSelector.score("EURUSD", _analysis(), result))
        baseline = _fit("GBPUSD")
        baseline.update(
            AdaptiveMarketSelector.score("GBPUSD", _analysis(), baseline)
        )

        self.assertTrue(performance["blocked"])
        self.assertTrue(result["performance_probation"])
        self.assertFalse(result["performance_blocked"])
        self.assertLess(result["selection_score"], baseline["selection_score"])
        self.assertEqual(
            AdaptiveMarketSelector.select([result], 1),
            ["EURUSD"],
        )

    def test_manual_closes_are_excluded_from_performance_evidence(self):
        history = [
            {
                "magic": settings.strategy_magic,
                "net_profit": -9.0,
                "rr_achieved": -9.0,
                "initial_risk_usd": 1.0,
                "close_reason": close_reason,
            }
            for close_reason in ("CLIENT", "mobile", "Web")
        ]
        history.append(
            {
                "magic": settings.strategy_magic,
                "net_profit": 0.25,
                "rr_achieved": 0.5,
                "initial_risk_usd": 0.5,
                "close_reason": "TP",
            }
        )

        performance = AdaptiveMarketSelector.summarize_performance(
            history, strategy_magic=settings.strategy_magic
        )

        self.assertEqual(performance["trade_count"], 1)
        self.assertEqual(performance["r_sample_count"], 1)
        self.assertEqual(performance["expectancy_usd"], 0.25)
        self.assertEqual(performance["expectancy_r"], 0.5)

    def test_missing_initial_risk_is_not_counted_as_an_r_sample(self):
        history = [
            {
                "magic": settings.strategy_magic,
                "net_profit": 5.0,
                "rr_achieved": 10.0,
                "close_reason": "TP",
            },
            {
                "magic": settings.strategy_magic,
                "net_profit": -0.5,
                "rr_achieved": -1.0,
                "initial_risk_usd": 0.5,
                "close_reason": "SL",
            },
        ]

        performance = AdaptiveMarketSelector.summarize_performance(
            history, strategy_magic=settings.strategy_magic
        )

        self.assertEqual(performance["trade_count"], 2)
        self.assertEqual(performance["r_sample_count"], 1)
        self.assertEqual(performance["expectancy_r"], -1.0)

    def test_broker_cross_pair_context_detects_currency_bias(self):
        contexts = build_currency_context([
            {
                "symbol": "EURUSD",
                "m5_impulse_atr": 1.0,
                "selection_regime": "CONFIRMED_BULLISH",
            },
            {
                "symbol": "EURGBP",
                "m5_impulse_atr": 0.8,
                "selection_regime": "CONFIRMED_BULLISH",
            },
            {
                "symbol": "GBPUSD",
                "m5_impulse_atr": -0.4,
                "selection_regime": "CONFIRMED_BEARISH",
            },
        ])

        self.assertTrue(contexts["EURUSD"]["reliable"])
        self.assertEqual(contexts["EURUSD"]["bias"], "BULLISH")

    def test_single_pair_cannot_confirm_itself_as_currency_context(self):
        contexts = build_currency_context([
            {
                "symbol": "EURUSD",
                "m5_impulse_atr": 2.0,
                "selection_regime": "CONFIRMED_BULLISH",
            },
        ])

        context = contexts["EURUSD"]
        self.assertFalse(context["reliable"])
        self.assertEqual(context["bias"], "NEUTRAL")
        self.assertEqual(context["pair_strength"], 0.0)
        self.assertEqual(context["currency_samples"], {"EUR": 0, "USD": 0})

    def test_target_pair_is_left_out_of_its_currency_strength_context(self):
        supporting_rows = [
            {
                "symbol": "EURGBP",
                "m5_impulse_atr": 0.8,
                "selection_regime": "CONFIRMED_BULLISH",
            },
            {
                "symbol": "GBPUSD",
                "m5_impulse_atr": -0.4,
                "selection_regime": "CONFIRMED_BEARISH",
            },
        ]
        bullish_target = build_currency_context([
            {
                "symbol": "EURUSD",
                "m5_impulse_atr": 2.0,
                "selection_regime": "CONFIRMED_BULLISH",
            },
            *supporting_rows,
        ])["EURUSD"]
        bearish_target = build_currency_context([
            {
                "symbol": "EURUSD",
                "m5_impulse_atr": -2.0,
                "selection_regime": "CONFIRMED_BEARISH",
            },
            *supporting_rows,
        ])["EURUSD"]

        self.assertTrue(bullish_target["reliable"])
        self.assertEqual(
            bullish_target["base_strength"],
            bearish_target["base_strength"],
        )
        self.assertEqual(
            bullish_target["quote_strength"],
            bearish_target["quote_strength"],
        )
        self.assertEqual(
            bullish_target["pair_strength"],
            bearish_target["pair_strength"],
        )
        self.assertEqual(bullish_target["bias"], bearish_target["bias"])

    def test_reliable_neutral_currency_context_adds_no_selection_points(self):
        baseline = _fit("EURUSD")
        baseline.update(
            AdaptiveMarketSelector.score("EURUSD", _analysis(), baseline)
        )
        neutral = _fit("EURUSD")
        neutral["forex_context"] = {
            "reliable": True,
            "bias": "NEUTRAL",
            "aligned_with_regime": True,
        }
        neutral.update(
            AdaptiveMarketSelector.score("EURUSD", _analysis(), neutral)
        )

        self.assertEqual(
            neutral["selection_score"],
            baseline["selection_score"],
        )


class AdaptiveMarketRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_model_eligibility_requires_actionable_m5_evidence(self):
        frame = MagicMock()
        frame.empty = False
        frame.attrs = {}
        reader = MagicMock()
        reader.get_ohlcv = AsyncMock(return_value=frame)
        database = MagicMock()
        database.get_closed_positions = AsyncMock(return_value=[])
        engine = TradingEngine.__new__(TradingEngine)
        engine.reader = reader
        engine.db = database
        base_analysis = {
            "timestamp": "2026-08-12 05:10:00+00:00",
            "indicators": {"adx_14": 30.0, "adx_delta": 1.0, "current_price": 100.0,
                           "atr_14": 1.0, "ema_9": 100.1, "ema_21": 100.0,
                           "rsi_14": 55.0, "macd": {"diff": .1}},
            "market_structure": {
                "trend": "BULLISH",
                "trend_state": "CONFIRMED_BULLISH",
                "trend_state_direction": "BULLISH",
                "structure_events": [],
                "breakout_status": "NONE",
            },
        }
        engine.analyzer = MagicMock()
        engine.analyzer.analyze.return_value = base_analysis

        with (
            patch("core.engine.annotate_range_reversion"),
            patch(
                "core.engine.DeterministicTradePlanner.assess_capital_fit",
                side_effect=lambda *args, **kwargs: {
                    "capital_fit": True,
                    "broker_open": True,
                    "spread_value": 1.0,
                    "asset_class": "FX/CFD",
                },
            ),
            patch("core.engine.dashboard_state.update_market_fit"),
        ):
            no_trigger = await engine._assess_market_candidate(
                "USDJPY", {"login": 1}
            )
            base_analysis["market_structure"]["structure_events"] = [
                {
                    "type": "BOS",
                    "direction": "BULLISH",
                    "time": "2026-08-12 05:10:00+00:00",
                    "level": 99.5,
                }
            ]
            with_trigger = await engine._assess_market_candidate(
                "USDJPY", {"login": 1}
            )

        self.assertFalse(no_trigger["actionable_entry_evidence"])
        self.assertFalse(no_trigger["model_eligible"])
        self.assertTrue(with_trigger["actionable_entry_evidence"])
        self.assertEqual(with_trigger["permitted_entry_actions"], ["BUY"])
        self.assertTrue(with_trigger["model_eligible"])

    async def test_refresh_ranks_open_markets_independently_of_capital_fit(self):
        isolated = replace(
            settings,
            broker_market_discovery_enabled=False,
            dynamic_market_selection_enabled=True,
            trading_symbols=["USDJPY"],
            market_candidate_symbols=["USDJPY", "USDCAD", "CADJPY"],
            dynamic_market_max_symbols=2,
            llm_entry_candidates_per_bar=1,
        )
        with (
            patch("core.engine.settings", isolated),
            patch("core.engine.is_weekend", return_value=False),
        ):
            engine = TradingEngine(
                object(), object(), object(), object(), object(), object()
            )
            engine.log = lambda *args, **kwargs: None
            rows = {
                "USDJPY": {**_fit("USDJPY"), "selection_score": 90.0, "model_eligible": True},
                "USDCAD": {**_fit("USDCAD"), "selection_score": 80.0, "model_eligible": True},
                "CADJPY": {
                    **_fit("CADJPY", fit=False),
                    "selection_score": 95.0,
                },
            }
            engine._assess_market_candidate = AsyncMock(
                side_effect=lambda symbol, account: dict(rows[symbol])
            )

            await engine._refresh_dynamic_market_selection({"balance": 12.0})

        self.assertEqual(engine._selected_symbols, ("USDJPY", "USDCAD"))
        self.assertTrue(engine._market_selection_initialized)
        self.assertTrue(engine._market_rankings["USDJPY"]["model_selected"])
        self.assertEqual(
            engine._market_rankings["USDJPY"]["model_selection_rank"], 1
        )
        self.assertFalse(engine._market_rankings["USDCAD"]["model_selected"])
        self.assertEqual(
            engine._market_rankings["USDCAD"]["model_selection_rank"], 2
        )

    async def test_refresh_from_superseded_account_cannot_replace_selection(self):
        isolated = replace(
            settings,
            broker_market_discovery_enabled=False,
            dynamic_market_selection_enabled=True,
            trading_symbols=["USDJPY"],
            market_candidate_symbols=["USDJPY", "USDCAD"],
            dynamic_market_max_symbols=1,
        )
        old_account = {
            "login": 1001,
            "server": "Pepperstone-Demo",
            "company": "Pepperstone",
            "trade_mode": 0,
        }
        new_account = {
            "login": 2002,
            "server": "Pepperstone-Live",
            "company": "Pepperstone",
            "trade_mode": 2,
        }
        with (
            patch("core.engine.settings", isolated),
            patch("core.engine.is_weekend", return_value=False),
        ):
            engine = TradingEngine(
                object(), object(), object(), object(), object(), object()
            )
            engine.log = lambda *args, **kwargs: None
            engine._active_account_identity = dict(new_account)
            engine._selected_symbols = ("USDJPY",)
            engine._market_selection_initialized = False
            engine._assess_market_candidate = AsyncMock(
                side_effect=lambda symbol, account: {
                    **_fit(symbol),
                    "selection_score": (
                        99.0 if symbol == "USDCAD" else 10.0
                    ),
                }
            )

            await engine._refresh_dynamic_market_selection(old_account)

        self.assertEqual(engine._selected_symbols, ("USDJPY",))
        self.assertFalse(engine._market_selection_initialized)

    async def test_market_refresh_failure_callback_sets_aggregate_error(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.log = MagicMock()

        async def fail():
            raise RuntimeError("discovery failed")

        task = asyncio.create_task(fail())
        try:
            await task
        except RuntimeError:
            pass
        engine.market_selection_task = task

        with patch("core.engine.dashboard_state") as state:
            engine._market_selection_finished(task)

        self.assertIsNone(engine.market_selection_task)
        engine.log.assert_called_once()
        state.update_automation.assert_called_once_with(
            scan_status="MARKET DISCOVERY ERROR"
        )

    async def test_market_refresh_waits_until_entry_analysis_drains(self):
        isolated = replace(
            settings,
            dynamic_market_selection_enabled=True,
            market_selection_refresh_seconds=60,
        )
        account = {
            "login": 1001,
            "server": "Pepperstone-Demo",
            "company": "Pepperstone",
            "trade_mode": 0,
        }
        with patch("core.engine.settings", isolated):
            engine = TradingEngine(
                object(), object(), object(), object(), object(), object()
            )
            engine.is_running = True
            engine._active_account_identity = dict(account)
            engine._market_selection_initialized = False
            engine._last_market_selection_monotonic = 0.0
            engine._refresh_dynamic_market_selection = AsyncMock()

            blocker = asyncio.create_task(asyncio.Event().wait())
            engine.analysis_tasks["EURUSD"] = blocker
            engine._request_market_selection_refresh(account)

            self.assertIsNone(engine.market_selection_task)
            self.assertEqual(
                engine._pending_market_selection_account,
                account,
            )

            blocker.cancel()
            try:
                await blocker
            except asyncio.CancelledError:
                pass
            engine._analysis_finished("EURUSD", blocker)
            refresh_task = engine.market_selection_task
            self.assertIsNotNone(refresh_task)
            await refresh_task

        engine._refresh_dynamic_market_selection.assert_awaited_once_with(
            account
        )


if __name__ == "__main__":
    unittest.main()

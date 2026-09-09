import asyncio
import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from app_config.settings import settings
from core.analysis_engine import MarketAnalysisEngine
from core.engine import TradingEngine
from core.market_selector import AdaptiveMarketSelector
from core.market_universe import discovery_batch, tradable_symbols
from core.opportunities import screen_opportunities
from core.scoring import DecisionScoringEngine
from mt5.safe_api import _SerializedMT5Proxy
from risk.instruments import analysis_atr_price, analysis_is_crypto, is_crypto_symbol
from risk.manager import RiskManager


def instrument(name, **overrides):
    values = dict(name=name, custom=False, visible=False, trade_mode=4,
                  point=0.01, trade_tick_size=0.01, volume_min=0.1,
                  volume_step=0.1, order_mode=49)
    return SimpleNamespace(**(values | overrides))


def analysis(timeframe, *, direction="BULLISH", range_setup=False):
    return {
        "symbol": "TEST", "timeframe": timeframe,
        "timestamp": "2026-09-08 03:20:00+00:00",
        "indicators": {
            "current_price": 99.05 if range_setup else 100.0,
            "atr_14": 1.0, "atr_14_pips": 10.0,
            "adx_14": 14.0 if range_setup else 30.0,
            "adx_delta": -0.2 if range_setup else 0.5,
            "rsi_14": 32.0 if range_setup else (55.0 if direction == "BULLISH" else 45.0),
            "candle_body_atr_signed": 0.2, "candle_return_atr": 0.2,
            "candle_range_atr": 0.8, "opening_gap_atr": 0.0,
            "stochastic": {"k": 18.0 if range_setup else 60.0, "d": 18.0 if range_setup else 60.0},
            "bollinger_bands": {"lower": 99.0, "middle": 100.0, "upper": 101.0, "width_pct": 0.2},
            "macd": {"diff": 0.0 if range_setup else (.1 if direction == "BULLISH" else -.1)},
            "ema_9": 99.2 if range_setup else (100.1 if direction == "BULLISH" else 99.8),
            "ema_21": 99.4 if range_setup else 100.0,
        },
        "market_structure": {
            "trend": direction, "trend_state_direction": direction,
            "trend_state": "NEUTRAL" if range_setup else f"CONFIRMED_{direction}",
            "breakout_status": "NONE", "structure_events": [],
            "support": 98.8 if range_setup else 98.0,
            "resistance": 101.2 if range_setup else 102.0,
            "demand_zones": [], "supply_zones": [], "order_blocks": [],
            "fair_value_gaps": [], "liquidity_zones": {}, "candlestick_patterns": [],
        },
    }


def frames_with_bos(direction="BULLISH"):
    frames = {tf: analysis(tf, direction=direction) for tf in ("M5", "M15", "H1", "H4")}
    frames["M5"]["market_structure"]["structure_events"] = [{
        "type": "BOS", "direction": direction, "time": frames["M5"]["timestamp"],
        "level": 99.5 if direction == "BULLISH" else 100.5,
    }]
    return frames


class BrokerUniverseTests(unittest.TestCase):
    def test_discovery_excludes_unusable_contracts_and_prioritizes_visible(self):
        symbols = [instrument("Index"), instrument("SpotCrude", visible=True),
                   instrument("Closed", trade_mode=3), instrument("Disabled", trade_mode=0),
                   instrument("Custom", custom=True), instrument("BadPoint", point=float("nan")),
                   instrument("BadLot", volume_step=0), instrument("NoStops", order_mode=1),
                   instrument("LongOnly", trade_mode=1), instrument("ShortOnly", trade_mode=2)]
        self.assertEqual(tradable_symbols(symbols), ["SpotCrude", "Index", "LongOnly", "ShortOnly"])

    def test_ambiguous_case_is_not_guessed(self):
        self.assertEqual(tradable_symbols([instrument("Asset"), instrument("ASSET")]), [])
        self.assertEqual(tradable_symbols([instrument(" Asset ")]), [])

    def test_batches_are_bounded_rotate_and_preserve_active_markets(self):
        first, cursor = discovery_batch(["FX"], ["SELECTED"], ["FX", "A", "B", "C", "D"], 0, 2)
        second, cursor = discovery_batch(["FX"], ["SELECTED"], ["FX", "A", "B", "C", "D"], cursor, 2)
        self.assertEqual(first, ["FX", "SELECTED", "A", "B"])
        self.assertEqual(second, ["FX", "SELECTED", "C", "D"])
        self.assertEqual(cursor, 0)
        self.assertEqual(discovery_batch(["FX"], ["FX"], [], 99, 12), (["FX"], 0))

    def test_selection_changes_do_not_shift_the_rotation_cursor(self):
        universe = ["A", "B", "C", "D"]
        _, cursor = discovery_batch([], [], universe, 0, 2)
        batch, _ = discovery_batch([], ["A"], universe, cursor, 1)
        self.assertEqual(batch, ["A", "C"])

    def test_priority_and_catalog_are_case_insensitively_deduplicated(self):
        batch, _ = discovery_batch(["fx"], ["FX"], ["Asset", "ASSET", "FX"], 0, 12)
        self.assertEqual(batch, ["FX", "ASSET"])

    def test_broker_discovery_keeps_weekend_universe(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._broker_scan_symbols = ("STOCK",)
        engine._market_candidates_for_current_market = lambda: ["BTCUSD"]
        configured = replace(settings, broker_market_discovery_enabled=True)
        with patch("core.engine.settings", configured), patch("core.engine.is_weekend", return_value=True):
            self.assertEqual(engine._candidate_universe(), ["BTCUSD"])
            self.assertEqual(asyncio.run(engine._broker_discovery_batch(["BTCUSD"])), ["BTCUSD"])

    def test_stock_names_are_not_crypto_substring_matches(self):
        for symbol in ("ADANIPORTS", "UNILEVER", "CONSOLIDATED", "NEARBY"):
            self.assertFalse(is_crypto_symbol(symbol))
        self.assertFalse(is_crypto_symbol("SOLUSD", SimpleNamespace(path="Markets\\Stocks")))
        for symbol in ("BTCUSD", "ETHUSD.a", "SOLUSDm", "XRPUSD", "broker.BTCUSD"):
            self.assertTrue(is_crypto_symbol(symbol), symbol)
        self.assertTrue(is_crypto_symbol("UnusualName", SimpleNamespace(path="Markets\\Crypto")))


class BrokerSymbolMappingTests(unittest.TestCase):
    def setUp(self):
        self.native = SimpleNamespace(
            symbol_info=MagicMock(), copy_rates_from_pos=MagicMock(),
            order_calc_profit=MagicMock(), positions_get=MagicMock(),
            order_check=MagicMock(), order_send=MagicMock(),
            initialize=MagicMock(return_value=True), login=MagicMock(return_value=True), shutdown=MagicMock(),
            symbols_get=MagicMock(return_value=[instrument("SpotCrude")]),
        )
        self.proxy = _SerializedMT5Proxy(self.native)
        self.proxy.register_symbols([instrument("SpotCrude"), instrument("EURUSD.a")])

    def test_quotes_bars_profit_and_position_filters_keep_exact_broker_case(self):
        self.assertEqual(self.proxy.broker_symbol_name("SPOTCRUDE"), "SpotCrude")
        self.proxy.symbol_info("SPOTCRUDE")
        self.native.symbol_info.assert_called_once_with("SpotCrude")
        self.proxy.copy_rates_from_pos("EURUSD.A", 5, 1, 220)
        self.native.copy_rates_from_pos.assert_called_once_with("EURUSD.a", 5, 1, 220)
        self.proxy.order_calc_profit(0, "SPOTCRUDE", 1.0, 70.0, 71.0)
        self.native.order_calc_profit.assert_called_once_with(0, "SpotCrude", 1.0, 70.0, 71.0)
        self.proxy.positions_get(symbol="SPOTCRUDE")
        self.native.positions_get.assert_called_once_with(symbol="SpotCrude")

    def test_native_order_dictionary_is_exact_and_caller_is_not_mutated(self):
        request = {"symbol": "SPOTCRUDE", "volume": 1.0}
        self.proxy.order_check(request)
        self.proxy.order_send(request)
        self.native.order_check.assert_called_once_with({"symbol": "SpotCrude", "volume": 1.0})
        self.native.order_send.assert_called_once_with({"symbol": "SpotCrude", "volume": 1.0})
        self.assertEqual(request["symbol"], "SPOTCRUDE")

    def test_session_initialization_rebuilds_mapping_and_shutdown_clears_it(self):
        self.proxy.initialize()
        self.proxy.symbol_info("SPOTCRUDE")
        self.native.symbol_info.assert_called_with("SpotCrude")
        self.proxy.shutdown()
        self.proxy.symbol_info("SPOTCRUDE")
        self.native.symbol_info.assert_called_with("SPOTCRUDE")

    def test_case_collision_never_redirects_to_another_contract(self):
        self.proxy.register_symbols([instrument("Asset"), instrument("ASSET")])
        self.proxy.symbol_info("ASSET")
        self.native.symbol_info.assert_called_with("ASSET")

    def test_login_refreshes_contract_mapping_for_the_new_account(self):
        self.native.symbols_get.return_value = [instrument("NatGas")]
        self.proxy.login(login=123)
        self.proxy.symbol_info("NATGAS")
        self.native.symbol_info.assert_called_with("NatGas")
        self.assertNotIn("SPOTCRUDE", self.proxy._symbol_names)


class BrokerDiscoveryRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_visibility_changes_do_not_restart_or_reorder_rotation(self):
        engine = TradingEngine.__new__(TradingEngine)
        configured = replace(settings, broker_market_discovery_enabled=True, broker_market_group="*", broker_market_batch_size=1, broker_market_refresh_seconds=60.0)
        catalogs = [
            [instrument("A"), instrument("B", visible=True), instrument("C")],
            [instrument("A", visible=True), instrument("B", visible=True), instrument("C")],
        ]
        with (
            patch("core.engine.settings", configured),
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.time.monotonic", side_effect=[100.0, 200.0]),
            patch("core.engine.mt5.symbols_get", side_effect=catalogs),
            patch("core.engine.mt5.register_symbols"),
        ):
            first = await engine._broker_discovery_batch([])
            second = await engine._broker_discovery_batch([])
        self.assertEqual(first, ["B"])
        self.assertEqual(second, ["A"])
        self.assertEqual(engine._broker_universe, ["B", "A", "C"])

    async def test_filtered_catalog_cannot_hide_a_case_collision(self):
        engine = TradingEngine.__new__(TradingEngine)
        configured = replace(settings, broker_market_discovery_enabled=True, broker_market_group="Asset")
        with (
            patch("core.engine.settings", configured),
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.mt5.symbols_get", side_effect=[[instrument("Asset")], [instrument("Asset"), instrument("ASSET")]]),
            patch("core.engine.mt5.register_symbols"),
        ):
            batch = await engine._broker_discovery_batch(["FX"])
        self.assertEqual(batch, ["FX"])


class OpportunityScreenTests(unittest.TestCase):
    def test_fresh_bos_is_eligible_for_both_directions(self):
        for direction, action in (("BULLISH", "BUY"), ("BEARISH", "SELL")):
            result = screen_opportunities(frames_with_bos(direction), {"capital_fit": True})
            self.assertEqual(result["viable_entry_actions"], [action], result)
            self.assertTrue(result["model_eligible"])

    def test_no_trigger_keeps_model_lane_free(self):
        frames = {tf: analysis(tf) for tf in ("M5", "M15", "H1", "H4")}
        self.assertFalse(screen_opportunities(frames, {"capital_fit": True})["model_eligible"])

    def test_direction_affordability_is_not_borrowed_from_opposite_side(self):
        result = screen_opportunities(frames_with_bos(), {
            "capital_fit": True, "directions": {"BUY": {"capital_fit": False}, "SELL": {"capital_fit": True}},
        })
        self.assertFalse(result["model_eligible"])
        self.assertIn("account", result["opportunity_rejections"]["BUY"])

    def test_exhausted_signal_fails_before_model_confidence_is_requested(self):
        frames = frames_with_bos()
        frames["M5"]["market_structure"]["breakout_status"] = "BULLISH BREAKOUT"
        frames["M5"]["indicators"].update(rsi_14=75.0, stochastic={"k": 99.0, "d": 99.0})
        result = screen_opportunities(frames, {"capital_fit": True})
        self.assertFalse(result["model_eligible"])
        self.assertIn("Continuation Confirmation", result["opportunity_rejections"]["BUY"])

    def test_qualified_failed_thesis_mode_is_considered_without_lowering_final_confidence(self):
        with (
            patch("core.opportunities.RiskManager.qualify_failed_thesis_reversal", return_value=(True, "qualified")) as qualify,
            patch("core.opportunities.RiskManager._check_entry_structure", side_effect=[(False, "H1 opposition"), (True, "")]) as gate,
        ):
            result = screen_opportunities(frames_with_bos(), {"capital_fit": True}, [{"action": "SELL", "net_profit": -1.0}])
        self.assertTrue(result["model_eligible"])
        self.assertEqual(qualify.call_args.args[1], settings.failed_thesis_reversal_min_confidence)
        self.assertEqual(gate.call_args.kwargs["strategy_mode"], "FAILED_THESIS_REVERSAL")
        self.assertNotIn("confidence", result)

    def test_closed_market_is_not_model_eligible(self):
        self.assertFalse(screen_opportunities(frames_with_bos(), {"capital_fit": True, "broker_open": False})["model_eligible"])

    def test_ready_setup_outranks_higher_scoring_waiting_market(self):
        ranked = [
            {"symbol": "WAIT", "capital_fit": True, "selection_score": 99.0, "model_eligible": False},
            {"symbol": "READY", "capital_fit": True, "selection_score": 60.0, "model_eligible": True},
        ]
        self.assertEqual(AdaptiveMarketSelector.select(ranked, 1), ["READY"])


class ConfirmedDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    def engine(self, analyses, times=None):
        engine = TradingEngine.__new__(TradingEngine)
        frame = pd.DataFrame({"time": times if times is not None else pd.date_range("2026-09-08 03:15", periods=2, freq="5min", tz="UTC")})
        engine.reader = SimpleNamespace(get_ohlcv=AsyncMock(return_value=frame))
        engine.db = SimpleNamespace(get_closed_positions=AsyncMock(return_value=[]))
        engine.analyzer = MagicMock()
        engine.analyzer.analyze.side_effect = lambda symbol, tf, candles: analyses[tf]
        return engine, frame

    async def test_newly_discovered_retest_uses_previous_closed_candle(self):
        analyses = {tf: analysis(tf) for tf in ("M5", "M15", "H1", "H4")}
        engine, frame = self.engine(analyses)
        previous = copy.deepcopy(analyses["M5"])
        previous["market_structure"]["trend_state"] = "PULLBACK_IN_BULLISH_TREND"
        engine.analyzer.analyze.side_effect = lambda symbol, tf, candles: previous if tf == "M5" and len(candles) == 1 else analyses[tf]
        with patch("core.entry_retest.settings", replace(settings, retest_continuation_enabled=True)):
            result = await engine._confirmed_entry_analyses("NEW", frame)
        self.assertIn("retest_continuation", result["M5"]["market_structure"])
        self.assertNotIn("retest_continuation", analyses["M5"]["market_structure"])

    async def test_session_gap_does_not_create_retest(self):
        analyses = {tf: analysis(tf) for tf in ("M5", "M15", "H1", "H4")}
        engine, frame = self.engine(analyses, pd.to_datetime(["2026-09-04 21:55Z", "2026-09-07 00:00Z"]))
        result = await engine._confirmed_entry_analyses("NEW", frame)
        self.assertEqual(engine.analyzer.analyze.call_count, 4)
        self.assertNotIn("retest_continuation", result["M5"]["market_structure"])

    async def test_range_discovery_uses_all_confirmations_before_ranking(self):
        analyses = {tf: analysis(tf, direction="NEUTRAL", range_setup=True) for tf in ("M5", "M15", "H1", "H4")}
        engine, _ = self.engine(analyses)
        configured = replace(settings, range_reversion_enabled=True, entry_min_adx=20.0)
        with (
            patch("core.range_reversion.settings", configured),
            patch("core.engine.settings", configured),
            patch("risk.manager.settings", configured),
            patch("core.engine.DeterministicTradePlanner.assess_capital_fit", return_value={"capital_fit": True, "broker_open": True}),
            patch("core.engine.dashboard_state.update_market_fit"),
        ):
            result = await engine._assess_market_candidate("NEW", {"login": 1})
        self.assertTrue(result["model_eligible"], result)
        self.assertEqual(result["opportunity_modes"]["BUY"], "RANGE_REVERSION")
        self.assertEqual(result["entry_prefilter_reason"], "")
        self.assertEqual({call.args[1] for call in engine.reader.get_ohlcv.await_args_list}, {"M5", "M15", "H1", "H4"})

    async def test_stale_confirmation_rejects_discovery_before_analysis(self):
        engine, frame = self.engine({})
        frame.attrs["is_stale"] = True
        self.assertEqual(await engine._confirmed_entry_analyses("NEW", frame), {})
        engine.analyzer.analyze.assert_not_called()


class AnalysisContractTests(unittest.TestCase):
    def test_crypto_volatility_is_independent_of_broker_quote_digits(self):
        for one_pip in (0.01, 0.1, 1.0):
            m5 = analysis("M5")
            m5.update(symbol="ETHUSD", asset_class="CRYPTO")
            m5["indicators"].update(current_price=2000.0, atr_14=0.5, atr_14_pips=0.5 / one_pip, pip_size=one_pip)
            _, factors = DecisionScoringEngine.calculate_confluence_score("BUY", m5, m5, m5, m5)
            self.assertFalse(factors["atr_expansion"])
            allowed, reason = RiskManager._check_minimum_volatility("ETHUSD", m5)
            self.assertFalse(allowed, reason)
            m5["indicators"].update(atr_14=2.0, atr_14_pips=2.0 / one_pip)
            self.assertTrue(RiskManager._check_minimum_volatility("ETHUSD", m5)[0])

    def test_cfd_support_proximity_uses_broker_pips_consistently(self):
        m5 = analysis("M5")
        m5.update(symbol="NAS100", asset_class="FX/CFD")
        m5["indicators"].update(current_price=100.0, atr_14=1.0, atr_14_pips=10.0, pip_size=0.1)
        m5["market_structure"]["support"] = 99.0
        _, factors = DecisionScoringEngine.calculate_confluence_score("BUY", m5, m5, m5, m5)
        self.assertTrue(factors["sup_res_proximity"])

    def test_raw_atr_fallback_and_metadata_classification(self):
        legacy = {"symbol": "EURUSD", "indicators": {"atr_14_pips": 10.0}}
        self.assertAlmostEqual(analysis_atr_price(legacy), 0.001)
        custom = {"symbol": "UnusualName", "asset_class": "CRYPTO", "indicators": {"atr_14_pips": 10.0, "pip_size": 0.25}}
        self.assertTrue(analysis_is_crypto(custom))
        self.assertAlmostEqual(analysis_atr_price(custom), 2.5)
        custom["indicators"]["atr_14"] = float("nan")
        self.assertEqual(analysis_atr_price(custom), 0.0)

    def test_invalid_volatility_fails_closed(self):
        for value in (None, 0.0, float("nan"), "invalid"):
            m5 = {"symbol": "EURUSD", "indicators": {"atr_14_pips": value}}
            self.assertFalse(RiskManager._check_minimum_volatility("EURUSD", m5)[0])

    def test_broker_pip_scale_and_previous_bar_cache(self):
        bars = 240
        prices = [100.0 + index * 0.01 + (index % 7) * 0.02 for index in range(bars)]
        frame = pd.DataFrame({
            "time": pd.date_range("2026-09-07", periods=bars, freq="5min", tz="UTC"),
            "open": prices, "close": prices,
            "high": [value + 0.3 for value in prices], "low": [value - 0.3 for value in prices],
            "tick_volume": [100] * bars,
        })
        frame.attrs["pip_size"] = 0.1
        analyzer = MarketAnalysisEngine()
        current = analyzer.analyze("SpotCrude", "M5", frame)
        previous = analyzer.analyze("SpotCrude", "M5", frame.iloc[:-1])
        self.assertAlmostEqual(current["indicators"]["atr_14_pips"], current["indicators"]["atr_14"] / 0.1, places=3)
        self.assertEqual(analyzer.analyze("SpotCrude", "M5", frame), current)
        self.assertEqual(analyzer.analyze("SpotCrude", "M5", frame.iloc[:-1]), previous)


if __name__ == "__main__":
    unittest.main()

"""Live-path regression checks. All broker interactions are mocked; no orders."""

import copy
import unittest
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch, Mock

from app_config.settings import settings
from core.evidence import build_evidence_ids, has_directional_trigger, permitted_entry_actions
from core.engine import TradingEngine
from core.live_reversal import qualify_live_reversal, live_reversal_risk_percent
from core.opportunities import classify_reversal_watch, screen_opportunities
from core.scoring import DecisionScoringEngine
from core.trade_planner import DeterministicTradePlanner
from core.validator import DecisionValidator
from prompt_builder.generator import PromptGenerator
from risk.manager import RiskManager
from tests.test_reversal_watch import fixture
from tests.test_trade_planner import _instrument, _profit


def live_fixture(action="BUY", scale=1, year=2032):
    analyses, bars = fixture(action, scale, year)
    expected = "BULLISH" if action == "BUY" else "BEARISH"
    for tf in ("M5", "M15"):
        analyses[tf]["market_structure"]["trend_state_direction"] = expected
    for tf, analysis in analyses.items():
        analysis.update(symbol="EURUSD", timeframe=tf)
        analysis.setdefault("indicators", {}).update(adx_14=30, adx_delta=1)
    price = float(bars.iloc[-1].close)
    analyses["M5"]["indicators"].update(
        current_price=price, candle_range_atr=.6, candle_body_atr_signed=.4 if action == "BUY" else -.4,
        candle_return_atr=.4 if action == "BUY" else -.4, atr_14_pips=10,
        macd={"diff": .1 if action == "BUY" else -.1},
        stochastic={"k": 55 if action == "BUY" else 45, "d": 50},
        bollinger_bands={"upper": price+scale*2, "lower": price-scale*2, "width_pct": .2},
    )
    analyses["M5"]["market_structure"].update(support=price-3*scale, resistance=price+3*scale)
    analyses["M5"]["market_structure"]["reversal_watch"] = classify_reversal_watch(analyses, bars)
    analyses["M5"]["market_structure"]["live_reversal"] = qualify_live_reversal(analyses)
    return analyses, bars


class LiveReversalTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(settings, live_reversal_enabled=True, reversal_watch_enabled=True,
                              entry_min_adx=20, confirmation_min_adx=18)
        for module in ("core.live_reversal", "risk.manager", "core.scoring", "core.opportunities",
                       "core.trade_planner", "risk.instruments", "risk.execution_costs", "prompt_builder.generator"):
            context = patch(module + ".settings", self.config)
            context.start()
            self.addCleanup(context.stop)

    def test_symmetric_scale_and_calendar_independent_qualification(self):
        for action in ("BUY", "SELL"):
            for scale in (.00001, .01, 1, 1000):
                for year in (2020, 2026, 2032, 2040):
                    with self.subTest(action=action, scale=scale, year=year):
                        analyses, _ = live_fixture(action, scale, year)
                        result = qualify_live_reversal(analyses)
                        self.assertTrue(result["eligible"], result)
                        self.assertEqual(result["direction"], action)
                        self.assertEqual(result["max_risk_percent"], 2)
                        self.assertEqual(permitted_entry_actions(build_evidence_ids(analyses)), (action,))

    def test_opt_in_is_required_and_does_not_mutate_research(self):
        analyses, _ = live_fixture()
        original = copy.deepcopy(analyses)
        self.assertFalse(qualify_live_reversal(analyses, config=replace(self.config, live_reversal_enabled=False))["eligible"])
        self.assertEqual(analyses, original)
        self.assertFalse(analyses["M5"]["market_structure"]["reversal_watch"]["live_eligible"])

    def test_real_analysis_timestamp_format_and_broker_h4_phase(self):
        analyses, _ = live_fixture()
        for analysis in analyses.values():
            analysis["timestamp"] = analysis["timestamp"].replace("T", " ").replace("+00:00", "")
        # UTC H4 opens can shift with broker timezone/DST; not fixed at 00/04/08.
        analyses["H4"]["timestamp"] = "2032-03-01 07:00:00"
        self.assertTrue(qualify_live_reversal(analyses)["eligible"])
        self.assertEqual(permitted_entry_actions(build_evidence_ids(analyses)), ("BUY",))

    def test_incomplete_contradictory_and_forged_evidence_fails_closed(self):
        for problem in ("m15_opposed", "macro_aligned", "falling_adx", "nan", "missing_price", "wrong_price",
                        "future_h4", "stale_m15", "unaligned_clock", "old_version", "wrong_target", "watch_disabled", "eligible_flag_only"):
            with self.subTest(problem=problem):
                analyses, _ = live_fixture()
                m5 = analyses["M5"]
                if problem == "m15_opposed": analyses["M15"]["market_structure"]["trend_state_direction"] = "BEARISH"
                if problem == "macro_aligned": analyses["H1"]["market_structure"]["trend_state_direction"] = "BULLISH"
                if problem == "falling_adx": m5["indicators"]["adx_delta"] = -.01
                if problem == "nan": m5["indicators"]["adx_delta"] = float("nan")
                if problem == "missing_price": del m5["indicators"]["current_price"]
                if problem == "wrong_price": m5["indicators"]["current_price"] += .1
                if problem == "future_h4": analyses["H4"]["timestamp"] = "2032-03-01T12:00:00+00:00"
                if problem == "stale_m15": analyses["M15"]["timestamp"] = "2032-03-01T11:45:00+00:00"
                if problem == "unaligned_clock": m5["timestamp"] = "2032-03-01T12:25:01+00:00"
                if problem == "old_version": m5["market_structure"]["reversal_watch"]["strategy_version"] = "unknown"
                if problem == "wrong_target": m5["market_structure"]["reversal_watch"]["take_profit"] *= 2
                if problem == "watch_disabled": m5["market_structure"]["reversal_watch"]["candidate"] = False
                if problem == "eligible_flag_only": m5["market_structure"].pop("reversal_watch")
                self.assertFalse(qualify_live_reversal(analyses)["eligible"])
                self.assertEqual(permitted_entry_actions(build_evidence_ids(analyses)), ())

    def test_signal_age_boundary_blocks_delayed_inference(self):
        analyses, bars = live_fixture()
        closed = bars.iloc[-1].time.to_pydatetime() + timedelta(minutes=5)
        for seconds, expected in ((-1, False), (0, True), (120, True), (121, False)):
            self.assertEqual(qualify_live_reversal(analyses, now=closed+timedelta(seconds=seconds))["eligible"], expected)

    def test_catalog_is_entry_only_and_validator_requires_exact_direction(self):
        analyses, _ = live_fixture()
        ids = build_evidence_ids(analyses)
        trigger = qualify_live_reversal(analyses)["evidence_id"]
        self.assertFalse(has_directional_trigger([trigger], "BUY"))  # not exit-invalidation evidence
        for action, expected in (("BUY", True), ("SELL", False)):
            ok, _, _ = DecisionValidator.validate_decision(
                {"action": action, "confidence": .85, "evidence_ids": [trigger]}, allowed_evidence_ids=ids)
            self.assertEqual(ok, expected)

    def test_screening_and_final_structure_gate_agree_and_confidence_is_not_fabricated(self):
        for action in ("BUY", "SELL"):
            analyses, _ = live_fixture(action)
            frames = [analyses[tf] for tf in ("M5", "M15", "H1", "H4")]
            report = screen_opportunities(analyses, {"capital_fit": True, "broker_open": True})
            self.assertEqual(report["viable_entry_actions"], [action], report)
            self.assertEqual(report["opportunity_modes"][action], "LOCAL_REVERSAL")
            self.assertEqual(RiskManager._resolve_strategy_mode(action, {"_strategy": {"mode": "CONFIRMED_REVERSAL"}}, *frames), "LOCAL_REVERSAL")
            for confidence, expected in ((.60, False), (.849, False), (.85, True), (float("nan"), False)):
                ok, reason = RiskManager._check_entry_structure(action, *frames, strategy_mode="LOCAL_REVERSAL", decision_confidence=confidence)
                self.assertEqual(ok, expected, reason)

    def test_opposing_zone_gate_remains_active(self):
        analyses, _ = live_fixture()
        analyses["M5"]["market_structure"]["resistance"] = analyses["M5"]["indicators"]["current_price"] + .01
        ok, reason = RiskManager._check_entry_structure("BUY", *(analyses[tf] for tf in ("M5", "M15", "H1", "H4")),
                                                     strategy_mode="LOCAL_REVERSAL", decision_confidence=.99)
        self.assertFalse(ok)
        self.assertIn("Opposing Zone", reason)

    def test_fast_path_does_not_replace_model_confirmation(self):
        analyses, _ = live_fixture()
        self.assertIsNone(TradingEngine._deterministic_entry_fast_path(
            decision_context={"analyses": analyses, "has_open_position": False}, entry_contract=("BUY",)))

    def test_final_gate_expires_stale_signal_even_with_high_confidence(self):
        analyses, bars = live_fixture()
        frames = [analyses[tf] for tf in ("M5", "M15", "H1", "H4")]
        closed = bars.iloc[-1].time.to_pydatetime() + timedelta(minutes=5)
        with patch("risk.manager.datetime") as clock:
            clock.now.return_value = closed + timedelta(seconds=121)
            ok, reason = RiskManager._check_entry_structure("BUY", *frames, strategy_mode="LOCAL_REVERSAL",
                                                           decision_confidence=.99, market_snapshot=object())
        self.assertFalse(ok)
        self.assertIn("120 seconds", reason)

    def test_full_risk_path_applies_cap_and_rejects_unaffordable_minimum(self):
        analyses, _ = live_fixture()
        m5 = analyses["M5"]
        setup = m5["market_structure"]["live_reversal"]
        price = m5["indicators"]["current_price"]
        decision = {"action": "BUY", "confidence": .90, "entry": price,
                    "stop_loss": setup["invalidation_price"], "take_profit": setup["target_price"],
                    "_strategy": {"mode": "TREND_CONTINUATION"},
                    "_forex_context": {"active_sessions_utc": ["LONDON"]}}
        manager = RiskManager()
        # Isolate orchestration from account/session dependencies. Existing
        # broker safety tests exercise those individual gates independently.
        for name in ("_check_weekend", "_check_session", "_check_news", "_check_spread", "_check_max_positions",
                     "_check_duplicate", "_check_daily_loss", "_check_drawdown", "_check_loss_cooldown",
                     "_check_free_margin", "_check_margin_level", "_check_margin_usage", "_check_portfolio_risk", "_check_margin_for_lot"):
            setattr(manager, name, Mock(return_value=(True, "")))
        info = SimpleNamespace(volume_min=.01, volume_step=.01, volume_max=100, point=.001, path="Forex\\Majors")
        tick = SimpleNamespace(bid=price, ask=price)
        config = replace(self.config, risk_percent=10, auto_close_loss_usd=2, auto_close_loss_enabled=True,
                         max_order_deviation_points=0, fixed_execution_cost_usd=0, fx_round_turn_cost_usd_per_lot=0)
        with patch("risk.manager.settings", config), patch("risk.execution_costs.settings", config), \
             patch("risk.manager.mt5.symbol_info", return_value=info), \
             patch("risk.manager.mt5.symbol_info_tick", return_value=tick), \
             patch("risk.manager.mt5.order_calc_profit", side_effect=lambda side, symbol, volume, entry, end: (end-entry)*100*volume), \
             patch("risk.manager.mt5.order_send") as send:
            for balance, expected in ((100, True), (17.5, False)):
                result = manager.validate(symbol="EURUSD", action="BUY", llm_decision=decision,
                    account_info={"balance": balance, "equity": balance}, open_positions=[], market_snapshot=None,
                    calendar_events=None, trade_history=[], m5_analysis=m5, m15_analysis=analyses["M15"],
                    h1_analysis=analyses["H1"], h4_analysis=analyses["H4"])
                self.assertEqual(result.approved, expected, result.reason)
                if expected:
                    self.assertLessEqual(result.risk_budget_usd, balance*.02)
                    self.assertLessEqual(result.estimated_risk_usd, result.risk_budget_usd)
                else:
                    self.assertIn("Minimum Lot Risk", result.reason)
            send.assert_not_called()

    def test_same_thesis_still_requires_post_close_candles_and_new_live_setup(self):
        analyses, _ = live_fixture()
        m5 = analyses["M5"]
        for closed, expected in (("2032-03-01T12:26:00+00:00", False), ("2032-03-01T12:10:00+00:00", True)):
            ok, reason = RiskManager._check_same_thesis_reentry("BUY", m5, [{"close_time": closed, "net_profit": -.5, "direction": "BUY"}])
            self.assertEqual(ok, expected, reason)

    def test_model_is_told_about_real_macro_conflict_without_research_leakage(self):
        analyses, _ = live_fixture()
        analyses["M5"]["market_structure"]["reversal_watch"]["reason"] = "PRIVATE_RESEARCH_SENTINEL"
        _, prompt = PromptGenerator.generate(symbol="EURUSD", timeframe="M5", analysis_data=analyses["M5"],
            m15_analysis=analyses["M15"], h1_analysis=analyses["H1"], h4_analysis=analyses["H4"],
            account_info={}, open_positions=[], trade_history=[], calendar_events=None)
        self.assertIn("M5_LOCAL_REVERSAL_BULLISH_", prompt)
        self.assertIn("H1/H4 oppose", prompt)
        self.assertIn("never inflate confidence", prompt)
        self.assertNotIn("PRIVATE_RESEARCH_SENTINEL", prompt)

    def test_confluence_uses_real_lower_timeframe_confirmation(self):
        analyses, _ = live_fixture()
        _, factors = DecisionScoringEngine.calculate_confluence_score("BUY", *(analyses[tf] for tf in ("M5", "M15", "H1", "H4")),
                                                                       strategy_mode="LOCAL_REVERSAL")
        self.assertTrue(factors["trend_align"])
        self.assertTrue(factors["tf_agreement"])

    def test_plan_uses_invalidation_bounded_target_and_never_orders(self):
        for action in ("BUY", "SELL"):
            analyses, _ = live_fixture(action, .01)
            price = analyses["M5"]["indicators"]["current_price"]
            tick = SimpleNamespace(bid=price, ask=price+.0001)
            config = replace(self.config, max_order_deviation_points=0, fx_round_turn_cost_usd_per_lot=0,
                             fixed_execution_cost_usd=0, require_technical_target=False)
            with patch("core.trade_planner.settings", config), patch("risk.execution_costs.settings", config), \
                 patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit), \
                 patch("core.trade_planner.mt5.order_send") as send:
                plan = DeterministicTradePlanner.build("EURUSD", action, analyses["M5"], info=_instrument(), tick=tick)
            self.assertTrue(plan.valid, plan.reason)
            self.assertIn("local reversal", plan.source)
            setup = analyses["M5"]["market_structure"]["live_reversal"]
            if action == "BUY":
                self.assertLessEqual(plan.stop_loss, setup["invalidation_price"] + 1e-5)
                self.assertLessEqual(plan.take_profit, setup["target_price"] + 1e-5)
            else:
                self.assertGreaterEqual(plan.stop_loss, setup["invalidation_price"] - 1e-5)
                self.assertGreaterEqual(plan.take_profit, setup["target_price"] - 1e-5)
            send.assert_not_called()

    def test_risk_cap_never_raises_users_smaller_limit(self):
        for configured, expected in ((10, 2), (2, 2), (.5, .5)):
            self.assertEqual(live_reversal_risk_percent(replace(self.config, risk_percent=configured)), expected)


if __name__ == "__main__":
    unittest.main()

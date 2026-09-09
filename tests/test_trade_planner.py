import unittest
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5

from core.trade_planner import DeterministicTradePlanner
from core.validator import DecisionValidator


def _instrument():
    return SimpleNamespace(
        point=0.00001,
        digits=5,
        trade_stops_level=0,
        trade_freeze_level=0,
        volume_min=0.01,
        path="Forex\\Majors",
        trade_mode=mt5.SYMBOL_TRADE_MODE_FULL,
    )


def _quote():
    return SimpleNamespace(bid=1.10000, ask=1.10010)


def _analysis():
    return {
        "indicators": {"atr_14": 0.00100},
        "market_structure": {"support": 1.09500, "resistance": 1.10500},
    }


def _planner_settings():
    return SimpleNamespace(
        plan_stop_atr=2.0,
        plan_target_rr=1.8,
        plan_net_rr_buffer=0.05,
        plan_max_cost_target_extension_r=0.50,
        min_risk_reward_ratio=1.47,
        max_spread_to_stop_pct=20.0,
        max_tick_age_seconds=5.0,
        max_spread_pips=3.0,
        max_crypto_spread_bps=30.0,
        risk_percent=1.0,
        max_daily_loss_usd=0.0,
        max_daily_loss_pct=0.0,
        auto_close_loss_enabled=False,
        auto_close_loss_usd=0.25,
        max_margin_usage_pct=35.0,
        max_order_deviation_points=0,
        fx_round_turn_cost_usd_per_lot=0.0,
        crypto_round_turn_cost_usd_per_lot=0.0,
        cfd_round_turn_cost_usd_per_lot=0.0,
        fixed_execution_cost_usd=0.0,
    )


def _profit(order_type, symbol, volume, entry, exit_price):
    del symbol
    direction = 1.0 if order_type == mt5.ORDER_TYPE_BUY else -1.0
    return (exit_price - entry) * 25_000.0 * volume * direction


class DeterministicPlannerTests(unittest.TestCase):
    def test_validator_strips_model_prices_and_planner_owns_levels(self):
        ok, normalized, error = DecisionValidator.validate_decision({
            "action": "BUY",
            "confidence": 0.82,
            "entry": 9.0,
            "stop_loss": 8.0,
            "take_profit": 12.0,
            "lot_size": 50.0,
            "reasoning": "Direction only",
        })
        self.assertTrue(ok, error)
        self.assertIsNone(normalized["entry"])
        self.assertIsNone(normalized["stop_loss"])
        self.assertIsNone(normalized["take_profit"])
        self.assertNotIn("lot_size", normalized)

        isolated = _planner_settings()
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
        ):
            plan = DeterministicTradePlanner.build(
                "EURUSD", "BUY", _analysis(), info=_instrument(), tick=_quote()
            )

        self.assertTrue(plan.valid, plan.reason)
        self.assertEqual(plan.entry, _quote().ask)
        self.assertLess(plan.stop_loss, plan.entry)
        self.assertGreater(plan.take_profit, plan.entry)
        self.assertAlmostEqual(plan.planned_rr, 1.8, places=3)
        self.assertNotEqual(plan.entry, 9.0)

    def test_baseline_target_is_bounded_adjusted_for_costs(self):
        isolated = _planner_settings()
        isolated.max_order_deviation_points = 20
        isolated.fx_round_turn_cost_usd_per_lot = 7.0
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
        ):
            plan = DeterministicTradePlanner.build(
                "EURUSD", "BUY", _analysis(), info=_instrument(), tick=_quote()
            )

        self.assertTrue(plan.valid, plan.reason)
        self.assertGreaterEqual(plan.planned_rr, 1.52)
        self.assertIn("execution-adjusted target", plan.source)

    def test_excessive_cost_target_extension_is_rejected(self):
        isolated = _planner_settings()
        isolated.max_order_deviation_points = 20
        isolated.fixed_execution_cost_usd = 1.0
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
        ):
            plan = DeterministicTradePlanner.build(
                "EURUSD", "BUY", _analysis(), info=_instrument(), tick=_quote()
            )

        self.assertFalse(plan.valid)
        self.assertIn("execution-cost limit", plan.reason)

    def test_unconfirmed_nearby_structure_is_advisory_not_a_hard_cap(self):
        isolated = _planner_settings()
        isolated.require_technical_target = False
        nearby = _analysis()
        nearby["market_structure"]["resistance"] = 1.10070
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
        ):
            plan = DeterministicTradePlanner.build(
                "EURUSD", "BUY", nearby, info=_instrument(), tick=_quote()
            )

        self.assertTrue(plan.valid, plan.reason)
        self.assertAlmostEqual(plan.planned_rr, 1.8, places=3)

    def test_confirmed_structure_does_not_duplicate_optional_target_gate(self):
        isolated = _planner_settings()
        isolated.require_technical_target = False
        nearby = _analysis()
        nearby["market_structure"]["resistance"] = 1.10070
        nearby["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-08-03 10:20:00",
        }]
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
        ):
            plan = DeterministicTradePlanner.build(
                "EURUSD", "BUY", nearby, info=_instrument(), tick=_quote()
            )

        self.assertTrue(plan.valid, plan.reason)
        self.assertAlmostEqual(plan.planned_rr, 1.8, places=3)

    def test_opposite_direction_structure_does_not_cap_target(self):
        isolated = _planner_settings()
        isolated.require_technical_target = False
        nearby = _analysis()
        nearby["market_structure"]["resistance"] = 1.10070
        nearby["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BEARISH",
            "time": "2026-08-03 10:20:00",
        }]
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
        ):
            plan = DeterministicTradePlanner.build(
                "EURUSD", "BUY", nearby, info=_instrument(), tick=_quote()
            )

        self.assertTrue(plan.valid, plan.reason)
        self.assertAlmostEqual(plan.planned_rr, 1.8, places=3)

    def test_close_only_market_never_builds_a_new_entry(self):
        instrument = _instrument()
        instrument.trade_mode = mt5.SYMBOL_TRADE_MODE_CLOSEONLY
        plan = DeterministicTradePlanner.build(
            "EURUSD", "BUY", _analysis(), info=instrument, tick=_quote()
        )

        self.assertFalse(plan.valid)
        self.assertIn("close-only", plan.reason)

    def _assess(self, balance):
        isolated = _planner_settings()
        account = {
            "balance": balance,
            "equity": balance,
            "margin": 0.0,
            "margin_free": balance,
        }
        with (
            patch("core.trade_planner.settings", isolated),
            patch("risk.instruments.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("core.trade_planner.mt5.symbol_info", return_value=_instrument()),
            patch("core.trade_planner.mt5.symbol_info_tick", return_value=_quote()),
            patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit),
            patch("core.trade_planner.mt5.order_calc_margin", return_value=10.0),
        ):
            return DeterministicTradePlanner.assess_capital_fit(
                "EURUSD", _analysis(), account
            )

    def test_capital_fit_reports_an_affordable_minimum_volume(self):
        result = self._assess(balance=100.0)
        self.assertTrue(result["capital_fit"])
        self.assertEqual(result["status"], "CAPITAL FIT")
        self.assertEqual(result["risk_budget_usd"], 1.0)
        self.assertTrue(result["directions"]["BUY"]["capital_fit"])

    def test_capital_fit_reports_when_minimum_volume_exceeds_budget(self):
        result = self._assess(balance=15.0)
        self.assertFalse(result["capital_fit"])
        self.assertEqual(result["status"], "UNAFFORDABLE")
        self.assertEqual(result["risk_budget_usd"], 0.15)
        self.assertGreater(result["min_stop_risk_usd"], result["risk_budget_usd"])


if __name__ == "__main__":
    unittest.main()

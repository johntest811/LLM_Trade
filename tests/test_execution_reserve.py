import unittest
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5

from risk.execution_costs import estimate_execution_risk
from risk.manager import RiskManager


def _profit(order_type, symbol, volume, entry, exit_price):
    del symbol
    direction = 1.0 if order_type == mt5.ORDER_TYPE_BUY else -1.0
    return (exit_price - entry) * 100.0 * volume * direction


def _reserve_settings(**overrides):
    values = {
        "max_order_deviation_points": 20,
        "fx_round_turn_cost_usd_per_lot": 7.0,
        "crypto_round_turn_cost_usd_per_lot": 0.0,
        "cfd_round_turn_cost_usd_per_lot": 0.0,
        "fixed_execution_cost_usd": 0.0,
        "risk_percent": 1.0,
        "auto_close_loss_enabled": False,
        "auto_close_loss_usd": 0.0,
        "auto_close_profit_enabled": False,
        "auto_close_profit_usd": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ExecutionReserveTests(unittest.TestCase):
    def setUp(self):
        self.info = SimpleNamespace(
            point=0.01,
            path="Forex\\Majors",
            trade_calc_mode=getattr(mt5, "SYMBOL_CALC_MODE_FOREX", 0),
            volume_min=0.01,
            volume_step=0.01,
            volume_max=100.0,
        )

    def test_fx_estimate_includes_adverse_deviation_and_round_turn_cost(self):
        isolated = _reserve_settings()
        with (
            patch("risk.execution_costs.settings", isolated),
            patch("risk.execution_costs.mt5.order_calc_profit", side_effect=_profit),
        ):
            estimate = estimate_execution_risk(
                "EURUSD", "BUY", 0.01, 100.0, 99.5, self.info
            )

        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.stop_risk_usd, 0.50)
        self.assertAlmostEqual(estimate.slippage_reserve_usd, 0.20)
        self.assertAlmostEqual(estimate.configured_cost_usd, 0.07)
        self.assertAlmostEqual(estimate.total_risk_usd, 0.77)

    def test_nonfinite_broker_result_fails_closed(self):
        isolated = _reserve_settings()
        with (
            patch("risk.execution_costs.settings", isolated),
            patch("risk.execution_costs.mt5.order_calc_profit", return_value=float("nan")),
        ):
            estimate = estimate_execution_risk(
                "EURUSD", "BUY", 0.01, 100.0, 99.5, self.info
            )

        self.assertIsNone(estimate)

    def test_lot_sizing_uses_the_same_execution_adjusted_budget(self):
        isolated = _reserve_settings()
        manager = RiskManager()
        decision = {
            "action": "BUY",
            "stop_loss": 99.5,
            "take_profit": 102.0,
            "risk_percentage": 1.0,
        }
        tick = SimpleNamespace(ask=100.0, bid=99.9)
        with (
            patch("risk.manager.settings", isolated),
            patch("risk.execution_costs.settings", isolated),
            patch("risk.manager.mt5.symbol_info", return_value=self.info),
            patch("risk.manager.mt5.symbol_info_tick", return_value=tick),
            patch("risk.manager.mt5.order_calc_profit", side_effect=_profit),
        ):
            lot, reason, sizing = manager._compute_lot_size(
                decision, {"balance": 100.0}, None, "EURUSD", []
            )

        self.assertEqual(reason, "")
        self.assertEqual(lot, 0.01)
        self.assertAlmostEqual(sizing["stop_risk_usd"], 0.50)
        self.assertAlmostEqual(sizing["slippage_reserve_usd"], 0.20)
        self.assertAlmostEqual(sizing["execution_cost_usd"], 0.07)
        self.assertAlmostEqual(sizing["risk_usd"], 0.77)
        self.assertEqual(sizing["risk_budget_usd"], 1.0)


if __name__ == "__main__":
    unittest.main()

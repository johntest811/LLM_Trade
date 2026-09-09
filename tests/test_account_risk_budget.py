import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from app_config.settings import settings
from risk.budget import risk_capital, entry_risk_budget, daily_loss_limit
from risk.manager import RiskManager
from core.trade_planner import DeterministicTradePlanner
from tests.test_trade_planner import _analysis, _instrument, _quote, _profit
from ui.state import DashboardState


class AccountRiskBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(settings, risk_percent=1, max_daily_loss_usd=0,
                              max_daily_loss_pct=0, auto_close_loss_enabled=False)

    def test_scales_across_balances_without_compounding_floating_gains(self):
        for balance in (1, 17.5, 100, 10000, 1000000):
            for fraction in (.5, 1, 1.5):
                with self.subTest(balance=balance, equity_fraction=fraction):
                    account = {"balance": balance, "equity": balance*fraction}
                    self.assertAlmostEqual(entry_risk_budget(account, self.config), min(balance, balance*fraction)*.01)

    def test_invalid_or_missing_equity_is_not_replaced_with_balance(self):
        for field in ("balance", "equity"):
            for value in (None, 0, -1, "bad", float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    account = {"balance": 100, "equity": 100, field: value}
                    self.assertEqual(risk_capital(account), 0)
                    self.assertEqual(entry_risk_budget(account, self.config), 0)
            self.assertEqual(entry_risk_budget({"balance" if field == "equity" else "equity": 100}, self.config), 0)

    def test_dollar_and_remaining_daily_caps_are_never_raised(self):
        config = replace(self.config, auto_close_loss_enabled=True, auto_close_loss_usd=2,
                         max_daily_loss_usd=2.5, max_daily_loss_pct=3)
        self.assertEqual(entry_risk_budget({"balance": 10000, "equity": 10000}, config), 2)
        self.assertAlmostEqual(entry_risk_budget({"balance": 10000, "equity": 10000}, config, daily_loss=2.2), .3)
        self.assertEqual(entry_risk_budget({"balance": 17.5, "equity": 17.5}, config, daily_loss=1.06), 0)
        self.assertAlmostEqual(daily_loss_limit({"balance": 100, "equity": 50}, config), 1.5)

    def test_final_sizing_uses_equity_and_rejects_forced_oversizing(self):
        manager = RiskManager()
        decision = {"action": "BUY", "stop_loss": 99.5, "take_profit": 102}
        info = SimpleNamespace(volume_min=.01, volume_step=.01, volume_max=100, point=.001, path="Forex\\Majors")
        config = replace(self.config, max_order_deviation_points=0, fx_round_turn_cost_usd_per_lot=0, fixed_execution_cost_usd=0)
        with patch("risk.manager.settings", config), patch("risk.execution_costs.settings", config), \
             patch("risk.manager.mt5.symbol_info", return_value=info), \
             patch("risk.manager.mt5.symbol_info_tick", return_value=SimpleNamespace(ask=100, bid=100)), \
             patch("risk.manager.mt5.order_calc_profit", side_effect=lambda kind, symbol, volume, entry, end: (end-entry)*100*volume):
            lot, reason, data = manager._compute_lot_size(decision, {"balance": 1000, "equity": 50}, None, "TEST", [])
            self.assertIsNotNone(lot, reason)
            self.assertEqual(data["risk_budget_usd"], .5)
            lot, reason, _ = manager._compute_lot_size(decision, {"balance": 1000, "equity": 17.5}, None, "TEST", [], force_minimum_lot=True)
            self.assertIsNone(lot)
            self.assertIn("Minimum Lot Risk", reason)

    def test_authorized_micro_profile_preserves_dollar_daily_stop(self):
        config = replace(self.config, risk_percent=10, max_portfolio_risk_pct=12,
                         max_daily_loss_usd=2.5, max_daily_loss_pct=0,
                         auto_close_loss_enabled=True, auto_close_loss_usd=2)
        account = {"balance": 17.5, "equity": 17.5}
        self.assertAlmostEqual(daily_loss_limit(account, config), 2.5)
        self.assertAlmostEqual(entry_risk_budget(account, config), 1.75)
        self.assertAlmostEqual(entry_risk_budget(account, config, daily_loss=1.06), 1.44)
        for loss in (2.5, 3.0):
            with self.subTest(loss=loss):
                self.assertEqual(entry_risk_budget(account, config, daily_loss=loss), 0)
        self.assertEqual(entry_risk_budget({"balance": 10000, "equity": 10000}, config), 2)

    def test_authorized_micro_profile_can_size_within_remaining_daily_capacity(self):
        # Sizing-only regression using the recorded AUDUSD prices. Passing
        # affordability is NOT entry approval: continuation checks still apply.
        manager = RiskManager()
        manager._daily_loss_usd = 1.06
        config = replace(self.config, risk_percent=10, max_portfolio_risk_pct=12,
                         max_daily_loss_usd=2.5, max_daily_loss_pct=0,
                         auto_close_loss_enabled=True, auto_close_loss_usd=2,
                         auto_close_profit_enabled=False, max_order_deviation_points=10,
                         fx_round_turn_cost_usd_per_lot=0, fixed_execution_cost_usd=0)
        info = SimpleNamespace(volume_min=.01, volume_step=.01, volume_max=100,
                               point=.00001, path="Forex\\Majors")
        decision = {"action": "BUY", "stop_loss": .72240, "take_profit": .72396}
        with patch("risk.manager.settings", config), patch("risk.execution_costs.settings", config), \
             patch("risk.manager.mt5.symbol_info", return_value=info), \
             patch("risk.manager.mt5.symbol_info_tick", return_value=SimpleNamespace(ask=.72293, bid=.72283)), \
             patch("risk.manager.mt5.order_calc_profit", side_effect=lambda kind, symbol, volume, entry, end: (end-entry)*100000*volume):
            lot, reason, data = manager._compute_lot_size(
                decision, {"balance": 17.5, "equity": 17.5}, None, "AUDUSD", [])
        self.assertIsNotNone(lot, reason)
        self.assertAlmostEqual(lot, .02)
        self.assertAlmostEqual(data["risk_budget_usd"], 1.44)
        self.assertAlmostEqual(data["risk_usd"], 1.26)
        self.assertLessEqual(data["risk_usd"], data["risk_budget_usd"])

    def test_authorized_portfolio_cap_does_not_bypass_remaining_daily_capacity(self):
        manager = RiskManager()
        config = replace(self.config, max_portfolio_risk_pct=12,
                         max_daily_loss_usd=2.5, max_daily_loss_pct=0)
        account = {"balance": 17.5, "equity": 17.5}
        with patch("risk.manager.settings", config):
            self.assertTrue(manager._check_portfolio_risk(account, [], 2.1)[0])
            self.assertFalse(manager._check_portfolio_risk(account, [], 2.11)[0])
            manager._daily_loss_usd = 1.06
            self.assertTrue(manager._check_portfolio_risk(account, [], 1.26)[0])
            ok, reason = manager._check_portfolio_risk(account, [], 1.45)
            self.assertFalse(ok)
            self.assertIn("Remaining Daily Risk", reason)

    def test_discovery_and_sizing_share_the_budget_basis(self):
        account = {"balance": 1000, "equity": 50, "margin": 0, "margin_free": 50}
        config = replace(self.config, max_order_deviation_points=0, fx_round_turn_cost_usd_per_lot=0, fixed_execution_cost_usd=0)
        with patch("core.trade_planner.settings", config), patch("risk.execution_costs.settings", config), \
             patch("core.trade_planner.mt5.symbol_info", return_value=_instrument()), \
             patch("core.trade_planner.mt5.symbol_info_tick", return_value=_quote()), \
             patch("core.trade_planner.mt5.order_calc_profit", side_effect=_profit), \
             patch("core.trade_planner.mt5.order_calc_margin", return_value=1):
            result = DeterministicTradePlanner.assess_capital_fit("EURUSD", _analysis(), account)
        self.assertEqual(result["risk_budget_usd"], entry_risk_budget(account, config))
        self.assertEqual(result["risk_capital"], 50)

    def test_open_plus_proposed_risk_cannot_overcommit_daily_budget(self):
        manager = RiskManager()
        manager._daily_loss_usd = 2
        config = replace(self.config, max_daily_loss_usd=2.5, max_portfolio_risk_pct=3)
        position = {"symbol": "TEST", "type": 0, "volume": .01, "price_open": 100, "sl": 99}
        with patch("risk.manager.settings", config), patch("risk.manager.mt5.order_calc_profit", return_value=-.4):
            ok, reason = manager._check_portfolio_risk({"balance": 100, "equity": 100}, [position], .2)
        self.assertFalse(ok)
        self.assertIn("Remaining Daily Risk", reason)

    def test_dashboard_portfolio_percentage_uses_the_same_capital_basis(self):
        state = DashboardState()
        state.update_account({"balance": 100, "equity": 50})
        state.update_positions([{"ticket": 1, "symbol": "TEST", "type": 0, "risk_to_sl_usd": 1}])
        self.assertEqual(state.account.portfolio_risk_pct, 2)

    def test_invalid_margin_never_authorizes_risk(self):
        manager = RiskManager()
        account = {"balance": 100, "equity": 100, "margin": 0,
                   "margin_free": 100, "margin_level": 0}
        for field, method in (("margin_free", manager._check_free_margin),
                              ("margin_level", manager._check_margin_level),
                              ("margin", manager._check_margin_usage),
                              ("equity", manager._check_margin_usage)):
            for value in (None, "bad", float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    malformed = dict(account, **{field: value})
                    self.assertFalse(method(malformed)[0])
                    if field != "margin_level":
                        with patch("risk.manager.mt5.symbol_info") as broker:
                            self.assertFalse(manager._check_margin_for_lot(malformed, .01, "TEST")[0])
                            broker.assert_not_called()
        self.assertTrue(manager._check_margin_level(account)[0])
        self.assertFalse(manager._check_margin_level(dict(account, margin=5))[0])

    def test_discovery_reports_bad_margin_without_crashing(self):
        with patch("core.trade_planner.mt5.symbol_info", return_value=_instrument()), \
             patch("core.trade_planner.mt5.symbol_info_tick", return_value=_quote()):
            for field in ("margin", "margin_free"):
                for value in (None, "bad", float("nan"), float("inf")):
                    with self.subTest(field=field, value=value):
                        account = {"balance": 100, "equity": 100, "margin": 0, "margin_free": 100, field: value}
                        result = DeterministicTradePlanner.assess_capital_fit("EURUSD", _analysis(), account)
                        self.assertFalse(result["capital_fit"])
                        self.assertEqual(result["status"], "DATA ERROR")


if __name__ == "__main__":
    unittest.main()

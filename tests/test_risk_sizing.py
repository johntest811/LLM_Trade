import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app_config.settings import settings
from risk.manager import RiskManager, _effective_losing_streak


class AdaptiveLossStreakTests(unittest.TestCase):
    def setUp(self):
        settings_patch = patch(
            "risk.manager.settings",
            replace(settings, loss_streak_pause_hours=24),
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)

    @staticmethod
    def _loss(closed_at):
        return {"close_time": closed_at, "net_profit": -0.20}

    def test_single_loss_penalty_expires(self):
        history = [self._loss((self.now - timedelta(hours=25)).isoformat())]

        self.assertEqual(_effective_losing_streak(history, self.now), 0)

    def test_two_loss_penalty_expires_from_latest_close(self):
        history = [
            self._loss((self.now - timedelta(hours=25)).isoformat()),
            self._loss((self.now - timedelta(hours=30)).isoformat()),
        ]

        self.assertEqual(_effective_losing_streak(history, self.now), 0)

    def test_recent_single_loss_penalty_remains_active(self):
        history = [self._loss((self.now - timedelta(hours=2)).isoformat())]

        self.assertEqual(_effective_losing_streak(history, self.now), 1)

    def test_scratch_trade_ends_consecutive_loss_streak(self):
        history = [
            {"close_time": self.now.isoformat(), "net_profit": 0.0},
            self._loss((self.now - timedelta(hours=1)).isoformat()),
        ]

        self.assertEqual(_effective_losing_streak(history, self.now), 0)

    def test_unordered_history_uses_latest_close_time(self):
        history = [
            self._loss((self.now - timedelta(hours=1)).isoformat()),
            {"close_time": self.now.isoformat(), "net_profit": 0.10},
        ]

        self.assertEqual(_effective_losing_streak(history, self.now), 0)

    def test_operator_reset_excludes_prior_losses_without_rewriting_history(self):
        history = [
            self._loss((self.now - timedelta(minutes=20)).isoformat()),
            self._loss((self.now - timedelta(minutes=30)).isoformat()),
            self._loss((self.now - timedelta(minutes=40)).isoformat()),
        ]
        reset_at = self.now - timedelta(minutes=10)

        self.assertEqual(
            _effective_losing_streak(history, self.now, reset_at),
            0,
        )
        self.assertEqual(len(history), 3)

    def test_post_reset_loss_starts_a_new_streak(self):
        reset_at = self.now - timedelta(minutes=30)
        history = [
            self._loss((self.now - timedelta(minutes=5)).isoformat()),
            self._loss((self.now - timedelta(hours=1)).isoformat()),
            self._loss((self.now - timedelta(hours=2)).isoformat()),
        ]

        self.assertEqual(
            _effective_losing_streak(history, self.now, reset_at),
            1,
        )

    def test_risk_manager_restores_persisted_streak_marker(self):
        manager = RiskManager()
        marker = self.now - timedelta(minutes=15)

        self.assertTrue(manager.restore_losing_streak_reset(marker.isoformat()))
        self.assertEqual(manager._loss_streak_reset_after_utc, marker)


class RiskSizingTests(unittest.TestCase):
    def setUp(self):
        settings_patch = patch(
            "risk.manager.settings",
            replace(
                settings,
                auto_close_loss_enabled=True,
                auto_close_loss_usd=0.50,
            ),
        )
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.manager = RiskManager()
        self.info = SimpleNamespace(volume_min=0.01, volume_step=0.01, volume_max=100.0)
        self.tick = SimpleNamespace(ask=100.0, bid=99.9)
        self.decision = {
            "action": "BUY", "stop_loss": 99.0, "take_profit": 102.0,
            "risk_percentage": 1.0,
        }

    def _profit(self, order_type, symbol, volume, entry, exit_price):
        return (exit_price - entry) * 100 * volume

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_minimum_lot_is_rejected_instead_of_rounded_up(self, info, tick, calc):
        info.return_value, tick.return_value, calc.side_effect = self.info, self.tick, self._profit
        lot, reason, sizing = self.manager._compute_lot_size(
            self.decision, {"balance": 15.0, "equity": 15.0}, None, "TEST", []
        )
        self.assertIsNone(lot)
        self.assertIn("Minimum Lot Risk", reason)
        self.assertEqual(sizing["risk_usd"], 0.0)

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_affordable_lot_returns_broker_calculated_metrics(self, info, tick, calc):
        info.return_value, tick.return_value, calc.side_effect = self.info, self.tick, self._profit
        decision = dict(self.decision, stop_loss=99.5, take_profit=101.0)
        lot, reason, sizing = self.manager._compute_lot_size(
            decision, {"balance": 1000.0, "equity": 1000.0}, None, "TEST", []
        )
        self.assertEqual(reason, "")
        self.assertEqual(lot, 0.01)
        self.assertAlmostEqual(sizing["risk_usd"], 0.5)
        self.assertAlmostEqual(sizing["rr"], 2.0)

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_model_one_percent_does_not_override_configured_budget(self, info, tick, calc):
        info.return_value, tick.return_value, calc.side_effect = self.info, self.tick, self._profit
        configured = replace(
            settings,
            risk_percent=10.0,
            auto_close_loss_enabled=True,
            auto_close_loss_usd=2.00,
            max_daily_loss_pct=0.0,  # Isolate the configured per-entry budget.
        )
        with patch("risk.manager.settings", configured):
            lot, reason, sizing = self.manager._compute_lot_size(
                self.decision, {"balance": 15.17, "equity": 15.17}, None, "TEST", []
            )

        self.assertEqual(reason, "")
        self.assertEqual(lot, 0.01)
        # The $2 emergency ceiling must not replace the tighter configured
        # 10% account-risk budget on a micro balance.
        self.assertAlmostEqual(sizing["risk_budget_usd"], 1.517)

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_loss_streak_floor_never_increases_a_smaller_configured_cap(self, info, tick, calc):
        info.return_value, tick.return_value, calc.side_effect = self.info, self.tick, self._profit
        configured = replace(settings, risk_percent=.01, auto_close_loss_enabled=False, loss_streak_pause_hours=24)
        decision = dict(self.decision, stop_loss=99.5, take_profit=101.0, _strategy={"mode": "LOCAL_REVERSAL"})
        with patch("risk.manager.settings", configured):
            lot, reason, sizing = self.manager._compute_lot_size(
                decision, {"balance": 10000.0, "equity": 10000.0}, None, "TEST", [], losing_streak=2)
        self.assertIsNotNone(lot, reason)
        self.assertAlmostEqual(sizing["risk_budget_usd"], 1.0)
        self.assertLessEqual(sizing["risk_usd"], 1.0)

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_manual_sizing_uses_configured_cap_not_streak_reduction(self, info, tick, calc):
        info.return_value, tick.return_value, calc.side_effect = self.info, self.tick, self._profit
        configured = replace(
            settings,
            risk_percent=6.0,
            auto_close_loss_enabled=False,
            auto_close_loss_usd=0.0,
            loss_streak_pause_hours=24,
            max_daily_loss_pct=0.0,  # Daily capacity is tested separately.
        )
        decision = dict(self.decision, stop_loss=99.5)
        with patch("risk.manager.settings", configured):
            auto_lot, _, _ = self.manager._compute_lot_size(
                decision,
                {"balance": 15.0, "equity": 15.0},
                None,
                "TEST",
                [],
                losing_streak=1,
            )
            manual_lot, reason, sizing = self.manager._compute_lot_size(
                decision,
                {"balance": 15.0, "equity": 15.0},
                None,
                "TEST",
                [],
                losing_streak=1,
                apply_streak_scaling=False,
            )

        self.assertIsNone(auto_lot)
        self.assertEqual(reason, "")
        self.assertEqual(manual_lot, 0.01)
        self.assertAlmostEqual(sizing["risk_budget_usd"], 0.90)

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_zero_hour_streak_pause_disables_automatic_risk_scaling(
        self, info, tick, calc
    ):
        info.return_value, tick.return_value, calc.side_effect = (
            self.info,
            self.tick,
            self._profit,
        )
        configured = replace(
            settings,
            risk_percent=10.0,
            auto_close_loss_enabled=False,
            auto_close_loss_usd=0.0,
            loss_streak_pause_hours=0,
            max_daily_loss_pct=0.0,
        )
        decision = dict(self.decision, stop_loss=99.5)
        with patch("risk.manager.settings", configured):
            lot, reason, sizing = self.manager._compute_lot_size(
                decision,
                {"balance": 10.0, "equity": 10.0},
                None,
                "TEST",
                [],
                losing_streak=2,
            )

        self.assertEqual(reason, "")
        self.assertEqual(lot, 0.02)
        self.assertAlmostEqual(sizing["risk_budget_usd"], 1.0)

    @patch("risk.manager.mt5.order_calc_profit")
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_force_minimum_lot_cannot_exceed_project_risk_budget(self, info, tick, calc):
        info.return_value, tick.return_value, calc.side_effect = self.info, self.tick, self._profit
        configured = replace(
            settings,
            risk_percent=1.0,
            auto_close_loss_enabled=False,
            auto_close_loss_usd=0.0,
        )
        with patch("risk.manager.settings", configured):
            lot, reason, sizing = self.manager._compute_lot_size(
                self.decision,
                {"balance": 15.0, "equity": 15.0},
                None,
                "TEST",
                [],
                force_minimum_lot=True,
                enforce_profit_objective=False,
            )

        self.assertIsNone(lot)
        self.assertIn("Minimum Lot Risk", reason)

    @patch("risk.manager.mt5.order_calc_margin", return_value=3.54)
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_margin_uses_sell_side_broker_calculation(self, info, tick, calc):
        info.return_value = SimpleNamespace(ask=100.0, bid=99.9)
        tick.return_value = self.tick
        ok, reason = self.manager._check_margin_for_lot(
            {"margin_free": 15.17, "equity": 15.17, "margin": 0.0},
            0.01,
            "TEST",
            action="SELL",
        )

        self.assertTrue(ok, reason)
        calc.assert_called_once_with(1, "TEST", 0.01, 99.9)

    @patch("risk.manager.mt5.order_calc_margin", return_value=11.00)
    @patch("risk.manager.mt5.symbol_info_tick")
    @patch("risk.manager.mt5.symbol_info")
    def test_projected_margin_limit_rejects_unaffordable_symbol(self, info, tick, calc):
        info.return_value = SimpleNamespace(ask=100.0, bid=99.9)
        tick.return_value = self.tick
        ok, reason = self.manager._check_margin_for_lot(
            {"margin_free": 15.17, "equity": 15.17, "margin": 0.0},
            0.01,
            "TEST",
            action="BUY",
        )

        self.assertFalse(ok)
        self.assertIn("Projected Margin", reason)


class ManualOverrideRiskTests(unittest.TestCase):
    @staticmethod
    def _manager_with_hard_checks(daily_result=(True, "")):
        manager = RiskManager()
        for name in (
            "_check_weekend",
            "_check_spread",
            "_check_max_positions",
            "_check_duplicate",
            "_check_drawdown",
            "_check_loss_cooldown",
            "_check_free_margin",
            "_check_margin_level",
            "_check_margin_usage",
            "_check_portfolio_risk",
            "_check_margin_for_lot",
        ):
            setattr(manager, name, MagicMock(return_value=(True, "")))
        manager._check_daily_loss = MagicMock(return_value=daily_result)
        manager._compute_lot_size = MagicMock(return_value=(
            0.01,
            "",
            {
                "risk_usd": 0.60,
                "reward_usd": 0.75,
                "risk_pct": 4.0,
                "rr": 1.8,
                "risk_budget_usd": 0.90,
            },
        ))
        return manager

    def test_manual_override_bypasses_model_gates_but_keeps_sizing(self):
        manager = self._manager_with_hard_checks()

        result = manager.validate_manual_override(
            symbol="USDJPY",
            action="BUY",
            decision={"action": "BUY", "stop_loss": 149.9, "take_profit": 150.2},
            account_info={"balance": 15.0, "equity": 15.0},
            open_positions=[],
            market_snapshot=object(),
            trade_history=[],
        )

        self.assertTrue(result.approved, result.reason)
        self.assertEqual(result.adjusted_lot, 0.01)
        self.assertEqual(result.planned_rr, 1.8)
        self.assertFalse(
            manager._compute_lot_size.call_args.kwargs["apply_streak_scaling"]
        )
        self.assertTrue(
            manager._compute_lot_size.call_args.kwargs["force_minimum_lot"]
        )
        self.assertFalse(
            manager._compute_lot_size.call_args.kwargs["enforce_profit_objective"]
        )

    def test_manual_override_preserves_daily_loss_and_drawdown_locks(self):
        manager = self._manager_with_hard_checks(
            (False, "REJECTED [Daily Loss Limit]: locked")
        )
        manager._check_drawdown.return_value = (
            False,
            "REJECTED [Drawdown Limit]: locked",
        )

        result = manager.validate_manual_override(
            symbol="USDJPY",
            action="BUY",
            decision={"action": "BUY", "stop_loss": 149.9, "take_profit": 150.2},
            account_info={"balance": 15.0, "equity": 15.0},
            open_positions=[],
            market_snapshot=object(),
            trade_history=[],
        )

        self.assertFalse(result.approved)
        self.assertIn("Daily Loss Limit", result.reason)
        manager._check_daily_loss.assert_called_once()

    def test_manual_override_bypasses_loss_cooldown_only(self):
        manager = self._manager_with_hard_checks()
        manager._check_loss_cooldown = MagicMock(return_value=(
            False,
            "REJECTED [Loss Cooldown]: Wait 53.2 more minutes.",
        ))

        result = manager.validate_manual_override(
            symbol="USDCAD",
            action="SELL",
            decision={"action": "SELL", "stop_loss": 1.404, "take_profit": 1.401},
            account_info={"balance": 15.0, "equity": 15.0},
            open_positions=[],
            market_snapshot=object(),
            trade_history=[],
        )

        self.assertTrue(result.approved, result.reason)
        manager._check_loss_cooldown.assert_not_called()

    def test_manual_override_stops_at_the_first_hard_policy_failure(self):
        manager = self._manager_with_hard_checks()
        policy_checks = (
            "_check_weekend",
            "_check_spread",
            "_check_max_positions",
            "_check_duplicate",
            "_check_daily_loss",
            "_check_drawdown",
            "_check_loss_cooldown",
            "_check_free_margin",
            "_check_margin_level",
            "_check_margin_usage",
            "_check_portfolio_risk",
            "_check_margin_for_lot",
        )
        for name in policy_checks:
            getattr(manager, name).return_value = (False, f"{name} blocked")

        result = manager.validate_manual_override(
            symbol="CADJPY",
            action="BUY",
            decision={"action": "BUY", "stop_loss": 115.8, "take_profit": 116.2},
            account_info={"balance": 13.25, "equity": 13.25},
            open_positions=[{"symbol": "CADJPY"}],
            market_snapshot=object(),
            trade_history=[{"net_profit": -1.0}] * 4,
        )

        self.assertFalse(result.approved)
        self.assertIn("_check_spread", result.reason)
        manager._compute_lot_size.assert_not_called()


class DrawdownEntryLockTests(unittest.TestCase):
    def setUp(self):
        self.manager = RiskManager()
        self.manager._peak_balance = 15.81
        self.account = {"equity": 13.34}

    def test_disabled_drawdown_lock_allows_automatic_validation_to_continue(self):
        with patch(
            "risk.manager.settings",
            replace(
                settings,
                drawdown_entry_lock_enabled=False,
                max_drawdown_pct=8.0,
            ),
        ):
            ok, reason = self.manager._check_drawdown(self.account)

        self.assertTrue(ok, reason)

    def test_enabled_drawdown_lock_still_reports_peak_decline(self):
        with patch(
            "risk.manager.settings",
            replace(
                settings,
                drawdown_entry_lock_enabled=True,
                max_drawdown_pct=8.0,
            ),
        ):
            ok, reason = self.manager._check_drawdown(self.account)

        self.assertFalse(ok)
        self.assertIn("15.62%", reason)
        self.assertIn("$15.81", reason)


class LossCooldownRuleTests(unittest.TestCase):
    def test_automatic_cooldown_rule_remains_active(self):
        manager = RiskManager()
        now = datetime.now(timezone.utc)
        manager._last_loss_time = now - timedelta(minutes=7)

        with patch(
            "risk.manager.settings",
            replace(settings, loss_cooldown_minutes=60),
        ):
            ok, reason = manager._check_loss_cooldown(now)

        self.assertFalse(ok)
        self.assertIn("Loss Cooldown", reason)

    def test_symbol_history_prevents_another_markets_loss_from_blocking(self):
        manager = RiskManager()
        now = datetime.now(timezone.utc)
        manager._last_loss_time = now - timedelta(minutes=2)

        with patch(
            "risk.manager.settings",
            replace(settings, loss_cooldown_minutes=15),
        ):
            ok, reason = manager._check_loss_cooldown(
                now,
                [{
                    "direction": "BUY",
                    "net_profit": 0.20,
                    "close_time": (now - timedelta(minutes=1)).isoformat(),
                }],
            )

        self.assertTrue(ok, reason)

    def test_recent_loss_in_this_symbol_is_blocked(self):
        manager = RiskManager()
        now = datetime.now(timezone.utc)

        with patch(
            "risk.manager.settings",
            replace(settings, loss_cooldown_minutes=15),
        ):
            ok, reason = manager._check_loss_cooldown(
                now,
                [{
                    "direction": "SELL",
                    "net_profit": -0.40,
                    "close_time": (now - timedelta(minutes=3)).isoformat(),
                }],
            )

        self.assertFalse(ok)
        self.assertIn("This market's last loss", reason)


if __name__ == "__main__":
    unittest.main()

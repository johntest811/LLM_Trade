import asyncio
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from core.engine import TradingEngine
from risk.instruments import downside_risk_usd, validate_spread
from risk.manager import RiskManager


class QuoteValidationTests(unittest.TestCase):
    def setUp(self):
        self.info = SimpleNamespace(point=0.00001, digits=5, path="Forex\\Majors")

    def test_valid_quote_is_accepted(self):
        ok, reason, _ = validate_spread(
            "EURUSD", self.info, SimpleNamespace(bid=1.10000, ask=1.10001)
        )

        self.assertTrue(ok, reason)

    def test_nonpositive_bid_or_ask_is_rejected(self):
        for bid, ask in ((0.0, 1.1), (1.1, 0.0), (-1.0, 1.1)):
            with self.subTest(bid=bid, ask=ask):
                ok, reason, _ = validate_spread(
                    "EURUSD", self.info, SimpleNamespace(bid=bid, ask=ask)
                )
                self.assertFalse(ok)
                self.assertIn("positive", reason)

    def test_nonfinite_quote_is_rejected(self):
        ok, reason, _ = validate_spread(
            "EURUSD", self.info, SimpleNamespace(bid=float("nan"), ask=1.1)
        )

        self.assertFalse(ok)
        self.assertIn("finite", reason)

    def test_crossed_quote_is_rejected(self):
        ok, reason, _ = validate_spread(
            "EURUSD", self.info, SimpleNamespace(bid=1.10002, ask=1.10001)
        )

        self.assertFalse(ok)
        self.assertIn("Crossed quote", reason)

    def test_exact_spread_to_stop_boundary_is_not_rejected_by_float_noise(self):
        with patch(
            "risk.instruments.settings",
            SimpleNamespace(
                max_crypto_spread_bps=30.0,
                max_spread_pips=3.0,
                max_spread_to_stop_pct=20.0,
            ),
        ):
            ok, reason, metrics = validate_spread(
                "EURUSD",
                self.info,
                SimpleNamespace(bid=1.10000, ask=1.10010),
                entry=1.10010,
                stop_loss=1.09960,
            )

        self.assertTrue(ok, reason)
        self.assertAlmostEqual(metrics["spread_to_stop_pct"], 20.0)


class DownsideRiskTests(unittest.TestCase):
    def setUp(self):
        self.manager = RiskManager()
        self.position = {
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.01,
            "price_open": 1.1000,
            "sl": 1.1010,
        }

    def test_profit_estimate_is_converted_to_downside_only(self):
        self.assertEqual(downside_risk_usd(5.0), 0.0)
        self.assertEqual(downside_risk_usd(-5.0), 5.0)
        with self.assertRaises(ValueError):
            downside_risk_usd(None)
        with self.assertRaises(ValueError):
            downside_risk_usd(float("nan"))

    @patch("risk.manager.mt5.order_calc_profit", return_value=5.0)
    def test_profitable_protected_stop_adds_no_portfolio_downside(self, _calc):
        with patch(
            "risk.manager.settings", SimpleNamespace(max_portfolio_risk_pct=3.0)
        ):
            ok, reason = self.manager._check_portfolio_risk(
                {"balance": 100.0}, [self.position], proposed_risk_usd=2.0
            )

        self.assertTrue(ok, reason)


    @patch("risk.manager.mt5.order_calc_profit", return_value=-2.0)
    def test_losing_stop_is_included_in_portfolio_downside(self, _calc):
        with patch(
            "risk.manager.settings", SimpleNamespace(max_portfolio_risk_pct=3.0)
        ):
            ok, reason = self.manager._check_portfolio_risk(
                {"balance": 100.0}, [self.position], proposed_risk_usd=2.0
            )

        self.assertFalse(ok)
        self.assertIn("$4.00", reason)

    @patch("risk.manager.mt5.order_calc_profit", return_value=None)
    def test_unknown_open_stop_risk_fails_closed(self, _calc):
        with patch(
            "risk.manager.settings", SimpleNamespace(max_portfolio_risk_pct=3.0)
        ):
            ok, reason = self.manager._check_portfolio_risk(
                {"balance": 100.0}, [self.position], proposed_risk_usd=1.0
            )

        self.assertFalse(ok)
        self.assertIn("could not calculate", reason)

    @patch("risk.manager.mt5.order_calc_profit", return_value=float("nan"))
    def test_nonfinite_open_stop_risk_fails_closed(self, _calc):
        with patch(
            "risk.manager.settings", SimpleNamespace(max_portfolio_risk_pct=3.0)
        ):
            ok, reason = self.manager._check_portfolio_risk(
                {"balance": 100.0}, [self.position], proposed_risk_usd=1.0
            )

        self.assertFalse(ok)
        self.assertIn("could not calculate", reason)


class RiskRewardBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.manager = RiskManager()

    def test_exact_147_boundary_is_not_rejected_by_float_noise(self):
        decision = {
            "action": "BUY",
            "stop_loss": 1.00000,
            "take_profit": 1.24700,
        }
        with (
            patch(
                "risk.manager.settings",
                SimpleNamespace(min_risk_reward_ratio=1.47),
            ),
            patch(
                "risk.manager.mt5.symbol_info_tick",
                return_value=SimpleNamespace(bid=1.09990, ask=1.10000),
            ),
        ):
            ok, reason = self.manager._check_risk_reward(
                decision, market_snapshot=None, symbol="EURUSD"
            )

        self.assertTrue(ok, reason)

    def test_materially_below_147_boundary_is_rejected(self):
        decision = {
            "action": "BUY",
            "stop_loss": 1.00000,
            "take_profit": 1.24600,
        }
        with (
            patch(
                "risk.manager.settings",
                SimpleNamespace(min_risk_reward_ratio=1.47),
            ),
            patch(
                "risk.manager.mt5.symbol_info_tick",
                return_value=SimpleNamespace(bid=1.09990, ask=1.10000),
            ),
        ):
            ok, reason = self.manager._check_risk_reward(
                decision, market_snapshot=None, symbol="EURUSD"
            )

        self.assertFalse(ok)
        self.assertIn("below the minimum required 1.47", reason)


class DailyLossStatusTests(unittest.TestCase):
    def test_dashboard_status_uses_the_stricter_usd_or_percent_limit(self):
        manager = RiskManager()
        manager._daily_loss_usd = 0.91
        manager._daily_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fake_settings = SimpleNamespace(
            max_daily_loss_usd=1.0,
            max_daily_loss_pct=6.0,
        )
        with patch("risk.manager.settings", fake_settings):
            ok, detail = manager.daily_loss_status({"balance": 15.0})

        self.assertFalse(ok)
        self.assertIn("$0.91 / $0.90", detail)

    def test_delayed_broker_snapshot_cannot_erase_observed_loss(self):
        manager = RiskManager()
        manager._daily_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        manager._daily_loss_usd = 0.70
        observed_at = datetime.now(timezone.utc)
        manager._last_loss_time = observed_at

        manager.synchronize_closed_trades([], balance=14.78)

        self.assertEqual(manager.daily_loss_usd, 0.70)
        self.assertEqual(manager._last_loss_time, observed_at)

    def test_external_closed_loss_is_not_counted_as_strategy_daily_loss(self):
        manager = RiskManager()

        manager.record_trade_closed(-0.83, 14.78, strategy_owned=False)

        self.assertEqual(manager.daily_loss_usd, 0.0)

    def test_operator_reset_ignores_earlier_losses_and_counts_new_losses(self):
        manager = RiskManager()
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        marker = start + timedelta(microseconds=1)
        manager._last_loss_time = start
        manager.reset_daily_loss_for_today(marker)

        self.assertIsNone(manager._last_loss_time)

        post_reset_loss = marker + timedelta(microseconds=1)
        manager.synchronize_closed_trades(
            [
                {"close_time": start.isoformat(), "net_profit": -1.28},
                {
                    "close_time": post_reset_loss.isoformat(),
                    "net_profit": -0.25,
                },
            ],
            balance=13.96,
        )

        self.assertEqual(manager.daily_loss_usd, 0.25)
        self.assertEqual(manager._last_loss_time, post_reset_loss)

    def test_operator_reset_keeps_pre_reset_loss_out_of_global_cooldown(self):
        manager = RiskManager()
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        marker = start + timedelta(minutes=10)
        old_loss = start + timedelta(minutes=5)
        manager.reset_daily_loss_for_today(marker)

        manager.synchronize_closed_trades(
            [{"close_time": old_loss.isoformat(), "net_profit": -1.28}],
            balance=13.96,
        )

        self.assertIsNone(manager._last_loss_time)

    def test_cooldown_only_reset_preserves_daily_loss_and_ignores_old_loss(self):
        manager = RiskManager()
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        marker = start + timedelta(minutes=10)
        old_loss = start + timedelta(minutes=5)
        manager._maybe_reset_daily(marker)
        manager._daily_loss_usd = 0.70
        manager._last_loss_time = old_loss

        manager.reset_loss_cooldown(marker)
        manager.synchronize_closed_trades(
            [{"close_time": old_loss.isoformat(), "net_profit": -0.70}],
            balance=14.39,
        )

        self.assertEqual(manager.daily_loss_usd, 0.70)
        self.assertIsNone(manager._last_loss_time)

    def test_new_loss_after_cooldown_reset_starts_a_new_cooldown(self):
        manager = RiskManager()
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        marker = start + timedelta(minutes=10)
        new_loss = marker + timedelta(minutes=1)
        manager.reset_loss_cooldown(marker)

        manager.synchronize_closed_trades(
            [{"close_time": new_loss.isoformat(), "net_profit": -0.20}],
            balance=14.19,
        )

        self.assertEqual(manager._last_loss_time, new_loss)


class _BarReader:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    async def get_ohlcv(self, symbol, timeframe, count):
        self.calls.append((symbol, timeframe, count))
        return self.frame


class EntryBarFreshnessTests(unittest.TestCase):
    @staticmethod
    def _engine_with(frame):
        engine = TradingEngine.__new__(TradingEngine)
        engine.reader = _BarReader(frame)
        return engine

    def test_matching_latest_completed_bar_remains_executable(self):
        bar = pd.Timestamp("2026-07-13T12:00:00Z")
        frame = pd.DataFrame({"time": [bar]})
        frame.attrs["age_after_close_seconds"] = 0.0
        engine = self._engine_with(frame)

        ok, reason = asyncio.run(engine._entry_bar_is_current("EURUSD", str(bar)))

        self.assertTrue(ok, reason)
        self.assertEqual(engine.reader.calls, [("EURUSD", "M5", 3)])

    def test_missing_bar_close_age_fails_closed(self):
        bar = pd.Timestamp("2026-07-13T12:00:00Z")
        engine = self._engine_with(pd.DataFrame({"time": [bar]}))

        ok, reason = asyncio.run(engine._entry_bar_is_current("EURUSD", str(bar)))

        self.assertFalse(ok)
        self.assertIn("bar age is unavailable", reason)

    def test_new_completed_bar_expires_entry_decision(self):
        analyzed = pd.Timestamp("2026-07-13T12:00:00Z")
        latest = pd.Timestamp("2026-07-13T12:05:00Z")
        frame = pd.DataFrame({"time": [latest]})
        frame.attrs["age_after_close_seconds"] = 0.0
        engine = self._engine_with(frame)

        ok, reason = asyncio.run(
            engine._entry_bar_is_current("EURUSD", str(analyzed))
        )

        self.assertFalse(ok)
        self.assertIn("Decision expired", reason)
        self.assertIn(str(latest), reason)

    def test_stale_refresh_fails_closed(self):
        bar = pd.Timestamp("2026-07-13T12:00:00Z")
        frame = pd.DataFrame({"time": [bar]})
        frame.attrs["is_stale"] = True
        engine = self._engine_with(frame)

        ok, reason = asyncio.run(engine._entry_bar_is_current("EURUSD", str(bar)))

        self.assertFalse(ok)
        self.assertIn("stale", reason)


class StopRiskReadinessTests(unittest.TestCase):
    def test_unavailable_open_stop_risk_blocks_readiness(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.is_running = True
        engine._history_ready = True
        engine.entries_armed = True
        engine._position_risk_healthy = False
        engine._protection_heartbeat_monotonic = time.monotonic()
        engine._protection_state_healthy = True
        engine.risk = SimpleNamespace(
            daily_loss_status=lambda account: (True, "$0.00 / $3.00")
        )
        account = {
            "login": 123,
            "server": "Pepperstone-Demo",
            "company": "Pepperstone",
            "trade_mode": 0,
            "trade_mode_name": "DEMO",
            "terminal_connected": True,
        }
        engine._armed_account_identity = engine._account_identity(account)
        fake_dashboard = SimpleNamespace(
            automation=SimpleNamespace(llm_online=True),
            account=SimpleNamespace(portfolio_risk_pct=0.0),
            update_readiness=MagicMock(),
        )
        fake_settings = SimpleNamespace(
            dry_run=True,
            max_portfolio_risk_pct=3.0,
            engine_heartbeat_stale_seconds=30.0,
        )

        with (
            patch("core.engine.dashboard_state", fake_dashboard),
            patch("core.engine.settings", fake_settings),
        ):
            engine._update_execution_readiness(account)

        payload = fake_dashboard.update_readiness.call_args.kwargs
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["code"], "STOP_RISK")
        stop_check = next(
            check for check in payload["checks"] if check["code"] == "STOP_RISK"
        )
        self.assertFalse(stop_check["ok"])
        self.assertIn("Unavailable", stop_check["detail"])

    def test_reconciled_daily_loss_blocks_readiness_before_entry_authorization(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.is_running = True
        engine._history_ready = True
        engine.entries_armed = True
        engine._position_risk_healthy = True
        engine._protection_heartbeat_monotonic = time.monotonic()
        engine._protection_state_healthy = True
        engine.risk = SimpleNamespace(
            daily_loss_status=lambda account: (
                False,
                "Gross losses $1.57 / $0.90; new entries resume after the UTC daily reset",
            )
        )
        account = {
            "login": 123,
            "server": "Pepperstone-Live",
            "company": "Pepperstone",
            "trade_mode": 2,
            "trade_mode_name": "LIVE",
            "terminal_connected": True,
            "account_trade_allowed": True,
            "expert_trading_allowed": True,
            "terminal_trade_allowed": True,
            "tradeapi_disabled": False,
            "balance": 15.17,
        }
        engine._armed_account_identity = engine._account_identity(account)
        fake_dashboard = SimpleNamespace(
            automation=SimpleNamespace(llm_online=True),
            account=SimpleNamespace(portfolio_risk_pct=0.0),
            update_readiness=MagicMock(),
        )
        fake_settings = SimpleNamespace(
            dry_run=False,
            max_portfolio_risk_pct=6.0,
            engine_heartbeat_stale_seconds=30.0,
        )

        with (
            patch("core.engine.dashboard_state", fake_dashboard),
            patch("core.engine.settings", fake_settings),
        ):
            engine._update_execution_readiness(account)

        payload = fake_dashboard.update_readiness.call_args.kwargs
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["code"], "DAILY_LOSS")
        daily_check = next(
            check for check in payload["checks"] if check["code"] == "DAILY_LOSS"
        )
        self.assertFalse(daily_check["ok"])
        self.assertIn("$1.57 / $0.90", daily_check["detail"])


if __name__ == "__main__":
    unittest.main()

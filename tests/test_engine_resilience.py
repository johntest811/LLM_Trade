import asyncio
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import MetaTrader5 as mt5

from core.engine import TradingEngine
from database.storage import TradingDatabase
from execution.executor import ExecutionResult
from app_config.settings import settings


class _Executor:
    def __init__(self):
        self.shutdown_calls = 0

    def begin_shutdown(self):
        self.shutdown_calls += 1

    async def wait_until_idle(self, timeout=0):
        del timeout
        return True


class _Connection:
    def __init__(self):
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1


class EngineResilienceTests(unittest.TestCase):
    def test_initial_risk_calculation_does_not_block_event_loop(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.db = SimpleNamespace(
            set_position_risk_baseline=AsyncMock(
                return_value={"initial_risk_pips": 10.0}
            )
        )
        engine._initial_risk_pips = {}
        engine._baseline_error_tickets = set()
        engine.log = lambda *args, **kwargs: None

        def slow_native_calculation(*args, **kwargs):
            del args, kwargs
            time.sleep(0.08)
            return -1.0

        async def exercise():
            started = asyncio.get_running_loop().time()
            task = asyncio.create_task(
                engine._persist_initial_risk(
                    account={
                        "company": "Pepperstone",
                        "server": "Demo",
                        "login": 1001,
                        "trade_mode": 0,
                    },
                    ticket=77,
                    symbol="EURUSD",
                    direction="BUY",
                    entry_price=1.1000,
                    initial_sl=1.0990,
                    volume=0.01,
                    info=SimpleNamespace(point=0.00001, digits=5),
                )
            )
            await asyncio.sleep(0.01)
            event_loop_delay = asyncio.get_running_loop().time() - started
            result = await task
            return event_loop_delay, result

        with patch(
            "core.engine.mt5.order_calc_profit",
            side_effect=slow_native_calculation,
        ):
            event_loop_delay, (risk_pips, healthy) = asyncio.run(exercise())

        self.assertLess(event_loop_delay, 0.05)
        self.assertTrue(healthy)
        self.assertGreater(risk_pips, 0.0)

    def test_manual_override_rejects_wrong_confirmation_without_broker_work(self):
        engine = TradingEngine.__new__(TradingEngine)

        ok, message = asyncio.run(
            engine.open_rejected_trade("USDJPY", "OPEN USDJPY")
        )

        self.assertFalse(ok)
        self.assertEqual(message, "Type FORCE OPEN REJECTED USDJPY to confirm")

    def test_restart_restores_original_r_after_stop_has_moved(self):
        baseline = {
            "ticket": 77,
            "symbol": "USDJPY",
            "direction": "BUY",
            "entry_price": 150.0,
            "initial_sl": 149.9,
            "initial_risk_pips": 10.0,
            "initial_risk_usd": 0.67,
            "initial_volume": 0.01,
        }
        account = {
            "company": "Pepperstone",
            "server": "Demo",
            "login": 1001,
            "trade_mode": 0,
            "trade_mode_name": "DEMO",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database = TradingDatabase(str(Path(temp_dir) / "risk.db"))
            asyncio.run(database.set_position_risk_baseline(
                TradingEngine._account_scope(account), baseline
            ))
            engine = TradingEngine.__new__(TradingEngine)
            engine.db = database
            engine._initial_risk_pips = {}
            engine._baseline_error_tickets = set()
            engine.log = lambda *args, **kwargs: None
            position = SimpleNamespace(
                ticket=77,
                symbol="USDJPY",
                type=0,
                volume=0.01,
                price_open=150.0,
                sl=150.02,
            )
            restored, healthy = asyncio.run(
                engine._initial_risk_for_position(
                    account,
                    position,
                    SimpleNamespace(point=0.001, digits=3),
                )
            )

        self.assertTrue(healthy)
        self.assertEqual(restored, 10.0)

    def test_restart_restores_persisted_profit_peak(self):
        account = {
            "company": "Pepperstone",
            "server": "Demo",
            "login": 1001,
            "trade_mode": 0,
            "trade_mode_name": "DEMO",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database = TradingDatabase(str(Path(temp_dir) / "peak.db"))

            first = TradingEngine.__new__(TradingEngine)
            first.db = database
            first._peak_profits = {}
            first._peak_profit_usd = {}
            first._peak_state_loaded = set()
            first._peak_persisted_usd = {}
            first.log = lambda *args, **kwargs: None
            peak_pips, peak_usd = asyncio.run(
                first._restore_and_update_position_peak(
                    account,
                    ticket=88,
                    profit_pips=3.0,
                    observed_profit_usd=0.24,
                )
            )
            self.assertEqual(peak_pips, 3.0)
            self.assertEqual(peak_usd, 0.24)

            restarted = TradingEngine.__new__(TradingEngine)
            restarted.db = database
            restarted._peak_profits = {}
            restarted._peak_profit_usd = {}
            restarted._peak_state_loaded = set()
            restarted._peak_persisted_usd = {}
            restarted.log = lambda *args, **kwargs: None
            restored_pips, restored_usd = asyncio.run(
                restarted._restore_and_update_position_peak(
                    account,
                    ticket=88,
                    profit_pips=-2.0,
                    observed_profit_usd=-0.10,
                )
            )

        self.assertEqual(restored_pips, 3.0)
        self.assertEqual(restored_usd, 0.24)

    def test_r_based_profit_giveback_closes_after_one_r_peak(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(
                    True, 88, 115.10, 0.01, None
                )
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {88: 12.0}
        engine._peak_profit_usd = {88: 0.20}
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()

        position = {
            "ticket": 88,
            "symbol": "CADJPY",
            "profit": 0.08,
            "estimated_net_profit_usd": 0.01,
            "profit_pips": 4.0,
            "peak_profit_usd": 0.20,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 115.10,
            "sl": 114.90,
            "tp": 115.40,
        }
        protection_settings = replace(
            settings,
            early_profit_lock_enabled=False,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=1.0,
            profit_giveback_fraction=0.50,
            profit_lock_floor_usd=0.03,
        )
        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            88, expected_account=engine._active_account_identity
        )

    def test_mid_r_profit_lock_scales_floor_from_cost_adjusted_peak(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(
                return_value=ExecutionResult(True, 188, 0.70715, 0.02, None)
            ),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {188: 7.3}
        engine._peak_profit_usd = {188: 0.88}
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 188,
            "symbol": "AUDUSD",
            "profit": 0.84,
            "estimated_net_profit_usd": 0.80,
            "profit_pips": 7.0,
            "peak_profit_usd": 0.88,
            "initial_risk_pips": 10.0,
            "volume": 0.02,
            "price_current": 0.70764,
            "sl": 0.70634,
            "tp": 0.70807,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            early_profit_lock_enabled=False,
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_lock_min_live_fraction=0.75,
            profit_lock_floor_usd=0.03,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=0.50,
            profit_giveback_fraction=0.50,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_awaited_once_with(
            188,
            floor_usd=0.42,
            expected_account=engine._active_account_identity,
        )

    def test_mid_r_giveback_closes_after_half_of_peak_is_lost(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(True, 189, 0.70729, 0.02, None)
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {189: 7.3}
        engine._peak_profit_usd = {189: 0.88}
        engine._profit_lock_tickets = {189}
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 189,
            "symbol": "AUDUSD",
            "profit": 0.42,
            "estimated_net_profit_usd": 0.38,
            "profit_pips": 3.5,
            "peak_profit_usd": 0.88,
            "initial_risk_pips": 10.0,
            "volume": 0.02,
            "price_current": 0.70729,
            "sl": 0.70634,
            "tp": 0.70807,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            early_profit_lock_enabled=False,
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=0.50,
            profit_giveback_fraction=0.50,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            189, expected_account=engine._active_account_identity
        )
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_fixed_dollar_giveback_works_without_risk_baseline(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(True, 89, 1.1375, 0.01, None)
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {89: 4.2}
        engine._peak_profit_usd = {89: 0.42}
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 89,
            "symbol": "EURUSD",
            "profit": 0.18,
            "estimated_net_profit_usd": 0.11,
            "profit_pips": 1.8,
            "peak_profit_usd": 0.42,
            "initial_risk_pips": 0.0,
            "volume": 0.01,
            "price_current": 1.1375,
            "sl": 1.1380,
            "tp": 1.1356,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=True,
            early_profit_lock_enabled=False,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=1.0,
            profit_giveback_trigger_usd=0.20,
            profit_giveback_fraction=0.50,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            89, expected_account=engine._active_account_identity
        )

    def test_protection_revalidates_profit_after_waiting_for_lock(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {}
        engine._peak_profit_usd = {}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine._initial_risk_for_position = AsyncMock(
            return_value=(10.0, True)
        )
        engine._restore_and_update_position_peak = AsyncMock(
            return_value=(1.0, 0.10)
        )
        engine.log = MagicMock()
        stale_snapshot = {
            "ticket": 105,
            "symbol": "EURUSD",
            "type": mt5.POSITION_TYPE_BUY,
            "volume": 0.01,
            "price_open": 1.1000,
            "price_current": 1.0990,
            "sl": 1.0980,
            "tp": 1.1020,
            "profit": -0.80,
            "estimated_net_profit_usd": -0.80,
            "profit_pips": -10.0,
            "peak_profit_usd": 0.0,
            "initial_risk_pips": 10.0,
        }
        fresh_position = SimpleNamespace(
            ticket=105,
            symbol="EURUSD",
            type=mt5.POSITION_TYPE_BUY,
            volume=0.01,
            price_open=1.1000,
            price_current=1.1001,
            sl=1.0980,
            tp=1.1020,
            profit=0.10,
        )
        protection_settings = replace(
            settings,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=True,
            auto_close_loss_usd=0.50,
            micro_profit_protection_enabled=False,
            early_profit_lock_enabled=False,
            profit_giveback_enabled=False,
            profit_lock_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with (
            patch("core.engine.settings", protection_settings),
            patch(
                "core.engine.mt5.positions_get",
                return_value=(fresh_position,),
            ),
            patch(
                "core.engine.mt5.symbol_info",
                return_value=SimpleNamespace(point=0.00001, digits=5),
            ),
            patch("core.engine.pip_size", return_value=0.0001),
            patch(
                "core.engine.configured_execution_cost_usd",
                return_value=0.0,
            ),
        ):
            asyncio.run(
                engine._apply_protections(
                    [stale_snapshot],
                    refresh_from_broker=True,
                )
            )

        engine.executor.close_position.assert_not_awaited()

    def test_protection_snapshot_rejects_an_account_switch_before_reading_ticket(self):
        engine = TradingEngine.__new__(TradingEngine)
        expected = {
            "login": 1001,
            "server": "Pepperstone-Demo",
            "company": "Pepperstone",
            "trade_mode": 0,
        }
        engine._active_account_identity = dict(expected)
        snapshot = {
            "ticket": 106,
            "account_identity": dict(expected),
        }
        positions_get = MagicMock()
        changed_account = SimpleNamespace(
            login=2002,
            server="Pepperstone-Live",
            company="Pepperstone",
            trade_mode=2,
        )

        with (
            patch(
                "core.engine.mt5.account_info",
                return_value=changed_account,
            ),
            patch(
                "core.engine.mt5.positions_get",
                positions_get,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "account changed"):
                asyncio.run(engine._refresh_protection_snapshot(snapshot))

        positions_get.assert_not_called()

    def test_micro_profit_guard_locks_ten_cents_at_twenty_cent_peak(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(
                return_value=ExecutionResult(True, 90, 1.1001, 0.01, None)
            ),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {90: 2.0}
        engine._peak_profit_usd = {90: 0.20}
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 90,
            "symbol": "EURUSD",
            "profit": 0.20,
            "estimated_net_profit_usd": 0.20,
            "profit_pips": 2.0,
            "peak_profit_usd": 0.20,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 1.1002,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=True,
            early_profit_lock_enabled=False,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=10.0,
            profit_giveback_trigger_usd=0.20,
            profit_giveback_fraction=0.25,
            profit_lock_enabled=True,
            profit_lock_trigger_r=10.0,
            profit_lock_trigger_usd=0.20,
            profit_lock_floor_usd=0.10,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_awaited_once_with(
            90,
            floor_usd=0.10,
            expected_account=engine._active_account_identity,
        )

    def test_twenty_cent_peak_keeps_headroom_when_micro_toggle_is_off(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {94: 2.2}
        engine._peak_profit_usd = {94: 0.20}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 94,
            "symbol": "AUDCAD",
            "profit": 0.03,
            "estimated_net_profit_usd": 0.03,
            "profit_pips": 0.4,
            "peak_profit_usd": 0.20,
            "initial_risk_pips": 8.0,
            "volume": 0.01,
            "price_current": 0.97814,
            "sl": 0.97898,
            "tp": 0.97674,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            early_profit_lock_enabled=False,
            profit_lock_enabled=True,
            profit_lock_trigger_r=10.0,
            profit_lock_trigger_usd=0.20,
            profit_lock_floor_usd=0.10,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=10.0,
            profit_giveback_trigger_usd=0.20,
            profit_giveback_fraction=0.25,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_not_awaited()
        engine.executor.move_to_breakeven.assert_not_awaited()
        engine.executor.apply_trailing_stop.assert_not_awaited()

    def test_micro_profit_guard_closes_after_twenty_five_percent_giveback(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(True, 91, 1.10015, 0.01, None)
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {91: 2.0}
        engine._peak_profit_usd = {91: 0.20}
        engine._profit_lock_tickets = {91}
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 91,
            "symbol": "EURUSD",
            "profit": 0.15,
            "estimated_net_profit_usd": 0.15,
            "profit_pips": 1.5,
            "peak_profit_usd": 0.20,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 1.10015,
            "sl": 1.1001,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=True,
            early_profit_lock_enabled=False,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=10.0,
            profit_giveback_trigger_usd=0.20,
            profit_giveback_fraction=0.25,
            profit_lock_enabled=True,
            profit_lock_trigger_r=10.0,
            profit_lock_trigger_usd=0.20,
            profit_lock_floor_usd=0.10,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            91, expected_account=engine._active_account_identity
        )
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_early_profit_lock_preserves_headroom_at_twelve_cent_peak(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(
                return_value=ExecutionResult(True, 92, 1.10003, 0.01, None)
            ),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {92: 1.2}
        engine._peak_profit_usd = {92: 0.12}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 92,
            "symbol": "EURUSD",
            "profit": 0.12,
            "estimated_net_profit_usd": 0.12,
            "profit_pips": 1.2,
            "peak_profit_usd": 0.12,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 1.10012,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=True,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            early_profit_lock_enabled=True,
            early_profit_lock_trigger_usd=0.12,
            early_profit_lock_floor_usd=0.03,
            profit_lock_enabled=False,
            profit_giveback_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_awaited_once_with(
            92,
            floor_usd=0.03,
            expected_account=engine._active_account_identity,
        )

    def test_early_profit_lock_fallback_closes_before_gain_turns_negative(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(True, 93, 1.10003, 0.01, None)
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {93: 1.2}
        engine._peak_profit_usd = {93: 0.12}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 93,
            "symbol": "EURUSD",
            "profit": 0.03,
            "estimated_net_profit_usd": 0.03,
            "profit_pips": 0.3,
            "peak_profit_usd": 0.12,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 1.10003,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=True,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            early_profit_lock_enabled=True,
            early_profit_lock_trigger_usd=0.12,
            early_profit_lock_floor_usd=0.03,
            profit_lock_enabled=False,
            profit_giveback_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            93, expected_account=engine._active_account_identity
        )
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_early_profit_lock_fallback_uses_net_profit_after_costs(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(True, 94, 1.10005, 0.01, None)
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {94: 1.5}
        engine._peak_profit_usd = {94: 0.15}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 94,
            "symbol": "EURUSD",
            "profit": 0.05,
            "estimated_net_profit_usd": 0.03,
            "profit_pips": 0.5,
            "peak_profit_usd": 0.15,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 1.10005,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=True,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            early_profit_lock_enabled=True,
            early_profit_lock_trigger_usd=0.12,
            early_profit_lock_floor_usd=0.03,
            profit_lock_enabled=False,
            profit_giveback_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            94, expected_account=engine._active_account_identity
        )
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_r_profit_lock_waits_for_current_price_to_retain_peak_progress(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {195: 6.0}
        engine._peak_profit_usd = {195: 0.30}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(update_replay_outcome=MagicMock())
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 195,
            "symbol": "EURUSD",
            "profit": 0.15,
            "estimated_net_profit_usd": 0.15,
            "profit_pips": 3.0,
            "peak_profit_usd": 0.30,
            "initial_risk_pips": 10.0,
            "duration_min": 10.0,
            "volume": 0.01,
            "price_current": 1.1003,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            profit_giveback_enabled=False,
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_lock_min_live_fraction=0.75,
            position_stagnation_exit_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_stronger_stop_satisfies_break_even_without_repeated_requests(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(
                return_value=ExecutionResult(
                    False,
                    197,
                    None,
                    None,
                    "Break-even would worsen the current stop",
                )
            ),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {197: 8.0}
        engine._peak_profit_usd = {197: 0.80}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = {197}
        engine._breakeven_tickets = set()
        engine.replay_logger = SimpleNamespace(update_replay_outcome=MagicMock())
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 197,
            "symbol": "USDJPY",
            "profit": 0.80,
            "estimated_net_profit_usd": 0.80,
            "profit_pips": 8.0,
            "peak_profit_usd": 0.80,
            "initial_risk_pips": 10.0,
            "duration_min": 10.0,
            "volume": 0.02,
            "price_current": 158.80,
            "sl": 158.79,
            "tp": 158.95,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            profit_giveback_enabled=False,
            profit_lock_enabled=False,
            position_stagnation_exit_enabled=False,
            breakeven_trigger_r=0.75,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))
            asyncio.run(engine._apply_protections([position]))

        engine.executor.move_to_breakeven.assert_awaited_once_with(
            197,
            buffer_pips=protection_settings.breakeven_buffer_pips,
            expected_account=engine._active_account_identity,
        )
        self.assertIn(197, engine._breakeven_tickets)

    def test_stagnation_exit_closes_only_after_bars_without_progress(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(
                return_value=ExecutionResult(True, 196, 1.0998, 0.01, None)
            ),
            lock_minimum_net_profit=AsyncMock(),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {196: 1.0}
        engine._peak_profit_usd = {196: 0.05}
        engine._early_profit_lock_tickets = set()
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(update_replay_outcome=MagicMock())
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 196,
            "symbol": "EURUSD",
            "profit": -0.08,
            "estimated_net_profit_usd": -0.10,
            "profit_pips": -2.0,
            "peak_profit_usd": 0.05,
            "initial_risk_pips": 10.0,
            "duration_min": 61.0,
            "volume": 0.01,
            "price_current": 1.0998,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            profit_giveback_enabled=False,
            profit_lock_enabled=False,
            position_stagnation_exit_enabled=True,
            position_stagnation_bars=12,
            position_stagnation_max_peak_r=0.15,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            196, expected_account=engine._active_account_identity
        )
        engine.db.log_trade.assert_awaited_once()

    def test_confirmed_m1_and_fast_m5_reversal_bypasses_model_delay(self):
        analyses = {
            "M1": {
                "timestamp": "2026-07-29T06:01:00+00:00",
                "market_structure": {
                    "trend_state_direction": "BEARISH",
                    "structure_events": [
                        {
                            "type": "CHOCH",
                            "direction": "BEARISH",
                            "time": "2026-07-29T06:01:00+00:00",
                        }
                    ],
                },
            },
            "M5": {
                "market_structure": {
                    "trend_state_direction": "BULLISH",
                    "fast_trend": "BEARISH",
                    "structure_events": [],
                }
            },
        }

        confirmed, reason = TradingEngine._confirmed_adverse_reversal(
            analyses, 0
        )

        self.assertTrue(confirmed)
        self.assertIn("against BUY", reason)

    def test_m1_pullback_without_fast_m5_reversal_does_not_force_exit(self):
        analyses = {
            "M1": {
                "timestamp": "2026-07-29T06:01:00+00:00",
                "market_structure": {
                    "trend_state_direction": "BEARISH",
                    "structure_events": [
                        {
                            "type": "CHOCH",
                            "direction": "BEARISH",
                            "time": "2026-07-29T06:01:00+00:00",
                        }
                    ],
                },
            },
            "M5": {
                "market_structure": {
                    "trend_state_direction": "BULLISH",
                    "fast_trend": "BULLISH",
                    "structure_events": [],
                }
            },
        }

        confirmed, reason = TradingEngine._confirmed_adverse_reversal(
            analyses, 0
        )

        self.assertFalse(confirmed)
        self.assertEqual(reason, "")

    @staticmethod
    def _adverse_m1_analysis():
        return {
            "M1": {
                "market_structure": {
                    "trend_state_direction": "BEARISH",
                    "fast_trend": "BEARISH",
                    "structure_events": [],
                },
                "indicators": {
                    "rsi_14": 42.0,
                    "macd": {"diff": -0.00008},
                    "candle_body_atr_signed": -0.35,
                    "candle_return_atr": -0.40,
                },
            },
            "M5": {
                "market_structure": {
                    "trend_state_direction": "BULLISH",
                    "fast_trend": "BULLISH",
                    "structure_events": [],
                }
            },
        }

    def test_persistent_adverse_m1_momentum_can_cut_loss_before_full_stop(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._adverse_momentum_streaks = {}
        position = {
            "ticket": 701,
            "type": mt5.POSITION_TYPE_BUY,
            "initial_risk_pips": 8.0,
            "profit_pips": -3.2,
        }

        first, _ = engine._confirmed_adverse_momentum(
            self._adverse_m1_analysis(), position
        )
        second, reason = engine._confirmed_adverse_momentum(
            self._adverse_m1_analysis(), position
        )

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertIn("Persistent adverse M1 momentum", reason)
        self.assertIn("-0.40 R", reason)

    def test_one_adverse_bar_or_small_drawdown_does_not_force_exit(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._adverse_momentum_streaks = {}
        position = {
            "ticket": 702,
            "type": mt5.POSITION_TYPE_BUY,
            "initial_risk_pips": 8.0,
            "profit_pips": -1.6,
        }

        first, _ = engine._confirmed_adverse_momentum(
            self._adverse_m1_analysis(), position
        )
        second, _ = engine._confirmed_adverse_momentum(
            self._adverse_m1_analysis(), position
        )

        self.assertFalse(first)
        self.assertFalse(second)

    def test_unexpected_main_loop_exit_forces_engine_error_shutdown(self):
        async def scenario():
            engine = TradingEngine.__new__(TradingEngine)
            engine.is_running = True
            engine.entries_armed = True
            engine._armed_account_identity = {"login": 1}
            engine._failure_task = None
            engine._lifecycle_lock = asyncio.Lock()
            engine._execution_lock = asyncio.Lock()
            engine.analysis_tasks = {}
            engine.tick_loop_task = None
            engine.executor = _Executor()
            engine.conn = _Connection()
            engine.log = lambda *args, **kwargs: None
            engine.disarm_entries = lambda reason="": setattr(engine, "entries_armed", False)
            engine._update_execution_readiness = lambda *args, **kwargs: None

            async def fail():
                raise RuntimeError("boom")

            engine.loop_task = asyncio.create_task(fail())
            engine.loop_task.add_done_callback(
                lambda task: engine._core_task_finished("decision", task)
            )
            await asyncio.sleep(0.1)
            if engine._failure_task is not None:
                await engine._failure_task
            return engine

        fake_dashboard = SimpleNamespace(
            engine_running=True,
            update_automation=MagicMock(),
        )
        with patch("core.engine.dashboard_state", fake_dashboard):
            engine = asyncio.run(scenario())

        self.assertFalse(engine.is_running)
        self.assertFalse(engine.entries_armed)
        self.assertFalse(fake_dashboard.engine_running)
        self.assertGreaterEqual(engine.executor.shutdown_calls, 1)
        self.assertEqual(engine.conn.shutdown_calls, 1)
        statuses = [
            call.kwargs.get("scan_status", "")
            for call in fake_dashboard.update_automation.call_args_list
        ]
        self.assertTrue(any("ENGINE ERROR" in status for status in statuses))


if __name__ == "__main__":
    unittest.main()

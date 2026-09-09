import asyncio
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

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
    @staticmethod
    def _fast_path_analysis(direction="BULLISH", *, trigger=False):
        bullish = direction == "BULLISH"
        structure = {
            "trend": direction,
            "trend_state_direction": direction,
            "trend_state": f"CONFIRMED_{direction}",
            "structure_events": [],
            "breakout_status": "NONE",
        }
        if trigger:
            structure["structure_events"] = [
                {
                    "type": "BOS",
                    "direction": direction,
                    "level": 1.19 if bullish else 1.21,
                    "time": "2026-08-12T00:05:00+00:00",
                }
            ]
        return {
            "timestamp": "2026-08-12T00:05:00+00:00",
            "indicators": {
                "adx_14": 30.0,
                "current_price": 1.2,
                "atr_14": .01,
                "ema_9": 1.2 if bullish else 1.0,
                "ema_21": 1.1,
                "rsi_14": 58.0 if bullish else 42.0,
                "macd": {"diff": 0.01 if bullish else -0.01},
                "candle_range_atr": 0.8,
                "opening_gap_atr": 0.0,
            },
            "market_structure": structure,
        }

    def _fast_path_context(self, direction="BULLISH"):
        return {
            "symbol": "USDJPY",
            "completed_bar": "2026-08-12 00:05:00",
            "has_open_position": False,
            "open_positions": [],
            "previous_trend_states": {},
            "market": {},
            "analyses": {
                timeframe: self._fast_path_analysis(
                    direction,
                    trigger=timeframe == "M5",
                )
                for timeframe in ("M5", "M15", "H1", "H4")
            },
        }

    def test_deterministic_entry_fast_path_accepts_one_fully_aligned_direction(self):
        context = self._fast_path_context()
        enabled = replace(settings, deterministic_entry_fast_path_enabled=True)

        with (
            patch("core.engine.settings", enabled),
            patch("llm.client.settings", enabled),
        ):
            decision = TradingEngine._deterministic_entry_fast_path(
                context,
                ("BUY",),
            )

        self.assertIsNotNone(decision)
        self.assertEqual(decision["action"], "BUY")
        self.assertEqual(decision["_decision_path"], "DETERMINISTIC_FAST_PATH")
        self.assertGreaterEqual(decision["confidence"], enabled.confidence_threshold)

    def test_deterministic_entry_fast_path_rejects_macro_disagreement(self):
        context = self._fast_path_context()
        context["analyses"]["H4"] = self._fast_path_analysis("BEARISH")
        enabled = replace(settings, deterministic_entry_fast_path_enabled=True)

        with (
            patch("core.engine.settings", enabled),
            patch("llm.client.settings", enabled),
        ):
            decision = TradingEngine._deterministic_entry_fast_path(
                context,
                ("BUY",),
            )

        self.assertIsNone(decision)

    def test_deterministic_entry_fast_path_requires_unique_fresh_m5_direction(self):
        context = self._fast_path_context()
        enabled = replace(settings, deterministic_entry_fast_path_enabled=True)

        with (
            patch("core.engine.settings", enabled),
            patch("llm.client.settings", enabled),
        ):
            ambiguous = TradingEngine._deterministic_entry_fast_path(
                context,
                ("BUY", "SELL"),
            )
            context["analyses"]["M5"] = self._fast_path_analysis("BULLISH")
            missing_trigger = TradingEngine._deterministic_entry_fast_path(
                context,
                ("BUY",),
            )

        self.assertIsNone(ambiguous)
        self.assertIsNone(missing_trigger)

    def test_deterministic_entry_fast_path_can_be_disabled(self):
        context = self._fast_path_context()
        disabled = replace(settings, deterministic_entry_fast_path_enabled=False)

        with patch("core.engine.settings", disabled):
            decision = TradingEngine._deterministic_entry_fast_path(
                context,
                ("BUY",),
            )

        self.assertIsNone(decision)

    def test_entry_model_admission_is_idempotent_and_bounded_per_bar(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._entry_model_admissions = {}
        bounded_settings = replace(settings, llm_entry_candidates_per_bar=2)

        with patch("core.engine.settings", bounded_settings):
            first = engine._reserve_entry_model_slot("USDJPY", "2026-08-12 00:00")
            repeated = engine._reserve_entry_model_slot("USDJPY", "2026-08-12 00:00")
            second = engine._reserve_entry_model_slot("EURUSD", "2026-08-12 00:00")
            rejected = engine._reserve_entry_model_slot("GBPUSD", "2026-08-12 00:00")
            next_bar = engine._reserve_entry_model_slot("GBPUSD", "2026-08-12 00:05")

        self.assertTrue(first[0])
        self.assertTrue(repeated[0])
        self.assertTrue(second[0])
        self.assertFalse(rejected[0])
        self.assertIn("already admitted 2 candidates", rejected[1])
        self.assertTrue(next_bar[0])

    def test_entry_model_admission_prunes_old_completed_bars(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._entry_model_admissions = {}

        for index in range(10):
            admitted, _ = engine._reserve_entry_model_slot(
                "USDJPY", f"2026-08-12 00:{index:02d}"
            )
            self.assertTrue(admitted)

        self.assertLessEqual(len(engine._entry_model_admissions), 8)
        self.assertNotIn("2026-08-12 00:00", engine._entry_model_admissions)

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
        engine._peak_profits = {88: 10.0}
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
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=0.50,
            profit_giveback_close_min_r=1.00,
            profit_giveback_fraction=0.50,
            profit_lock_floor_usd=0.03,
        )
        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_awaited_once_with(
            88, expected_account=engine._active_account_identity
        )

    def test_mid_profit_lock_protects_35_percent_of_live_net_profit(self):
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
            "profit_pips": 8.0,
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
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_lock_trigger_usd=0.35,
            profit_lock_floor_usd=0.08,
            profit_lock_mid_trigger_r=0.75,
            profit_lock_mid_trigger_usd=0.60,
            profit_lock_mid_fraction=0.35,
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
            floor_usd=0.28,
            expected_account=engine._active_account_identity,
        )

    def test_profit_lock_tiers_require_both_live_usd_and_r(self):
        protection_settings = replace(
            settings,
            profit_lock_trigger_r=0.50,
            profit_lock_trigger_usd=0.35,
            profit_lock_floor_usd=0.08,
            profit_lock_mid_trigger_r=0.75,
            profit_lock_mid_trigger_usd=0.60,
            profit_lock_mid_fraction=0.35,
            profit_lock_final_trigger_usd=1.15,
            profit_lock_final_floor_usd=1.00,
        )
        with patch("core.engine.settings", protection_settings):
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.50,
                    estimated_net_profit_usd=0.34,
                    has_r_baseline=True,
                ),
                (0.0, ""),
            )
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.49,
                    estimated_net_profit_usd=0.35,
                    has_r_baseline=True,
                ),
                (0.0, ""),
            )
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.50,
                    estimated_net_profit_usd=0.35,
                    has_r_baseline=True,
                ),
                (0.08, "first"),
            )
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.75,
                    estimated_net_profit_usd=0.60,
                    has_r_baseline=True,
                ),
                (0.21, "35%"),
            )

    def test_final_profit_lock_is_one_dollar_without_r_baseline(self):
        protection_settings = replace(
            settings,
            profit_lock_final_trigger_usd=1.15,
            profit_lock_final_floor_usd=1.00,
            profit_lock_final_fallback_only=True,
        )
        with patch("core.engine.settings", protection_settings):
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.0,
                    estimated_net_profit_usd=1.15,
                    has_r_baseline=False,
                ),
                (1.00, "final"),
            )

    def test_mature_profit_lock_scales_with_immutable_initial_risk(self):
        protection_settings = replace(
            settings,
            profit_lock_mature_trigger_r=1.25,
            profit_lock_mature_floor_r=0.70,
            profit_lock_final_fallback_only=True,
        )
        with patch("core.engine.settings", protection_settings):
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=1.25,
                    estimated_net_profit_usd=1.84,
                    has_r_baseline=True,
                    initial_risk_usd=1.40,
                ),
                (0.98, "mature 0.70R"),
            )
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=1.24,
                    estimated_net_profit_usd=1.84,
                    has_r_baseline=True,
                    initial_risk_usd=1.40,
                ),
                (0.64, "35%"),
            )

    def test_mature_profit_lock_keeps_execution_headroom(self):
        protection_settings = replace(
            settings,
            profit_lock_mature_trigger_r=1.25,
            profit_lock_mature_floor_r=0.70,
            profit_lock_final_fallback_only=True,
        )
        with patch("core.engine.settings", protection_settings):
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=1.25,
                    estimated_net_profit_usd=0.50,
                    has_r_baseline=True,
                    initial_risk_usd=1.00,
                ),
                (0.40, "mature 0.70R"),
            )

    def test_hybrid_one_dollar_lock_is_bounded_by_initial_risk(self):
        protection_settings = replace(
            settings,
            profit_lock_final_trigger_usd=1.15,
            profit_lock_final_trigger_r=0.60,
            profit_lock_final_floor_usd=1.00,
            profit_lock_final_max_floor_r=0.55,
            profit_lock_final_fallback_only=False,
        )
        with patch("core.engine.settings", protection_settings):
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.63,
                    estimated_net_profit_usd=1.20,
                    has_r_baseline=True,
                    initial_risk_usd=1.83,
                ),
                (1.00, "hybrid $1"),
            )
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.82,
                    estimated_net_profit_usd=1.20,
                    has_r_baseline=True,
                    initial_risk_usd=1.40,
                ),
                (0.77, "hybrid $1"),
            )
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=0.59,
                    estimated_net_profit_usd=1.20,
                    has_r_baseline=True,
                    initial_risk_usd=1.83,
                ),
                (0.08, "first"),
            )

    def test_fixed_final_tier_does_not_override_r_based_protection(self):
        protection_settings = replace(
            settings,
            profit_lock_mid_trigger_r=0.75,
            profit_lock_mid_trigger_usd=0.60,
            profit_lock_mid_fraction=0.35,
            profit_lock_final_trigger_usd=1.15,
            profit_lock_final_floor_usd=1.00,
            profit_lock_final_fallback_only=True,
        )
        with patch("core.engine.settings", protection_settings):
            self.assertEqual(
                TradingEngine._profit_lock_target(
                    live_r=1.00,
                    estimated_net_profit_usd=1.15,
                    has_r_baseline=True,
                ),
                (0.40, "35%"),
            )

    def test_profit_lock_can_upgrade_to_legacy_final_tier_when_enabled(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.executor = SimpleNamespace(
            close_position=AsyncMock(),
            lock_minimum_net_profit=AsyncMock(
                return_value=ExecutionResult(True, 190, 1.10008, 0.01, None)
            ),
            move_to_breakeven=AsyncMock(),
            apply_trailing_stop=AsyncMock(),
        )
        engine._active_account_identity = {"login": 1001}
        engine._peak_profits = {190: 5.0}
        engine._peak_profit_usd = {190: 0.35}
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(update_replay_outcome=MagicMock())
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 190,
            "symbol": "EURUSD",
            "profit": 0.35,
            "estimated_net_profit_usd": 0.35,
            "profit_pips": 5.0,
            "peak_profit_usd": 0.35,
            "initial_risk_pips": 10.0,
            "duration_min": 10.0,
            "volume": 0.01,
            "price_current": 1.1005,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            auto_close_profit_enabled=False,
            auto_close_loss_enabled=False,
            profit_giveback_enabled=False,
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_lock_trigger_usd=0.35,
            profit_lock_floor_usd=0.08,
            profit_lock_mid_trigger_r=0.75,
            profit_lock_mid_trigger_usd=0.60,
            profit_lock_mid_fraction=0.35,
            profit_lock_final_trigger_usd=1.15,
            profit_lock_final_floor_usd=1.00,
            profit_lock_final_fallback_only=False,
            position_stagnation_exit_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))
            position.update(
                profit=1.15,
                estimated_net_profit_usd=1.15,
                profit_pips=11.5,
                peak_profit_usd=1.15,
            )
            asyncio.run(engine._apply_protections([position]))

        engine.executor.lock_minimum_net_profit.assert_has_awaits(
            [
                call(
                    190,
                    floor_usd=0.08,
                    expected_account=engine._active_account_identity,
                ),
                call(
                    190,
                    floor_usd=1.00,
                    expected_account=engine._active_account_identity,
                ),
            ]
        )
        self.assertEqual(engine._profit_lock_levels[190], 1.00)

    def test_mid_r_giveback_keeps_protected_trade_open_below_one_r(self):
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
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=0.50,
            profit_giveback_close_min_r=1.00,
            profit_giveback_fraction=0.50,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
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

    def test_first_profit_tier_locks_eight_cents_at_threshold(self):
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
        engine._peak_profits = {90: 5.0}
        engine._peak_profit_usd = {90: 0.35}
        engine._profit_lock_tickets = set()
        engine.replay_logger = SimpleNamespace(
            update_replay_outcome=MagicMock()
        )
        engine.db = SimpleNamespace(log_trade=AsyncMock(return_value=True))
        engine.log = MagicMock()
        position = {
            "ticket": 90,
            "symbol": "EURUSD",
            "profit": 0.35,
            "estimated_net_profit_usd": 0.35,
            "profit_pips": 5.0,
            "peak_profit_usd": 0.35,
            "initial_risk_pips": 10.0,
            "volume": 0.01,
            "price_current": 1.1002,
            "sl": 1.0990,
            "tp": 1.1020,
        }
        protection_settings = replace(
            settings,
            micro_profit_protection_enabled=False,
            profit_giveback_enabled=False,
            profit_lock_enabled=True,
            profit_lock_trigger_r=0.50,
            profit_lock_trigger_usd=0.35,
            profit_lock_floor_usd=0.08,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_awaited_once_with(
            90,
            floor_usd=0.08,
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

    def test_micro_profit_guard_does_not_override_sub_one_r_maturity_gate(self):
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
            profit_giveback_enabled=True,
            profit_giveback_trigger_r=10.0,
            profit_giveback_close_min_r=1.00,
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
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_retired_small_dollar_lock_does_not_arm_at_twelve_cents(self):
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
            profit_lock_enabled=True,
            profit_giveback_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_retired_small_dollar_fallback_does_not_market_close(self):
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
            profit_lock_enabled=False,
            profit_giveback_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
        engine.executor.lock_minimum_net_profit.assert_not_awaited()

    def test_retired_small_dollar_fallback_ignores_cost_adjusted_retrace(self):
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
            profit_lock_enabled=False,
            profit_giveback_enabled=False,
            breakeven_trigger_r=10.0,
            trailing_trigger_r=10.0,
        )

        with patch("core.engine.settings", protection_settings):
            asyncio.run(engine._apply_protections([position]))

        engine.executor.close_position.assert_not_awaited()
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

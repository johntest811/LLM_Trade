import asyncio
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import MetaTrader5 as mt5

from app_config.settings import settings
from core.engine import TradingEngine
from core.profit_retention import RetentionState, advance_retention
from execution.executor import ExecutionResult, MT5OrderExecutor
from database.storage import TradingDatabase


POLICY = replace(
    settings,
    profit_lock_enabled=True,
    profit_retention_enabled=True,
    profit_retention_trigger_usd=1.50,
    profit_retention_trigger_r=1.0,
    profit_retention_keep_fraction=0.65,
    profit_retention_mature_trigger_r=2.0,
    profit_retention_mature_keep_fraction=0.75,
    profit_retention_min_headroom_usd=0.20,
    profit_lock_final_floor_usd=1.0,
    profit_giveback_enabled=True,
    auto_close_profit_enabled=False,
    auto_close_loss_enabled=False,
    position_stagnation_exit_enabled=False,
    micro_profit_protection_enabled=False,
    breakeven_trigger_r=100.0,
    trailing_trigger_r=100.0,
)


def advance(state, net, *, volume=0.01, risk=1.4, policy=POLICY):
    return advance_retention(
        state, net_profit_usd=net, volume=volume,
        initial_risk_usd=risk, policy=policy,
    )


def position(net, *, volume=0.01, risk=1.4):
    return {
        "ticket": 123, "symbol": "EURJPY", "profit": net,
        "estimated_net_profit_usd": net, "profit_pips": net / risk * 10,
        "initial_risk_pips": 10.0, "initial_risk_usd": risk,
        "volume": volume, "price_current": 178.5, "sl": 179, "tp": 178,
    }


def engine_with_cache(cache=None):
    cache = {} if cache is None else cache
    engine = TradingEngine.__new__(TradingEngine)
    engine._active_account_identity = {"login": 1001, "server": "test"}
    engine._peak_profits = {}
    engine.log = MagicMock()
    engine._remember_strategy_close_reason = AsyncMock()
    engine._update_replay_outcome = AsyncMock()
    engine.db = SimpleNamespace(
        get_cache=AsyncMock(side_effect=lambda key: cache.get(key)),
        set_cache=AsyncMock(side_effect=lambda key, value: cache.update({key: value}) or True),
        log_trade=AsyncMock(),
    )
    engine.executor = SimpleNamespace(
        close_position=AsyncMock(return_value=ExecutionResult(True, 123, 178.5, 0.01, None)),
        lock_minimum_net_profit=AsyncMock(return_value=ExecutionResult(True, 123, None, None, None)),
        move_to_breakeven=AsyncMock(),
        apply_trailing_stop=AsyncMock(),
    )
    return engine


class RetentionPolicyTests(unittest.TestCase):
    def test_observed_165_peak_requests_107_not_legacy_077_floor(self):
        state = RetentionState()
        for net, expected in ((0.73, 0), (1.22, 0), (1.50, 1), (1.65, 1.07), (1.10, 1.07)):
            state = advance(state, net)
            self.assertAlmostEqual(state.floor_usd, expected)

    def test_dollar_peak_must_also_reach_r_threshold_when_available(self):
        self.assertEqual(advance(RetentionState(), 1.65, risk=5).floor_usd, 0)
        self.assertEqual(advance(RetentionState(), 1.49).floor_usd, 0)
        self.assertEqual(advance(RetentionState(), 1.65, risk=0).floor_usd, 1.07)

    def test_continuing_winner_ratchets_without_a_fixed_cash_cap(self):
        state = RetentionState()
        for net, floor in ((1.65, 1.07), (2, 1.30), (2.8, 2.10), (4.2, 3.15), (3.8, 3.15)):
            state = advance(state, net)
            self.assertAlmostEqual(state.floor_usd, floor)
            self.assertLess(state.floor_usd, state.peak_net_usd)

    def test_valid_state_survives_serialization(self):
        state = advance(RetentionState(), 1.65)
        restored = RetentionState.from_dict(state.to_dict())
        self.assertEqual(advance(restored, 1.08), state)

    def test_partial_close_scales_floor_and_peak_to_remaining_volume(self):
        state = advance(RetentionState(), 1.65, volume=0.02)
        state = advance(state, 0.8, volume=0.01)
        self.assertAlmostEqual(state.floor_usd, 0.535)
        self.assertAlmostEqual(state.peak_net_usd, 0.825)
        self.assertAlmostEqual(advance(state, 1.4).floor_usd, 1.05)

    def test_volume_addition_does_not_reuse_previous_cash_peak(self):
        state = advance(RetentionState(), 1.65)
        state = advance(state, 0.2, volume=0.02)
        self.assertEqual(state.floor_usd, 0)
        self.assertEqual(state.peak_net_usd, 0.2)

    def test_invalid_observations_and_policy_cannot_change_valid_state(self):
        state = advance(RetentionState(), 1.65)
        for net in (float("nan"), float("inf"), None, "bad"):
            with self.subTest(net=net):
                self.assertEqual(advance(state, net), state)
        for policy in (
            replace(POLICY, profit_retention_keep_fraction=1.1),
            replace(POLICY, profit_retention_trigger_usd=0.9),
            replace(POLICY, profit_retention_mature_keep_fraction=0.5),
        ):
            self.assertEqual(advance(state, 2.0, policy=policy), state)
        self.assertEqual(advance(state, 2.0, volume=0), state)

    def test_corrupted_persisted_state_is_not_used(self):
        valid = advance(RetentionState(), 1.65).to_dict()
        for key, value in (("peak_net_usd", float("nan")), ("floor_usd", 99),
                           ("volume", -1), ("reference_volume", 0), ("floor_usd", "bad")):
            self.assertEqual(RetentionState.from_dict({**valid, key: value}), RetentionState())


class RetentionEngineTests(unittest.TestCase):
    def apply(self, engine, net, *, policy=POLICY, **kwargs):
        with patch("core.engine.settings", policy):
            asyncio.run(engine._apply_protections([position(net, **kwargs)]))

    def test_profitable_trade_stays_open_and_exits_only_after_floor_cross(self):
        engine = engine_with_cache()
        for net in (1.5, 1.65, 1.4, 1.08):
            self.apply(engine, net)
        engine.executor.close_position.assert_not_awaited()
        self.assertEqual(engine._profit_lock_levels[123], 1.07)
        self.apply(engine, 1.06)
        engine.executor.close_position.assert_awaited_once_with(
            123, expected_account=engine._active_account_identity,
        )
        engine._remember_strategy_close_reason.assert_awaited_once_with(123, "PROFIT_RETENTION")
        self.assertIn("floor $1.07", engine.db.log_trade.await_args.args[0]["reasoning"])
        self.assertNotIn(123, engine._profit_retention_states)

    def test_rejected_broker_stop_is_not_reported_as_armed_but_guard_still_exits(self):
        engine = engine_with_cache()
        engine.executor.lock_minimum_net_profit.return_value = ExecutionResult(
            False, 123, None, None, "Inside broker stop/freeze distance",
        )
        self.apply(engine, 1.65)
        self.assertNotIn(123, engine._profit_lock_tickets)
        self.assertNotIn(123, engine._profit_lock_levels)
        self.assertEqual(engine._profit_retention_states[123].floor_usd, 1.07)
        engine.executor.close_position.assert_not_awaited()
        self.apply(engine, 1.06)
        engine.executor.close_position.assert_awaited_once()

    def test_restart_restores_floor_even_during_pullback(self):
        cache = {}
        self.apply(engine_with_cache(cache), 1.65)
        restarted = engine_with_cache(cache)
        self.apply(restarted, 1.06)
        restarted._remember_strategy_close_reason.assert_awaited_once_with(123, "PROFIT_RETENTION")

    def test_real_sqlite_cache_restores_floor_across_database_instances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "retention.db")
            first = engine_with_cache()
            first.db = TradingDatabase(path)
            self.apply(first, 1.65)
            restarted = engine_with_cache()
            restarted.db = TradingDatabase(path)
            with patch("core.engine.settings", POLICY):
                restored = asyncio.run(restarted._update_profit_retention(position(1.08)))
            self.assertEqual(restored.floor_usd, 1.07)
            self.assertEqual(restored.peak_net_usd, 1.65)

    def test_runtime_config_reports_retention_policy(self):
        from ui.dashboard import get_config
        with patch("ui.dashboard.settings", POLICY):
            config = asyncio.run(get_config())
        for key in (
            "profit_retention_enabled", "profit_retention_trigger_usd",
            "profit_retention_trigger_r", "profit_retention_keep_fraction",
            "profit_retention_mature_trigger_r", "profit_retention_mature_keep_fraction",
            "profit_retention_min_headroom_usd",
        ):
            self.assertEqual(config[key], getattr(POLICY, key))

    def test_persisted_floor_is_account_scoped(self):
        cache = {}
        self.apply(engine_with_cache(cache), 1.65)
        other = engine_with_cache(cache)
        other._active_account_identity = {"login": 2002, "server": "test"}
        self.apply(other, 1.06)
        other.executor.close_position.assert_not_awaited()
        self.assertEqual(other._profit_retention_states[123].floor_usd, 0)

    def test_storage_failure_does_not_prevent_in_memory_retention_and_retries(self):
        engine = engine_with_cache()
        engine.db.get_cache.side_effect = RuntimeError("storage unavailable")
        self.apply(engine, 1.65)
        engine.db.set_cache.assert_not_awaited()
        self.assertEqual(engine._profit_retention_states[123].floor_usd, 1.07)
        engine.db.get_cache.side_effect = None
        engine.db.get_cache.return_value = None
        engine.db.set_cache.side_effect = None
        engine.db.set_cache.return_value = False
        self.apply(engine, 1.4)
        self.assertEqual(engine._profit_retention_states[123].floor_usd, 1.07)
        engine.db.set_cache.return_value = True
        self.apply(engine, 1.3)
        self.assertEqual(engine.db.set_cache.await_count, 2)
        self.assertEqual(engine._profit_retention_persisted[123].floor_usd, 1.07)

    def test_unchanged_peak_does_not_repeat_database_or_stop_writes(self):
        engine = engine_with_cache()
        self.apply(engine, 1.65)
        for _ in range(5):
            self.apply(engine, 1.4)
        self.assertEqual(engine.db.set_cache.await_count, 1)
        self.assertEqual(engine.executor.lock_minimum_net_profit.await_count, 1)

    def test_estimated_net_not_gross_profit_arms_retention(self):
        engine = engine_with_cache()
        p = position(1.4)
        p["profit"] = 1.65
        with patch("core.engine.settings", POLICY):
            asyncio.run(engine._apply_protections([p]))
        self.assertEqual(engine._profit_retention_states[123].floor_usd, 0)

    def test_giveback_disable_prevents_new_software_close(self):
        engine = engine_with_cache()
        self.apply(engine, 1.65)
        self.apply(engine, 1.06, policy=replace(POLICY, profit_giveback_enabled=False))
        engine.executor.close_position.assert_not_awaited()

    def test_retention_disable_preserves_legacy_behavior(self):
        engine = engine_with_cache()
        policy = replace(POLICY, profit_retention_enabled=False)
        self.apply(engine, 1.65, policy=policy)
        self.apply(engine, 1.06, policy=policy)
        engine.executor.close_position.assert_not_awaited()
        self.assertEqual(engine._profit_lock_levels[123], 0.77)

    def test_partial_close_rechecks_cash_value_of_existing_broker_stop(self):
        engine = engine_with_cache()
        self.apply(engine, 1.65, volume=0.02)
        remaining = position(0.8, volume=0.01)
        # Partial volume changes cash P/L, not price excursion or pip risk.
        remaining["profit_pips"] = 0.8 / 0.7 * 10
        with patch("core.engine.settings", POLICY):
            asyncio.run(engine._apply_protections([remaining]))
        engine.executor.close_position.assert_not_awaited()
        self.assertAlmostEqual(engine._profit_lock_levels[123], 0.535)

    def test_failed_close_keeps_floor_and_retries(self):
        engine = engine_with_cache()
        self.apply(engine, 1.65)
        engine.executor.close_position.return_value = ExecutionResult(False, 123, None, None, "busy")
        self.apply(engine, 1.06)
        self.assertEqual(engine._profit_retention_states[123].floor_usd, 1.07)
        engine._remember_strategy_close_reason.assert_not_awaited()
        engine.executor.close_position.return_value = ExecutionResult(True, 123, 178.5, 0.01, None)
        self.apply(engine, 1.05)
        self.assertEqual(engine.executor.close_position.await_count, 2)

    def test_fresh_broker_snapshot_overrides_stale_candidate_before_retention_exit(self):
        engine = engine_with_cache()
        self.apply(engine, 1.65)
        engine._refresh_protection_snapshot = AsyncMock(return_value=position(1.4))
        with patch("core.engine.settings", POLICY):
            asyncio.run(engine._apply_protections([position(0.8)], refresh_from_broker=True))
        engine.executor.close_position.assert_not_awaited()


class RetentionExecutorTests(unittest.TestCase):
    def test_buy_sell_tick_grid_and_swap_preserve_requested_net_floor(self):
        for side in (mt5.POSITION_TYPE_BUY, mt5.POSITION_TYPE_SELL):
            for swap in (-0.13, 0.0, 0.15):
                with self.subTest(side=side, swap=swap):
                    executor = MT5OrderExecutor(connection_manager=object())
                    pos = SimpleNamespace(ticket=123, symbol="TEST", type=side,
                                          price_open=100.0, sl=0.0, volume=0.01, swap=swap)
                    info = SimpleNamespace(point=0.01, digits=2, trade_tick_size=0.25,
                                           trade_stops_level=0, trade_freeze_level=0)
                    tick = (SimpleNamespace(bid=105, ask=105.1)
                            if side == mt5.POSITION_TYPE_BUY else
                            SimpleNamespace(bid=94.9, ask=95))
                    def profit(order_type, symbol, volume, entry, exit_price):
                        return (exit_price - entry) * (1 if side == mt5.POSITION_TYPE_BUY else -1)
                    with (
                        patch("execution.executor.mt5.positions_get", return_value=(pos,)),
                        patch("execution.executor.mt5.symbol_info", return_value=info),
                        patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
                        patch("execution.executor.mt5.order_calc_profit", side_effect=profit),
                        patch("execution.executor.configured_execution_cost_usd", return_value=0.07),
                        patch.object(executor, "modify_sl_tp", new_callable=AsyncMock,
                                     return_value=ExecutionResult(True, 123, None, None, None)) as modify,
                    ):
                        result = asyncio.run(executor.lock_minimum_net_profit(123, 1.07))
                    self.assertTrue(result.success, result.error)
                    sl = modify.await_args.kwargs["sl_price"]
                    self.assertAlmostEqual(sl / 0.25, round(sl / 0.25))
                    self.assertGreaterEqual(profit(side, "TEST", 0.01, 100, sl) + swap - 0.07, 1.07 - 1e-8)

    def test_invalid_floor_cannot_reach_broker(self):
        executor = MT5OrderExecutor(connection_manager=object())
        with patch("execution.executor.mt5.positions_get") as query:
            for floor in (float("nan"), float("inf"), -1, "bad", None):
                self.assertFalse(asyncio.run(executor.lock_minimum_net_profit(123, floor)).success)
        query.assert_not_called()

    def test_stop_restrictions_and_failed_profit_verification_never_modify(self):
        for scenario in ("freeze", "unverifiable", "below_floor", "existing_better_stop"):
            with self.subTest(scenario=scenario):
                executor = MT5OrderExecutor(connection_manager=object())
                pos = SimpleNamespace(ticket=123, symbol="TEST", type=mt5.POSITION_TYPE_BUY,
                                      price_open=100.0, sl=102 if scenario == "existing_better_stop" else 0,
                                      volume=0.01)
                info = SimpleNamespace(point=0.01, digits=2, trade_tick_size=0.01,
                                       trade_stops_level=0,
                                       trade_freeze_level=100 if scenario == "freeze" else 0)
                def profit(order_type, symbol, volume, entry, exit_price):
                    if abs(exit_price - entry) > 0.02:
                        if scenario == "unverifiable":
                            return None
                        if scenario == "below_floor":
                            return 0.01
                    return exit_price - entry
                with (
                    patch("execution.executor.mt5.positions_get", return_value=(pos,)),
                    patch("execution.executor.mt5.symbol_info", return_value=info),
                    patch("execution.executor.mt5.symbol_info_tick", return_value=SimpleNamespace(bid=101.65, ask=101.66)),
                    patch("execution.executor.mt5.order_calc_profit", side_effect=profit),
                    patch("execution.executor.configured_execution_cost_usd", return_value=0.07),
                    patch.object(executor, "modify_sl_tp", new_callable=AsyncMock) as modify,
                ):
                    result = asyncio.run(executor.lock_minimum_net_profit(123, 1.07))
                self.assertFalse(result.success)
                modify.assert_not_awaited()

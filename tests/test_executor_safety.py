import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import MetaTrader5 as mt5

from execution.executor import ExecutionResult, MAGIC_NUMBER, MT5OrderExecutor


class ExecutorSafetyTests(unittest.TestCase):
    def setUp(self):
        self.executor = MT5OrderExecutor(connection_manager=object())

    def test_decimal_lots_on_broker_step_are_accepted(self):
        info = SimpleNamespace(volume_min=0.01, volume_max=100.0, volume_step=0.01)
        with patch("execution.executor.mt5.symbol_info", return_value=info):
            for lot in (0.03, 0.30):
                with self.subTest(lot=lot):
                    self.assertIsNone(self.executor._validate_lot("TEST", lot))
            self.assertIn(
                "valid step multiple",
                self.executor._validate_lot("TEST", 0.035),
            )

    def test_operator_override_bypasses_spread_policy_but_requires_fresh_quote(self):
        info = SimpleNamespace(digits=5)
        tick = SimpleNamespace(time=time.time(), bid=1.0, ask=1.1)
        override_settings = SimpleNamespace(
            dry_run=True,
            max_tick_age_seconds=10.0,
        )
        with (
            patch("execution.executor.settings", override_settings),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch.object(self.executor, "_validate_account_identity", return_value=None),
            patch.object(self.executor, "_validate_symbol", return_value=None),
            patch.object(self.executor, "_validate_spread", return_value="too wide") as spread,
            patch.object(self.executor, "_validate_lot", return_value=None),
            patch.object(self.executor, "_validate_margin", return_value=None),
            patch.object(self.executor, "_validate_stops", return_value=None),
        ):
            result = self.executor._submit_open(
                "TEST",
                "BUY",
                0.01,
                0.9,
                1.2,
                "override",
                None,
                None,
                True,
            )

        self.assertTrue(result.success, result.error)
        spread.assert_not_called()

    def test_operator_override_cannot_open_a_close_only_market(self):
        info = SimpleNamespace(
            trade_mode=mt5.SYMBOL_TRADE_MODE_CLOSEONLY,
        )
        with (
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch.object(
                self.executor, "_validate_account_identity", return_value=None
            ),
            patch.object(self.executor, "_validate_symbol", return_value=None),
        ):
            result = self.executor._submit_open(
                "ETHUSD",
                "BUY",
                0.01,
                100.0,
                120.0,
                "override",
                None,
                None,
                True,
            )

        self.assertFalse(result.success)
        self.assertIn("close-only", result.error)

    def _assert_breakeven_does_not_worsen(self, position, tick):
        info = SimpleNamespace(point=0.001, digits=3)
        with (
            patch("execution.executor.mt5.positions_get", return_value=(position,)),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch.object(
                self.executor, "modify_sl_tp", new_callable=AsyncMock
            ) as modify,
        ):
            result = asyncio.run(
                self.executor.move_to_breakeven(position.ticket, buffer_pips=1.0)
            )

        self.assertFalse(result.success)
        self.assertIn("worsen", result.error.lower())
        modify.assert_not_awaited()

    def test_buy_breakeven_never_lowers_an_existing_better_stop(self):
        position = SimpleNamespace(
            ticket=101,
            symbol="USDJPY",
            type=mt5.POSITION_TYPE_BUY,
            price_open=100.000,
            sl=100.050,
        )
        self._assert_breakeven_does_not_worsen(
            position, SimpleNamespace(bid=100.100, ask=100.110)
        )

    def test_sell_breakeven_never_raises_an_existing_better_stop(self):
        position = SimpleNamespace(
            ticket=102,
            symbol="USDJPY",
            type=mt5.POSITION_TYPE_SELL,
            price_open=100.000,
            sl=99.950,
        )
        self._assert_breakeven_does_not_worsen(
            position, SimpleNamespace(bid=99.890, ask=99.900)
        )

    def test_cost_aware_profit_lock_uses_account_currency_floor(self):
        position = SimpleNamespace(
            ticket=104,
            symbol="USDJPY",
            type=mt5.POSITION_TYPE_BUY,
            price_open=100.0,
            price_current=100.3,
            volume=0.01,
            sl=99.5,
        )
        info = SimpleNamespace(
            point=0.001,
            digits=3,
            trade_stops_level=0,
            trade_freeze_level=0,
        )
        tick = SimpleNamespace(bid=100.3, ask=100.31)

        def profit(_order_type, _symbol, _volume, entry, exit_price):
            return (exit_price - entry)

        with (
            patch("execution.executor.mt5.positions_get", return_value=(position,)),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch("execution.executor.mt5.order_calc_profit", side_effect=profit),
            patch(
                "execution.executor.configured_execution_cost_usd",
                return_value=0.07,
            ),
            patch.object(
                self.executor,
                "modify_sl_tp",
                new_callable=AsyncMock,
                return_value=ExecutionResult(
                    True, 104, 100.3, 0.01, None, verified=True
                ),
            ) as modify,
        ):
            result = asyncio.run(
                self.executor.lock_minimum_net_profit(104, floor_usd=0.03)
            )

        self.assertTrue(result.success, result.error)
        self.assertAlmostEqual(modify.await_args.kwargs["sl_price"], 100.1)

    def test_identical_sl_tp_modification_is_idempotent_success(self):
        position = SimpleNamespace(
            ticket=103,
            symbol="USDJPY",
            type=mt5.POSITION_TYPE_BUY,
            price_open=100.000,
            price_current=100.100,
            volume=0.01,
            sl=100.0500000001,
            tp=100.2000000001,
        )
        info = SimpleNamespace(digits=3)
        with (
            patch("execution.executor.settings", SimpleNamespace(dry_run=False)),
            patch("execution.executor.mt5.positions_get", return_value=(position,)),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.order_check") as order_check,
            patch.object(self.executor, "_validate_account_identity", return_value=None),
        ):
            result = self.executor._submit_modify(
                position.ticket,
                sl_price=100.050,
                tp_price=100.200,
                expected_account={"login": 1},
            )

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertFalse(result.state_changed)
        self.assertIsNone(result.error)
        order_check.assert_not_called()

    def test_modify_reconciliation_verifies_both_retained_levels(self):
        raw = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
        parsed = ExecutionResult(
            True, 103, 100.1, 0.01, None, retcode=mt5.TRADE_RETCODE_DONE
        )
        retained = SimpleNamespace(
            ticket=103,
            price_current=100.1,
            volume=0.01,
            sl=100.05,
            tp=100.20,
        )
        info = SimpleNamespace(point=0.001, trade_tick_size=0.001)

        with patch(
            "execution.executor.mt5.positions_get",
            return_value=(retained,),
        ):
            result = self.executor._reconcile_modify_result(
                raw,
                parsed,
                ticket=103,
                requested_sl=100.05,
                requested_tp=100.20,
                symbol_info=info,
            )

        self.assertTrue(result.success, result.error)
        self.assertTrue(result.verified)

    def test_modify_reconciliation_rejects_accepted_but_ignored_levels(self):
        raw = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
        parsed = ExecutionResult(
            True, 103, 100.1, 0.01, None, retcode=mt5.TRADE_RETCODE_DONE
        )
        unchanged = SimpleNamespace(
            ticket=103,
            price_current=100.1,
            volume=0.01,
            sl=99.90,
            tp=100.20,
        )
        info = SimpleNamespace(point=0.001, trade_tick_size=0.001)

        with (
            patch(
                "execution.executor.mt5.positions_get",
                return_value=(unchanged,),
            ),
            patch("execution.executor.time.sleep"),
        ):
            result = self.executor._reconcile_modify_result(
                raw,
                parsed,
                ticket=103,
                requested_sl=100.05,
                requested_tp=100.20,
                symbol_info=info,
            )

        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertIn("did not retain", result.error)

    def test_final_rr_uses_tick_normalized_target(self):
        info = SimpleNamespace(
            trade_mode=mt5.SYMBOL_TRADE_MODE_FULL,
            digits=3,
            point=0.001,
            trade_tick_size=0.001,
        )
        tick = SimpleNamespace(time=time.time(), bid=1.099, ask=1.100)
        risk = SimpleNamespace(
            worst_entry=1.100,
            total_risk_usd=1.0,
            stop_risk_usd=0.9,
            slippage_reserve_usd=0.1,
            configured_cost_usd=0.0,
        )
        isolated = SimpleNamespace(
            dry_run=True,
            min_risk_reward_ratio=1.47,
        )

        def reward(_order_type, _symbol, _volume, entry, exit_price):
            return (exit_price - entry) * 1000.0

        with (
            patch("execution.executor.settings", isolated),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch(
                "execution.executor.mt5.order_calc_profit",
                side_effect=reward,
            ) as calc_profit,
            patch(
                "execution.executor.estimate_execution_risk",
                return_value=risk,
            ),
            patch.object(
                self.executor, "_validate_account_identity", return_value=None
            ),
            patch.object(self.executor, "_validate_symbol", return_value=None),
            patch.object(self.executor, "_validate_spread", return_value=None),
            patch.object(self.executor, "_validate_lot", return_value=None),
            patch.object(self.executor, "_validate_margin", return_value=None),
            patch.object(self.executor, "_validate_stops", return_value=None),
        ):
            result = self.executor._submit_open(
                "TEST",
                "BUY",
                0.01,
                1.099,
                1.10149,
                "normalized-rr",
                None,
                1.0,
                False,
            )

        self.assertFalse(result.success)
        self.assertIn("Final execution-adjusted R:R is 1.00", result.error)
        self.assertAlmostEqual(calc_profit.call_args.args[-1], 1.101)

    def test_open_passes_one_positional_request_to_serialized_mt5_proxy(self):
        info = SimpleNamespace(
            trade_mode=mt5.SYMBOL_TRADE_MODE_FULL,
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            filling_mode=2,
        )
        tick = SimpleNamespace(time=time.time(), bid=1.10000, ask=1.10010)
        check_result = SimpleNamespace(retcode=0, comment="Done")
        send_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
        reconciled = ExecutionResult(
            True, 123, 1.10010, 0.01, None, verified=True
        )

        with (
            patch(
                "execution.executor.settings",
                SimpleNamespace(dry_run=False, max_order_deviation_points=20),
            ),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch("execution.executor.mt5.positions_get", return_value=()),
            patch(
                "execution.executor.mt5.order_check",
                side_effect=lambda request: check_result,
            ) as order_check,
            patch(
                "execution.executor.mt5.order_send",
                side_effect=lambda request: send_result,
            ) as order_send,
            patch.object(self.executor, "_validate_account_identity", return_value=None),
            patch.object(self.executor, "_validate_symbol", return_value=None),
            patch.object(self.executor, "_validate_spread", return_value=None),
            patch.object(self.executor, "_validate_lot", return_value=None),
            patch.object(self.executor, "_validate_margin", return_value=None),
            patch.object(self.executor, "_validate_stops", return_value=None),
            patch.object(
                self.executor, "_reconcile_open_result", return_value=reconciled
            ),
        ):
            result = self.executor._submit_open(
                "EURUSD",
                "BUY",
                0.01,
                1.09900,
                1.10200,
                "keyword-open",
                {"login": 1},
                None,
                False,
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(len(order_check.call_args.args), 1)
        self.assertEqual(order_check.call_args.kwargs, {})
        self.assertEqual(len(order_send.call_args.args), 1)
        self.assertEqual(order_send.call_args.kwargs, {})
        self.assertEqual(
            order_send.call_args.args[0],
            order_check.call_args.args[0],
        )

    def test_modify_passes_one_positional_request_to_serialized_mt5_proxy(self):
        position = SimpleNamespace(
            ticket=456,
            symbol="EURUSD",
            type=mt5.POSITION_TYPE_BUY,
            price_open=1.10000,
            price_current=1.10100,
            volume=0.01,
            sl=1.09900,
            tp=1.10300,
        )
        info = SimpleNamespace(
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
        )
        tick = SimpleNamespace(bid=1.10100, ask=1.10110)
        check_result = SimpleNamespace(retcode=0, comment="Done")
        send_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
        reconciled = ExecutionResult(
            True, 456, 1.10100, 0.01, None, verified=True
        )

        with (
            patch("execution.executor.settings", SimpleNamespace(dry_run=False)),
            patch("execution.executor.mt5.positions_get", return_value=(position,)),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch(
                "execution.executor.mt5.order_check",
                side_effect=lambda request: check_result,
            ) as order_check,
            patch(
                "execution.executor.mt5.order_send",
                side_effect=lambda request: send_result,
            ) as order_send,
            patch.object(self.executor, "_validate_account_identity", return_value=None),
            patch.object(self.executor, "_validate_stops", return_value=None),
            patch.object(
                self.executor, "_reconcile_modify_result", return_value=reconciled
            ),
        ):
            result = self.executor._submit_modify(
                position.ticket,
                sl_price=1.09950,
                tp_price=1.10350,
                expected_account={"login": 1},
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(len(order_check.call_args.args), 1)
        self.assertEqual(order_check.call_args.kwargs, {})
        self.assertEqual(len(order_send.call_args.args), 1)
        self.assertEqual(order_send.call_args.kwargs, {})

    def test_close_passes_one_positional_request_to_serialized_mt5_proxy(self):
        position = SimpleNamespace(
            ticket=789,
            symbol="EURUSD",
            type=mt5.POSITION_TYPE_BUY,
            volume=0.01,
        )
        info = SimpleNamespace(filling_mode=2, volume_step=0.01)
        tick = SimpleNamespace(bid=1.10100, ask=1.10110)
        check_result = SimpleNamespace(retcode=0, comment="Done")
        send_result = SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE)
        reconciled = ExecutionResult(
            True, 789, 1.10100, 0.01, None, verified=True
        )

        with (
            patch(
                "execution.executor.settings",
                SimpleNamespace(dry_run=False, max_order_deviation_points=20),
            ),
            patch("execution.executor.mt5.symbol_info", return_value=info),
            patch("execution.executor.mt5.symbol_info_tick", return_value=tick),
            patch(
                "execution.executor.mt5.order_check",
                side_effect=lambda request: check_result,
            ) as order_check,
            patch(
                "execution.executor.mt5.order_send",
                side_effect=lambda request: send_result,
            ) as order_send,
            patch.object(self.executor, "_validate_account_identity", return_value=None),
            patch.object(
                self.executor, "_reconcile_close_result", return_value=reconciled
            ),
        ):
            result = self.executor._send_close_request(
                position,
                position.volume,
                "keyword-close",
                {"login": 1},
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(len(order_check.call_args.args), 1)
        self.assertEqual(order_check.call_args.kwargs, {})
        self.assertEqual(len(order_send.call_args.args), 1)
        self.assertEqual(order_send.call_args.kwargs, {})

    def test_partial_open_is_bound_to_actual_position_volume(self):
        raw = SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE_PARTIAL,
            order=501,
            deal=601,
            price=1.1001,
            volume=0.04,
            comment="partial",
        )
        position = SimpleNamespace(
            ticket=501,
            symbol="EURUSD",
            type=mt5.POSITION_TYPE_BUY,
            magic=MAGIC_NUMBER,
            volume=0.04,
            price_open=1.10012,
        )
        parsed = self.executor._parse_result(raw, "BUY")
        with patch("execution.executor.mt5.positions_get", return_value=(position,)):
            result = self.executor._reconcile_open_result(
                raw,
                parsed,
                symbol="EURUSD",
                action="BUY",
                requested_volume=0.10,
                before_positions=(),
                symbol_info=SimpleNamespace(volume_step=0.01),
            )

        self.assertTrue(result.success)
        self.assertTrue(result.partial)
        self.assertTrue(result.verified)
        self.assertEqual(result.ticket, 501)
        self.assertEqual(result.volume, 0.04)
        self.assertEqual(result.requested_volume, 0.10)

    def test_partial_full_close_is_not_reported_as_closed(self):
        position = SimpleNamespace(
            ticket=701,
            symbol="EURUSD",
            type=mt5.POSITION_TYPE_BUY,
            volume=0.10,
        )
        remaining = SimpleNamespace(ticket=701, volume=0.06)
        raw = SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE_PARTIAL,
            order=801,
            deal=901,
            price=1.1000,
            volume=0.04,
            comment="partial",
        )
        parsed = self.executor._parse_result(raw, "CLOSE")
        with patch("execution.executor.mt5.positions_get", return_value=(remaining,)):
            result = self.executor._reconcile_close_result(
                raw,
                parsed,
                pos=position,
                requested_volume=0.10,
                symbol_info=SimpleNamespace(volume_step=0.01),
            )

        self.assertFalse(result.success)
        self.assertTrue(result.partial)
        self.assertTrue(result.verified)
        self.assertTrue(result.state_changed)
        self.assertAlmostEqual(result.volume, 0.04)
        self.assertAlmostEqual(result.remaining_volume, 0.06)
        self.assertIn("remaining", result.error)


if __name__ == "__main__":
    unittest.main()

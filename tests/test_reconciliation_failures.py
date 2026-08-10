import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from database.reconciliation import BrokerHistoryReconciler


class BrokerHistoryFailureTests(unittest.TestCase):
    def setUp(self):
        self.reconciler = BrokerHistoryReconciler(strategy_magic=202600)

    def test_none_deal_history_raises_instead_of_becoming_empty_history(self):
        with (
            patch("database.reconciliation.mt5.history_deals_get", return_value=None),
            patch("database.reconciliation.mt5.positions_get") as positions_get,
            patch("database.reconciliation.mt5.last_error", return_value=(1, "history unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "history_deals_get failed"):
                self.reconciler.fetch()

        positions_get.assert_not_called()

    def test_none_position_snapshot_raises_instead_of_marking_positions_closed(self):
        with (
            patch("database.reconciliation.mt5.history_deals_get", return_value=()),
            patch("database.reconciliation.mt5.positions_get", return_value=None),
            patch("database.reconciliation.mt5.last_error", return_value=(2, "positions unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "positions_get failed"):
                self.reconciler.fetch()

    def test_fetch_queries_in_detected_broker_clock_domain(self):
        now_epoch = 1_700_000_000.0
        offset = 3 * 3600
        reconciler = BrokerHistoryReconciler(202600, symbols=["CADJPY"])
        with (
            patch("database.reconciliation.time.time", return_value=now_epoch),
            patch(
                "database.reconciliation.mt5.symbol_info_tick",
                return_value=SimpleNamespace(time=now_epoch + offset),
            ),
            patch(
                "database.reconciliation.mt5.history_deals_get", return_value=()
            ) as history_get,
            patch("database.reconciliation.mt5.positions_get", return_value=()),
        ):
            rows = reconciler.fetch(days=1)

        self.assertEqual(rows, [])
        date_from, date_to = history_get.call_args.args
        self.assertEqual(date_to, datetime.fromtimestamp(now_epoch + offset, timezone.utc))
        self.assertEqual((date_to - date_from).total_seconds(), 24 * 3600)
        self.assertEqual(reconciler.server_offset_seconds, offset)


if __name__ == "__main__":
    unittest.main()

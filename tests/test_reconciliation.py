import unittest
from collections import namedtuple

from database.reconciliation import aggregate_closed_positions


Deal = namedtuple(
    "Deal",
    "position_id time time_msc entry type volume price profit commission swap fee magic symbol reason",
)


class ReconciliationTests(unittest.TestCase):
    def test_costs_and_direction_are_reconstructed(self):
        deals = [
            Deal(7, 100, 100000, 0, 0, 0.01, 1.1000, 0, -0.04, 0, 0, 202600, "EURUSD", 3),
            Deal(7, 200, 200000, 1, 1, 0.01, 1.1020, 2.0, -0.04, -0.02, 0, 202600, "EURUSD", 5),
        ]
        rows = aggregate_closed_positions(deals, 202600)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction"], "BUY")
        self.assertAlmostEqual(rows[0]["net_profit"], 1.90)
        self.assertEqual(rows[0]["close_reason"], "TAKE_PROFIT")

    def test_open_or_foreign_positions_are_excluded(self):
        open_deal = Deal(8, 100, 100000, 0, 0, 0.01, 1.1, 0, 0, 0, 0, 202600, "EURUSD", 3)
        foreign_in = Deal(9, 100, 100000, 0, 0, 0.01, 1.1, 0, 0, 0, 0, 99, "EURUSD", 3)
        foreign_out = Deal(9, 200, 200000, 1, 1, 0.01, 1.2, 1, 0, 0, 0, 99, "EURUSD", 3)
        self.assertEqual(aggregate_closed_positions([open_deal, foreign_in, foreign_out], 202600), [])

    def test_broker_clock_offset_is_removed_from_persisted_times(self):
        utc_open = 1_700_000_000
        offset = 3 * 3600
        deals = [
            Deal(10, utc_open + offset, 100000, 0, 0, 0.01, 115.47, 0, 0, 0, 0, 202600, "CADJPY", 3),
            Deal(10, utc_open + 60 + offset, 200000, 1, 1, 0.01, 115.37, -0.7, 0, 0, 0, 202600, "CADJPY", 4),
        ]

        rows = aggregate_closed_positions(
            deals, 202600, server_offset_seconds=offset
        )

        self.assertEqual(rows[0]["open_time"], "2023-11-14T22:13:20+00:00")
        self.assertEqual(rows[0]["close_time"], "2023-11-14T22:14:20+00:00")

    def test_profitable_stop_activation_is_labeled_as_protective(self):
        deals = [
            Deal(11, 100, 100000, 0, 1, 0.01, 182.700, 0, 0, 0, 0, 202600, "EURJPY", 3),
            Deal(11, 200, 200000, 1, 0, 0.01, 182.550, 1.0, 0, 0, 0, 202600, "EURJPY", 4),
        ]

        rows = aggregate_closed_positions(deals, 202600)

        self.assertEqual(rows[0]["close_reason"], "PROTECTIVE_STOP")
        self.assertGreater(rows[0]["net_profit"], 0)

    def test_losing_stop_activation_remains_stop_loss(self):
        deals = [
            Deal(12, 100, 100000, 0, 0, 0.01, 1.1000, 0, 0, 0, 0, 202600, "EURUSD", 3),
            Deal(12, 200, 200000, 1, 1, 0.01, 1.0980, -1.0, 0, 0, 0, 202600, "EURUSD", 4),
        ]

        rows = aggregate_closed_positions(deals, 202600)

        self.assertEqual(rows[0]["close_reason"], "STOP_LOSS")
        self.assertLess(rows[0]["net_profit"], 0)


if __name__ == "__main__":
    unittest.main()

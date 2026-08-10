import unittest
import math

from mt5.timebase import (
    broker_tick_age_seconds,
    infer_positive_server_offset_seconds,
    normalized_broker_epoch,
)


class MT5TimebaseTests(unittest.TestCase):
    def test_detects_positive_whole_hour_server_offset(self):
        now = 1_700_000_000.0
        self.assertEqual(
            infer_positive_server_offset_seconds(now + 3 * 3600 + 2, now_epoch=now),
            3 * 3600,
        )

    def test_old_tick_is_not_misclassified_as_a_server_offset(self):
        now = 1_700_000_000.0
        self.assertEqual(
            infer_positive_server_offset_seconds(now - 3600, now_epoch=now),
            0,
        )
        self.assertEqual(broker_tick_age_seconds(now - 30, now_epoch=now), 30)

    def test_normalizes_broker_epoch_and_tick_age(self):
        now = 1_700_000_000.0
        raw = now + 3 * 3600 - 4
        offset = infer_positive_server_offset_seconds(raw, now_epoch=now)
        self.assertEqual(normalized_broker_epoch(raw, offset), now - 4)
        self.assertEqual(
            broker_tick_age_seconds(raw, now_epoch=now, offset_seconds=offset),
            4,
        )

    def test_remembered_offset_exposes_a_stale_broker_clock_quote(self):
        now = 1_700_000_000.0
        symbol = "OFFSETCACHE_TEST"
        self.assertEqual(
            broker_tick_age_seconds(
                now + 3 * 3600 - 2,
                now_epoch=now,
                symbol=symbol,
            ),
            2,
        )
        self.assertEqual(
            broker_tick_age_seconds(
                now + 3 * 3600 - 60,
                now_epoch=now,
                symbol=symbol,
            ),
            60,
        )

    def test_ambiguous_future_quote_fails_closed(self):
        now = 1_700_000_000.0
        self.assertTrue(
            math.isinf(
                broker_tick_age_seconds(
                    now + 47 * 60,
                    now_epoch=now,
                    symbol="AMBIGUOUS_FUTURE_TEST",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()

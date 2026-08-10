import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from mt5.data import MT5DataReader


class _Connection:
    async def is_connected(self):
        return True


def _rate(epoch, close):
    return {
        "time": epoch,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "tick_volume": 1,
        "spread": 1,
        "real_volume": 0,
    }


class MT5DataReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = MT5DataReader(_Connection())
        self.now = datetime(2026, 7, 13, 11, 23, tzinfo=timezone.utc)

    def test_utc_fetch_excludes_forming_and_future_bars(self):
        rates = [
            _rate(int(datetime(2026, 7, 13, 11, 5, tzinfo=timezone.utc).timestamp()), 1.0),
            _rate(int(datetime(2026, 7, 13, 11, 10, tzinfo=timezone.utc).timestamp()), 2.0),
            _rate(int(datetime(2026, 7, 13, 11, 15, tzinfo=timezone.utc).timestamp()), 3.0),
            _rate(int(datetime(2026, 7, 13, 11, 20, tzinfo=timezone.utc).timestamp()), 4.0),
            _rate(int(datetime(2026, 7, 13, 12, 55, tzinfo=timezone.utc).timestamp()), 5.0),
        ]

        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.copy_rates_from", return_value=rates) as fetch,
        ):
            frame = asyncio.run(self.reader.get_ohlcv("USDJPY", "M5", count=3))

        self.assertEqual(frame["close"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(frame.attrs["source"], "broker_clock")
        self.assertFalse(frame.attrs["is_stale"])
        self.assertEqual(fetch.call_args.args[2], self.now)

    def test_stale_completed_data_is_marked(self):
        rates = [
            _rate(int(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc).timestamp()), 1.0),
        ]

        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.copy_rates_from", return_value=rates),
            patch("mt5.data.mt5.copy_rates_from_pos", return_value=None),
        ):
            frame = asyncio.run(self.reader.get_ohlcv("USDJPY", "M5", count=10))

        self.assertTrue(frame.attrs["is_stale"])
        self.assertGreater(frame.attrs["age_after_close_seconds"], 60 * 60)

    def test_stale_utc_cache_uses_fresher_position_data(self):
        utc_rates = [
            _rate(int(datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc).timestamp()), 1.0),
        ]
        positional_rates = [
            _rate(int(datetime(2026, 7, 13, 11, 15, tzinfo=timezone.utc).timestamp()), 2.0),
            _rate(int(datetime(2026, 7, 13, 11, 20, tzinfo=timezone.utc).timestamp()), 3.0),
        ]

        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.copy_rates_from", return_value=utc_rates),
            patch("mt5.data.mt5.copy_rates_from_pos", return_value=positional_rates),
        ):
            frame = asyncio.run(self.reader.get_ohlcv("USDJPY", "M5", count=10))

        self.assertEqual(frame["close"].tolist(), [2.0])
        self.assertEqual(frame.attrs["source"], "position_fallback")
        self.assertFalse(frame.attrs["is_stale"])

    def test_position_fallback_is_filtered_by_utc_cutoff(self):
        rates = [
            _rate(int(datetime(2026, 7, 13, 11, 15, tzinfo=timezone.utc).timestamp()), 1.0),
            _rate(int(datetime(2026, 7, 13, 12, 55, tzinfo=timezone.utc).timestamp()), 2.0),
        ]

        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.copy_rates_from", return_value=None),
            patch("mt5.data.mt5.copy_rates_from_pos", return_value=rates),
        ):
            frame = asyncio.run(self.reader.get_ohlcv("USDJPY", "M5", count=10))

        self.assertEqual(frame["close"].tolist(), [1.0])
        self.assertEqual(frame.attrs["source"], "position_fallback")

    def test_broker_server_offset_is_normalized_before_bar_filtering(self):
        offset = 3 * 60 * 60
        tick = SimpleNamespace(time=self.now.timestamp() + offset)
        rates = [
            _rate(int(datetime(2026, 7, 13, 14, 5, tzinfo=timezone.utc).timestamp()), 1.0),
            _rate(int(datetime(2026, 7, 13, 14, 10, tzinfo=timezone.utc).timestamp()), 2.0),
            _rate(int(datetime(2026, 7, 13, 14, 15, tzinfo=timezone.utc).timestamp()), 3.0),
            _rate(int(datetime(2026, 7, 13, 14, 20, tzinfo=timezone.utc).timestamp()), 4.0),
        ]

        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.symbol_info_tick", return_value=tick),
            patch("mt5.data.mt5.copy_rates_from", return_value=rates) as fetch,
        ):
            frame = asyncio.run(self.reader.get_ohlcv("USDCAD", "M5", count=3))

        expected_anchor = datetime(2026, 7, 13, 14, 23, tzinfo=timezone.utc)
        self.assertEqual(fetch.call_args.args[2], expected_anchor)
        self.assertEqual(frame["close"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(
            frame.iloc[-1]["time"],
            datetime(2026, 7, 13, 11, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(frame.attrs["server_offset_seconds"], offset)
        self.assertFalse(frame.attrs["is_stale"])

    def test_frozen_broker_bars_are_rebuilt_from_current_tick_history(self):
        offset = 3 * 60 * 60
        tick = SimpleNamespace(time=self.now.timestamp() + offset)
        stale_rates = [
            _rate(int(datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc).timestamp()), 1.0),
        ]
        ticks = []
        for minute in range(0, 80):
            epoch = int(datetime(2026, 7, 13, 13, 0, tzinfo=timezone.utc).timestamp()) + minute * 60
            price = 100.0 + minute / 100.0
            ticks.append({
                "time": epoch,
                "time_msc": epoch * 1000,
                "bid": price,
                "ask": price + 0.01,
                "last": 0.0,
                "volume": 1.0,
                "volume_real": 0.0,
            })

        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.symbol_info_tick", return_value=tick),
            patch("mt5.data.mt5.symbol_info", return_value=SimpleNamespace(point=0.01)),
            patch("mt5.data.mt5.copy_rates_from", return_value=stale_rates),
            patch("mt5.data.mt5.copy_rates_from_pos", return_value=stale_rates),
            patch("mt5.data.mt5.copy_ticks_range", return_value=ticks) as tick_fetch,
        ):
            frame = asyncio.run(self.reader.get_ohlcv("USDJPY", "M5", count=260))

        self.assertTrue(tick_fetch.called)
        self.assertEqual(frame.attrs["source"], "tick_rebuild")
        self.assertTrue(frame.attrs["rebuilt_from_ticks"])
        self.assertFalse(frame.attrs["is_stale"])
        self.assertEqual(
            frame.iloc[-1]["time"],
            datetime(2026, 7, 13, 11, 15, tzinfo=timezone.utc),
        )

    def test_live_tick_timestamp_is_normalized_to_utc(self):
        offset = 3 * 60 * 60
        raw_epoch = self.now.timestamp() + offset
        tick = SimpleNamespace(
            time=raw_epoch,
            time_msc=raw_epoch * 1000,
            bid=1.0,
            ask=1.1,
            last=0.0,
            volume=1.0,
        )
        with (
            patch("mt5.data._utc_now", return_value=self.now),
            patch("mt5.data.mt5.symbol_select", return_value=True),
            patch("mt5.data.mt5.symbol_info_tick", return_value=tick),
        ):
            result = asyncio.run(self.reader.get_live_tick("USDCAD"))

        self.assertEqual(result["time"], self.now.timestamp())
        self.assertEqual(result["time_msc"], self.now.timestamp() * 1000)


if __name__ == "__main__":
    unittest.main()

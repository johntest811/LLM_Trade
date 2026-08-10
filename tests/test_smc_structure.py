import unittest

import pandas as pd

from market_structure.smc import SMCAnalyzer


def _candles(*closes):
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-07-28 10:00:00", periods=len(closes), freq="5min"),
            "close": closes,
        }
    )


SWING_HIGHS = [{"price": 1.1000}]
SWING_LOWS = [{"price": 1.0900}]


class StructureTransitionTests(unittest.TestCase):
    def test_bullish_bos_is_emitted_only_on_first_close_above_level(self):
        first_cross = SMCAnalyzer.detect_bos_choch(
            _candles(1.0990, 1.1010),
            SWING_HIGHS,
            SWING_LOWS,
            "BULLISH",
        )
        already_broken = SMCAnalyzer.detect_bos_choch(
            _candles(1.1010, 1.1020),
            SWING_HIGHS,
            SWING_LOWS,
            "BULLISH",
        )

        self.assertEqual(
            [(event["type"], event["direction"]) for event in first_cross],
            [("BOS", "BULLISH")],
        )
        self.assertEqual(already_broken, [])

    def test_bearish_bos_is_emitted_only_on_first_close_below_level(self):
        first_cross = SMCAnalyzer.detect_bos_choch(
            _candles(1.0910, 1.0890),
            SWING_HIGHS,
            SWING_LOWS,
            "BEARISH",
        )
        already_broken = SMCAnalyzer.detect_bos_choch(
            _candles(1.0890, 1.0880),
            SWING_HIGHS,
            SWING_LOWS,
            "BEARISH",
        )

        self.assertEqual(
            [(event["type"], event["direction"]) for event in first_cross],
            [("BOS", "BEARISH")],
        )
        self.assertEqual(already_broken, [])

    def test_choch_is_emitted_only_when_latest_close_crosses_level(self):
        bullish_choch = SMCAnalyzer.detect_bos_choch(
            _candles(1.1000, 1.1010),
            SWING_HIGHS,
            SWING_LOWS,
            "BEARISH",
        )
        bearish_choch = SMCAnalyzer.detect_bos_choch(
            _candles(1.0900, 1.0890),
            SWING_HIGHS,
            SWING_LOWS,
            "BULLISH",
        )

        self.assertEqual(
            [(event["type"], event["direction"]) for event in bullish_choch],
            [("CHOCH", "BULLISH")],
        )
        self.assertEqual(
            [(event["type"], event["direction"]) for event in bearish_choch],
            [("CHOCH", "BEARISH")],
        )

    def test_one_candle_cannot_establish_a_structure_transition(self):
        events = SMCAnalyzer.detect_bos_choch(
            _candles(1.1010),
            SWING_HIGHS,
            SWING_LOWS,
            "BULLISH",
        )

        self.assertEqual(events, [])

    def test_tiny_bos_breach_is_rejected_until_displaced(self):
        weak = _candles(1.0910, 1.08995)
        weak["atr_14"] = 0.001
        strong = _candles(1.0910, 1.08989)
        strong["atr_14"] = 0.001

        weak_events = SMCAnalyzer.detect_bos_choch(
            weak,
            SWING_HIGHS,
            SWING_LOWS,
            "BEARISH",
            min_displacement_atr=0.10,
        )
        strong_events = SMCAnalyzer.detect_bos_choch(
            strong,
            SWING_HIGHS,
            SWING_LOWS,
            "BEARISH",
            min_displacement_atr=0.10,
        )

        self.assertEqual(weak_events, [])
        self.assertEqual(
            [(event["type"], event["direction"]) for event in strong_events],
            [("BOS", "BEARISH")],
        )


if __name__ == "__main__":
    unittest.main()

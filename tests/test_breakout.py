import unittest

from market_structure.breakout import BreakoutDetector


class BreakoutDetectorTests(unittest.TestCase):
    def test_first_completed_close_above_resistance_emits_breakout(self):
        result = BreakoutDetector.detect(
            101.1, 98.0, 101.0, previous_close=100.9
        )

        self.assertIn("BULLISH BREAKOUT", result)

    def test_repeated_close_above_resistance_is_not_fresh_breakout(self):
        result = BreakoutDetector.detect(
            101.2, 98.0, 101.0, previous_close=101.1
        )

        self.assertEqual(result, "None")

    def test_first_completed_close_below_support_emits_breakout(self):
        result = BreakoutDetector.detect(
            97.9, 98.0, 101.0, previous_close=98.1
        )

        self.assertIn("BEARISH BREAKOUT", result)

    def test_repeated_close_below_support_is_not_fresh_breakout(self):
        result = BreakoutDetector.detect(
            97.8, 98.0, 101.0, previous_close=97.9
        )

        self.assertEqual(result, "None")

    def test_legacy_call_remains_compatible(self):
        result = BreakoutDetector.detect(101.1, 98.0, 101.0)

        self.assertIn("BULLISH BREAKOUT", result)

    def test_tiny_level_breach_is_not_a_confirmed_breakout(self):
        result = BreakoutDetector.detect(
            97.95,
            98.0,
            101.0,
            previous_close=98.1,
            atr=1.0,
            min_displacement_atr=0.10,
        )

        self.assertEqual(result, "None")

    def test_displaced_level_break_is_confirmed(self):
        result = BreakoutDetector.detect(
            97.89,
            98.0,
            101.0,
            previous_close=98.1,
            atr=1.0,
            min_displacement_atr=0.10,
        )

        self.assertIn("BEARISH BREAKOUT", result)


if __name__ == "__main__":
    unittest.main()

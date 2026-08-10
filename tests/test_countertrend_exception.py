import unittest
from types import SimpleNamespace
from unittest.mock import patch

from risk.manager import RiskManager


def _analysis(trend, adx=40.0, rsi=50.0):
    return {
        "market_structure": {"trend": trend},
        "indicators": {"adx_14": adx, "rsi_14": rsi},
    }


class StrongCountertrendExceptionTests(unittest.TestCase):
    def setUp(self):
        self.enabled = SimpleNamespace(
            allow_strong_countertrend_entries=True,
            countertrend_min_confidence=0.85,
            countertrend_min_adx=34.5,
            countertrend_min_confluence=50.0,
            countertrend_sell_min_rsi=15.0,
            countertrend_buy_max_rsi=85.0,
        )

    def test_allows_high_confidence_strong_lower_timeframe_alignment(self):
        with patch("risk.manager.settings", self.enabled):
            allowed, detail = RiskManager._strong_countertrend_exception(
                "SELL",
                {"confidence": 0.85},
                _analysis("BEARISH", 52.5),
                _analysis("BEARISH"),
                _analysis("BEARISH"),
            )

        self.assertTrue(allowed, detail)

    def test_rejects_below_countertrend_confidence_even_if_momentum_is_strong(self):
        with patch("risk.manager.settings", self.enabled):
            allowed, detail = RiskManager._strong_countertrend_exception(
                "SELL",
                {"confidence": 0.75},
                _analysis("BEARISH", 52.5),
                _analysis("BEARISH"),
                _analysis("BEARISH"),
            )

        self.assertFalse(allowed)
        self.assertIn("75%/85%", detail)

    def test_rejects_misaligned_lower_timeframe_or_weak_adx(self):
        with patch("risk.manager.settings", self.enabled):
            misaligned, _ = RiskManager._strong_countertrend_exception(
                "SELL",
                {"confidence": 0.90},
                _analysis("BEARISH", 50.0),
                _analysis("NEUTRAL"),
                _analysis("BEARISH"),
            )
            weak, detail = RiskManager._strong_countertrend_exception(
                "SELL",
                {"confidence": 0.90},
                _analysis("BEARISH", 25.0),
                _analysis("BEARISH"),
                _analysis("BEARISH"),
            )

        self.assertFalse(misaligned)
        self.assertFalse(weak)
        self.assertIn("25.0/34.5", detail)

    def test_allows_the_observed_adx_boundary_but_rejects_extreme_rsi_chasing(self):
        with patch("risk.manager.settings", self.enabled):
            boundary, _ = RiskManager._strong_countertrend_exception(
                "SELL",
                {"confidence": 0.85},
                _analysis("BEARISH", 34.9, 16.0),
                _analysis("BEARISH"),
                _analysis("BEARISH"),
            )
            exhausted_sell, sell_detail = RiskManager._strong_countertrend_exception(
                "SELL",
                {"confidence": 0.90},
                _analysis("BEARISH", 45.0, 14.9),
                _analysis("BEARISH"),
                _analysis("BEARISH"),
            )
            exhausted_buy, buy_detail = RiskManager._strong_countertrend_exception(
                "BUY",
                {"confidence": 0.90},
                _analysis("BULLISH", 45.0, 85.1),
                _analysis("BULLISH"),
                _analysis("BULLISH"),
            )

        self.assertTrue(boundary)
        self.assertFalse(exhausted_sell)
        self.assertIn("14.9/>=15.0", sell_detail)
        self.assertFalse(exhausted_buy)
        self.assertIn("85.1/<=85.0", buy_detail)


if __name__ == "__main__":
    unittest.main()

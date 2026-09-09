from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import pandas as pd

from core.analysis_engine import MarketAnalysisEngine
from trading_indicators.calculations import TechnicalIndicators


def bars():
    values = [100 + i * .01 + (i % 7) * .02 for i in range(240)]
    frame = pd.DataFrame({"time": pd.date_range("2032-01-01", periods=240, freq="5min", tz="UTC"),
                          "open": values, "close": values,
                          "high": [v + .3 for v in values], "low": [v - .3 for v in values],
                          "tick_volume": [100] * 240})
    frame.attrs.update(pip_size=.1, asset_class="COMMODITY/CFD")
    return frame


class AnalysisCacheTests(unittest.TestCase):
    def test_unchanged_frames_hit_cache_but_annotations_are_isolated(self):
        engine, frame = MarketAnalysisEngine(), bars()
        with patch.object(TechnicalIndicators, "calculate_all", wraps=TechnicalIndicators.calculate_all) as calculate:
            first = engine.analyze("TEST", "M5", frame)
            first["market_structure"]["retest_continuation"] = {"direction": "BULLISH"}
            second = engine.analyze("TEST", "M5", frame.copy())
            self.assertEqual(calculate.call_count, 1)
            self.assertNotIn("retest_continuation", second["market_structure"])

    def test_corrections_to_older_bars_and_metadata_invalidate_cache(self):
        engine, frame = MarketAnalysisEngine(), bars()
        with patch.object(TechnicalIndicators, "calculate_all", wraps=TechnicalIndicators.calculate_all) as calculate:
            first = engine.analyze("TEST", "M5", frame)
            frame.loc[100, "high"] += 5
            engine.analyze("TEST", "M5", frame)
            frame.attrs["pip_size"] = .01
            changed = engine.analyze("TEST", "M5", frame)
            frame.attrs["asset_class"] = "INDEX/CFD"
            final = engine.analyze("TEST", "M5", frame)
            self.assertEqual(calculate.call_count, 4)
            self.assertNotEqual(first["indicators"]["atr_14_pips"], changed["indicators"]["atr_14_pips"])
            self.assertEqual(final["asset_class"], "INDEX/CFD")

    def test_latest_candle_correction_and_stale_frame(self):
        engine, frame = MarketAnalysisEngine(), bars()
        first = engine.analyze("TEST", "M5", frame)
        frame.loc[239, "close"] += .2
        second = engine.analyze("TEST", "M5", frame)
        self.assertNotEqual(first["indicators"]["current_price"], second["indicators"]["current_price"])
        frame.attrs["is_stale"] = True
        self.assertIsNone(engine.analyze("TEST", "M5", frame))

    def test_previous_cache_is_bounded_and_thread_safe(self):
        engine, frame = MarketAnalysisEngine(max_cache_entries=2), bars()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda symbol: engine.analyze(symbol, "M5", frame), ["A", "B", "C", "D"]))
        self.assertTrue(all(results))
        self.assertLessEqual(len(engine._cache), 2)
        for symbol in list(key[0] for key in engine._cache):
            engine.analyze(symbol, "M5", frame.iloc[:-1])
        engine.analyze("E", "M5", frame)
        self.assertLessEqual(len(engine._previous_cache), 2)
        self.assertTrue(set(engine._previous_cache).issubset(engine._cache))


if __name__ == "__main__":
    unittest.main()

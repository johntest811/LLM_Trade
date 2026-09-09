import copy
import asyncio
from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import pandas as pd

from app_config.settings import settings
from core.evidence import build_evidence_ids, permitted_entry_actions
from core.opportunities import classify_reversal_watch, screen_opportunities
from core.engine import TradingEngine


def fixture(direction="BUY", scale=1, year=2032):
    closes = [100, 99.8, 99.6, 99.7, 99.9, 100.3]
    opens = [100, 99.9, 99.7, 99.5, 99.65, 99.9]
    frame = pd.DataFrame({"time": pd.date_range(f"{year}-03-01T12:00:00Z", periods=6, freq="5min"),
                          "open": opens, "close": closes,
                          "high": [max(o, c) + .1 for o, c in zip(opens, closes)],
                          "low": [min(o, c) - .1 for o, c in zip(opens, closes)]})
    if direction == "SELL":
        frame[["open", "close"]] = 200 - frame[["open", "close"]]
        high, low = frame.high.copy(), frame.low.copy()
        frame["high"], frame["low"] = 200 - low, 200 - high
    frame[["open", "high", "low", "close"]] *= scale
    closed = frame.iloc[-1].time + pd.Timedelta(minutes=5)
    analyses = {tf: {"timestamp": (closed.floor(f"{minutes}min") - pd.Timedelta(minutes=minutes)).isoformat(),
                     "market_structure": {"trend_state_direction": "BEARISH" if direction == "BUY" else "BULLISH"}}
                for tf, minutes in (("M15", 15), ("H1", 60), ("H4", 240))}
    analyses["M5"] = {"timestamp": frame.iloc[-1].time.isoformat(), "market_structure": {},
                      "indicators": {"atr_14": scale, "ema_9": 100 * scale,
                                     "ema_21": (99.8 if direction == "BUY" else 100.2) * scale,
                                     "rsi_14": 56 if direction == "BUY" else 44,
                                     "adx_14": 30, "adx_delta": -2.1}}
    return analyses, frame


class ReversalWatchTests(unittest.TestCase):
    def test_shared_discovery_and_entry_analysis_attaches_only_research(self):
        analyses, bars = fixture()
        engine = TradingEngine.__new__(TradingEngine)
        engine.analyzer = SimpleNamespace(analyze=Mock(side_effect=lambda symbol, timeframe, frame: copy.deepcopy(analyses[timeframe])))
        prepared = asyncio.run(engine._confirmed_entry_analyses("TEST", bars, [bars, bars, bars]))
        self.assertTrue(prepared["M5"]["market_structure"]["reversal_watch"]["candidate"])
        self.assertEqual(permitted_entry_actions(build_evidence_ids(prepared)), ())

    def test_symmetric_scale_and_calendar_independent(self):
        for side in ("BUY", "SELL"):
            for scale in (.00001, .01, 1, 1000):
                for year in (2020, 2026, 2032, 2040):
                    with self.subTest(side=side, scale=scale, year=year):
                        analyses, bars = fixture(side, scale, year)
                        watch = classify_reversal_watch(analyses, bars)
                        self.assertTrue(watch["candidate"], watch)
                        self.assertFalse(watch["live_eligible"])
                        self.assertEqual(watch["direction"], side)
                        self.assertAlmostEqual(abs(watch["take_profit"] - watch["reference_price"]) /
                                               abs(watch["reference_price"] - watch["stop_loss"]), 2)

    def test_watch_cannot_create_live_evidence_or_model_admission(self):
        analyses, bars = fixture()
        before = build_evidence_ids(analyses)
        analyses["M5"]["market_structure"]["reversal_watch"] = classify_reversal_watch(analyses, bars)
        self.assertEqual(before, build_evidence_ids(analyses))
        self.assertEqual(permitted_entry_actions(build_evidence_ids(analyses)), ())
        report = screen_opportunities(analyses, {"capital_fit": True, "broker_open": True})
        self.assertTrue(report["research_watch"]["candidate"])
        self.assertFalse(report["model_eligible"])
        self.assertEqual(report["viable_entry_actions"], [])
        self.assertEqual(set(report["missing_entry_reasons"]), {"BUY", "SELL"})

    def test_invalid_and_stale_inputs_do_not_become_candidates(self):
        for problem in ("gap", "duplicate", "nan", "bounds", "mismatch", "stale", "atr", "future_h4", "old_h1", "extended", "recycled"):
            with self.subTest(problem=problem):
                analyses, bars = fixture()
                if problem == "gap": bars.loc[2, "time"] -= pd.Timedelta(minutes=5)
                if problem == "duplicate": bars.loc[4, "time"] = bars.loc[3, "time"]
                if problem == "nan": bars.loc[5, "close"] = float("nan")
                if problem == "bounds": bars.loc[5, "high"] = 99
                if problem == "mismatch": analyses["M5"]["timestamp"] = "2032-03-01T12:30:00Z"
                if problem == "stale": bars.attrs["is_stale"] = True
                if problem == "atr": analyses["M5"]["indicators"]["atr_14"] = 0
                if problem == "future_h4": analyses["H4"]["timestamp"] = "2032-03-01T12:00:00Z"
                if problem == "old_h1": analyses["H1"]["timestamp"] = "2032-03-01T09:00:00Z"
                if problem == "extended": bars.loc[5, ["close", "high"]] = [103, 103.1]
                if problem == "recycled": bars.loc[4, ["close", "high"]] = [100.2, 100.25]
                self.assertFalse(classify_reversal_watch(analyses, bars)["candidate"])

    def test_watch_is_disabled_explicitly_and_never_mutates_inputs(self):
        analyses, bars = fixture()
        original = copy.deepcopy(analyses)
        with patch("core.opportunities.settings", replace(settings, reversal_watch_enabled=False)):
            self.assertFalse(classify_reversal_watch(analyses, bars)["candidate"])
        self.assertEqual(analyses, original)

    def test_macro_alignment_is_not_mislabeled_countertrend(self):
        analyses, bars = fixture()
        for tf in ("M15", "H1", "H4"):
            analyses[tf]["market_structure"]["trend_state_direction"] = "BULLISH"
        self.assertFalse(classify_reversal_watch(analyses, bars)["candidate"])


if __name__ == "__main__":
    unittest.main()

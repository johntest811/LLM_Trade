from dataclasses import replace
import unittest
from unittest.mock import patch

import pandas as pd

from app_config.settings import settings
from core.engine import TradingEngine
from core.entry_retest import annotate_retest_continuation
from core.evidence import build_evidence_ids, permitted_entry_actions
from core.validator import DecisionValidator
from risk.manager import RiskManager


POLICY = replace(
    settings, retest_continuation_enabled=True, price_pullback_enabled=True,
    entry_require_adx_rising=True, entry_adx_decline_tolerance=0.5,
    entry_aligned_adx_decline_tolerance=1.5,
    entry_min_adx=15.0, breakout_min_adx=25.0, confirmation_min_adx=19.1,
    retest_min_resumption_atr=0.1, entry_max_candle_range_atr=1.5,
    price_pullback_lookback_bars=6, price_pullback_min_depth_atr=0.2,
    price_pullback_max_resumption_bars=3,
    price_pullback_max_depth_atr=1.5, price_pullback_break_buffer_atr=0.05,
    price_pullback_max_resumption_atr=0.85,
)


def fixture(direction="BULLISH", scale=1.0, start="2026-09-08T02:55:00Z"):
    rows = [
        (100.0, 100.3, 99.9, 100.25), (100.25, 100.5, 100.15, 100.45),
        (100.45, 100.7, 100.4, 100.6), (100.6, 100.8, 100.5, 100.7),
        (100.7, 100.75, 100.4, 100.5), (100.5, 100.55, 100.15, 100.2),
        (100.2, 100.72, 100.18, 100.65),
    ]
    sign = 1 if direction == "BULLISH" else -1
    if sign < 0:
        rows = [(200-o, 200-l, 200-h, 200-c) for o,h,l,c in rows]
    bars = pd.DataFrame([[v*scale for v in row] for row in rows], columns=["open","high","low","close"])
    bars["time"] = pd.date_range(start, periods=len(rows), freq="5min")
    bars.attrs["pip_size"] = 0.1*scale
    frames = {}
    for tf in ("M5", "M15", "H1", "H4"):
        frames[tf] = {
            "symbol": "TEST", "timeframe": tf, "timestamp": str(bars.iloc[-1]["time"]),
            "indicators": {
                "current_price": bars.iloc[-1]["close"], "atr_14": scale,
                "atr_14_pips": 10.0, "pip_size": 0.1*scale,
                "adx_14": 36.76 if tf == "M5" else 40.0,
                "adx_delta": -1.14 if tf == "M5" else 0.5,
                "ema_9": (100+sign*0.4)*scale, "ema_21": (100+sign*0.3)*scale,
                "rsi_14": 50+sign*5, "macd": {"diff": sign*0.1*scale},
                "stochastic": {"k":50+sign*10,"d":50+sign*8},
                "bollinger_bands": {"upper":103*scale,"lower":97*scale,"middle":100*scale},
                "candle_range_atr":0.54,"candle_body_atr_signed":sign*0.45,
                "candle_return_atr":sign*0.45,"opening_gap_atr":0.0,
            },
            "market_structure": {
                "trend": direction,"trend_state_direction":direction,
                "trend_state":f"CONFIRMED_{direction}","structure_events":[],
                "breakout_status":"None","support":97*scale,"resistance":103*scale,
                "supply_zones":[],"demand_zones":[],"order_blocks":[],"fair_value_gaps":[],
                "liquidity_zones":{},"candlestick_patterns":[],
            },
        }
    return frames, bars


def annotate(frames, bars, previous="CONFIRMED_BULLISH", policy=POLICY):
    with patch("core.entry_retest.settings", policy):
        return annotate_retest_continuation(
            frames["M5"], frames["M15"], frames["H1"], {"M5":previous},
            completed_bars=bars, h4_analysis=frames["H4"],
        )


class PricePullbackTests(unittest.TestCase):
    def test_signal_is_symmetric_and_scale_and_calendar_independent(self):
        for year in (2020, 2026, 2032, 2040):
            for month in (1, 3, 11):
                for scale in (0.00001, 0.01, 1, 1000):
                    for direction in ("BULLISH", "BEARISH"):
                        with self.subTest(year=year, month=month, scale=scale, direction=direction):
                            frames, bars = fixture(direction, scale, f"{year}-{month:02}-01T23:40:00Z")
                            event = annotate(frames, bars, f"CONFIRMED_{direction}")
                            self.assertIsNotNone(event)
                            self.assertEqual(event["kind"], "PRICE_PULLBACK_RESUMPTION")
                            self.assertEqual(event["direction"], direction)
                            self.assertAlmostEqual(event["pullback_depth_atr"], 0.6)
                            self.assertEqual(permitted_entry_actions(build_evidence_ids(frames)),
                                             ("BUY" if direction == "BULLISH" else "SELL",))

    def test_discovery_prefilter_model_contract_and_final_risk_agree(self):
        for direction, action in (("BULLISH","BUY"),("BEARISH","SELL")):
            frames, bars = fixture(direction)
            self.assertIsNotNone(annotate(frames, bars, f"CONFIRMED_{direction}"))
            with patch("core.engine.settings", POLICY), patch("risk.manager.settings", POLICY):
                self.assertEqual(TradingEngine._entry_prefilter_reason(frames["M5"],frames["M15"],frames["H1"]), "")
                ok, reason = RiskManager._check_entry_structure(action, *frames.values(), {"trend_align":True,"tf_agreement":True})
                self.assertTrue(ok, reason)
            evidence = build_evidence_ids(frames)
            trigger = next(value for value in evidence if value.startswith("M5_RETEST"))
            ok, _, reason = DecisionValidator.validate_decision(
                {"action":action,"confidence":0.85,"evidence_ids":[trigger],"reasoning":"Confirmed price pullback"},
                allowed_evidence_ids=evidence,
            )
            self.assertTrue(ok, reason)

    def test_exact_label_retest_can_use_existing_bounded_adx_exception(self):
        frames, bars = fixture()
        # This used to be filtered before the retest could reach the exception.
        event = annotate(frames, bars, "PULLBACK_IN_BULLISH_TREND",
                         replace(POLICY, price_pullback_enabled=False))
        self.assertIsNotNone(event)
        self.assertEqual(event["kind"], "STATE_PULLBACK_RESUMPTION")

    def test_signal_does_not_persist_into_the_next_candle(self):
        frames, bars = fixture()
        self.assertIsNotNone(annotate(frames, bars))
        next_row = {"open":100.65,"high":100.9,"low":100.6,"close":100.85,
                    "time":bars.iloc[-1]["time"]+pd.Timedelta(minutes=5)}
        bars = pd.concat([bars, pd.DataFrame([next_row])], ignore_index=True)
        frames["M5"]["timestamp"] = str(bars.iloc[-1]["time"])
        self.assertIsNone(annotate(frames, bars))
        self.assertNotIn("retest_continuation", frames["M5"]["market_structure"])

    def test_two_bar_resumption_emits_only_on_first_close_through_anchor(self):
        frames, bars = fixture()
        bars.loc[bars.index[-1], ["open","high","low","close"]] = [100.2,100.58,100.18,100.5]
        self.assertIsNone(annotate(frames, bars))
        next_row = {"open":100.5,"high":100.75,"low":100.45,"close":100.7,
                    "time":bars.iloc[-1]["time"]+pd.Timedelta(minutes=5)}
        bars = pd.concat([bars,pd.DataFrame([next_row])], ignore_index=True)
        frames["M5"]["timestamp"] = str(bars.iloc[-1]["time"])
        event = annotate(frames, bars)
        self.assertIsNotNone(event)
        self.assertEqual(event["bars_since_pullback"], 2)
        self.assertAlmostEqual(event["break_level"], 100.55)

    def test_no_retest_from_trend_labels_or_unbroken_pullback(self):
        frames, bars = fixture()
        bars.loc[bars.index[-1], "close"] = 100.53
        self.assertIsNone(annotate(frames, bars))
        frames, bars = fixture()
        bars.loc[bars.index[-2], ["open","high","low","close"]] = [100.5,100.65,100.45,100.6]
        self.assertIsNone(annotate(frames, bars))

    def test_weak_collapsing_missing_momentum_and_macro_conflicts_stay_blocked(self):
        for field, value in (("adx_delta",-1.6),("adx_delta",float("nan")),
                             ("adx_delta",None),("adx_14",20),("atr_14",0),
                             ("ema_9",100.1),("rsi_14",45)):
            frames, bars = fixture()
            frames["M5"]["indicators"][field] = value
            with self.subTest(field=field, value=value):
                self.assertIsNone(annotate(frames, bars))
        for tf in ("M15","H1","H4"):
            frames, bars = fixture()
            frames[tf]["market_structure"]["trend_state_direction"] = "BEARISH"
            self.assertIsNone(annotate(frames, bars))

    def test_stale_future_duplicate_gapped_and_corrupt_bars_cannot_create_evidence(self):
        for kind in ("stale","future","duplicate","weekend_gap","bad_ohlc","nan","unordered"):
            frames, bars = fixture()
            if kind == "stale": bars.attrs["is_stale"] = True
            if kind == "future": bars.loc[bars.index[-1],"time"] += pd.Timedelta(minutes=5)
            if kind == "duplicate": bars.loc[bars.index[-2],"time"] = bars.iloc[-1]["time"]
            if kind == "weekend_gap": bars.loc[bars.index[-2],"time"] -= pd.Timedelta(days=2)
            if kind == "bad_ohlc": bars.loc[bars.index[-1],"high"] = 99
            if kind == "nan": bars.loc[bars.index[-1],"low"] = float("nan")
            if kind == "unordered": bars = bars.iloc[::-1]
            with self.subTest(kind=kind): self.assertIsNone(annotate(frames, bars))
        frames, bars = fixture()
        bars.loc[bars.index[-2],"time"] -= pd.Timedelta(days=2)
        self.assertIsNone(annotate(frames, bars, "PULLBACK_IN_BULLISH_TREND"))

    def test_large_chase_and_gap_are_not_new_continuation_opportunities(self):
        for kind in ("chase","gap","deep_pullback","opposing_structure"):
            frames, bars = fixture()
            if kind == "chase": bars.loc[bars.index[-1],["high","close"]] = [101.3,101.2]
            if kind == "gap": bars.loc[bars.index[-1],"open"] = 100.52
            if kind == "deep_pullback": bars.loc[bars.index[1],"high"] = 102.0
            if kind == "opposing_structure":
                frames["M5"]["market_structure"]["structure_events"] = [{"type":"CHOCH","direction":"BEARISH"}]
            with self.subTest(kind=kind): self.assertIsNone(annotate(frames, bars))

    def test_new_trigger_does_not_bypass_outer_band_overextension(self):
        frames, bars = fixture()
        self.assertIsNotNone(annotate(frames, bars))
        frames["M5"]["indicators"].update(rsi_14=80, stochastic={"k":99,"d":99})
        frames["M5"]["indicators"]["bollinger_bands"]["upper"] = 100.1
        with patch("risk.manager.settings", POLICY):
            ok, reason = RiskManager._check_entry_structure("BUY", *frames.values(), {"trend_align":True,"tf_agreement":True})
        self.assertFalse(ok)
        self.assertIn("Overextension", reason)

    def test_feature_switch_disables_only_new_price_pattern(self):
        frames, bars = fixture()
        self.assertIsNone(annotate(frames, bars, policy=replace(POLICY,price_pullback_enabled=False)))

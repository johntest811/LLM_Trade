import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from app_config.settings import settings
from risk.manager import RiskManager


def _analysis(
    timeframe,
    *,
    trend="BULLISH",
    direction="BULLISH",
    adx=25.0,
    breakout="None",
    events=None,
    timestamp="2026-07-23 07:20:00",
    current=100.0,
    atr=1.0,
    adx_delta=1.0,
    stoch=60.0,
    rsi=55.0,
    candle_range_atr=1.0,
    candle_return_atr=0.0,
    candle_body_atr_signed=0.0,
    upper=101.0,
    lower=99.0,
):
    return {
        "timeframe": timeframe,
        "timestamp": timestamp,
        "indicators": {
            "adx_14": adx,
            "adx_delta": adx_delta,
            "current_price": current,
            "atr_14": atr,
            "rsi_14": rsi,
            "candle_range_atr": candle_range_atr,
            "candle_return_atr": candle_return_atr,
            "candle_body_atr_signed": candle_body_atr_signed,
            "stochastic": {"k": stoch, "d": stoch},
            "bollinger_bands": {"upper": upper, "lower": lower},
        },
        "market_structure": {
            "trend": trend,
            "trend_state_direction": direction,
            "breakout_status": breakout,
            "structure_events": events or [],
            "support": current - 2.0 * atr,
            "resistance": current + 2.0 * atr,
            "demand_zones": [],
            "supply_zones": [],
        },
    }


class EntryStructureGateTests(unittest.TestCase):
    def test_confirmed_aligned_trend_without_fresh_trigger_is_rejected(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, {"trend_align": True, "tf_agreement": True}
        )

        self.assertFalse(ok)
        self.assertIn("Fresh Structure", reason)

    def test_confirmed_aligned_trend_with_fresh_bos_is_allowed(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-23 07:20:00",
        }]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, {"trend_align": True, "tf_agreement": True}
        )

        self.assertTrue(ok, reason)

    def test_verified_pullback_retest_is_a_fresh_structure_trigger(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["retest_continuation"] = {
            "direction": "BULLISH",
            "time": "2026-07-23 07:20:00",
            "previous_state": "PULLBACK_IN_BULLISH_TREND",
            "current_state": "CONFIRMED_BULLISH",
        }

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="PULLBACK_RESUMPTION"
        )

        self.assertTrue(ok, reason)

    def test_falling_adx_rejects_fresh_trigger(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["indicators"]["adx_delta"] = -0.75
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-23 07:20:00",
        }]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, {"trend_align": True, "tf_agreement": True}
        )

        self.assertFalse(ok)
        self.assertIn("ADX Direction", reason)

    def test_small_adx_decline_is_treated_as_noise(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["indicators"]["adx_delta"] = -0.25
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-23 07:20:00",
        }]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, {"trend_align": True, "tf_agreement": True}
        )

        self.assertTrue(ok, reason)

    def test_sell_too_close_to_demand_is_rejected(self):
        analyses = [
            _analysis(tf, trend="BEARISH", direction="BEARISH")
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BEARISH",
            "time": "2026-07-23 07:20:00",
        }]
        analyses[0]["market_structure"]["demand_zones"] = [{
            "bottom": 99.2,
            "top": 99.5,
        }]

        ok, reason = RiskManager._check_entry_structure(
            "SELL", *analyses, {"trend_align": True, "tf_agreement": True}
        )

        self.assertFalse(ok)
        self.assertIn("Opposing Zone", reason)

    def test_sell_ignores_support_and_demand_already_broken_behind_price(self):
        analyses = [
            _analysis(tf, trend="BEARISH", direction="BEARISH", current=99.0)
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"].update(
            support=99.2,
            demand_zones=[{"bottom": 99.2, "top": 99.5}],
            structure_events=[{
                "type": "BOS",
                "direction": "BEARISH",
                "time": "2026-07-23 07:20:00",
            }],
        )

        ok, reason = RiskManager._check_entry_structure(
            "SELL", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_buy_ignores_resistance_and_supply_already_broken_behind_price(self):
        analyses = [
            _analysis(tf, current=101.0)
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"].update(
            resistance=100.8,
            supply_zones=[{"bottom": 100.5, "top": 100.8}],
            structure_events=[{
                "type": "BOS",
                "direction": "BULLISH",
                "time": "2026-07-23 07:20:00",
            }],
        )

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_buy_retest_does_not_resurrect_level_broken_by_completed_close(self):
        analyses = [
            _analysis(tf, current=101.0)
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"].update(
            resistance=100.95,
            supply_zones=[{"bottom": 100.75, "top": 100.95}],
            structure_events=[{
                "type": "BOS",
                "direction": "BULLISH",
                "time": "2026-08-10 05:35:00",
            }],
        )
        snapshot = SimpleNamespace(
            metrics=SimpleNamespace(bid=100.90, ask=100.91)
        )

        ok, reason = RiskManager._check_entry_structure(
            "BUY",
            *analyses,
            strategy_mode="TREND_CONTINUATION",
            market_snapshot=snapshot,
        )

        self.assertTrue(ok, reason)

    def test_sell_retest_does_not_resurrect_level_broken_by_completed_close(self):
        analyses = [
            _analysis(
                tf,
                trend="BEARISH",
                direction="BEARISH",
                current=99.0,
                upper=101.0,
                lower=98.0,
            )
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"].update(
            support=99.05,
            demand_zones=[{"bottom": 99.05, "top": 99.25}],
            structure_events=[{
                "type": "BOS",
                "direction": "BEARISH",
                "time": "2026-08-10 05:35:00",
            }],
        )
        snapshot = SimpleNamespace(
            metrics=SimpleNamespace(bid=99.10, ask=99.11)
        )

        ok, reason = RiskManager._check_entry_structure(
            "SELL",
            *analyses,
            strategy_mode="TREND_CONTINUATION",
            market_snapshot=snapshot,
        )

        self.assertTrue(ok, reason)

    def test_same_thesis_loss_requires_new_post_loss_structure(self):
        m5 = _analysis(
            "M5",
            trend="BEARISH",
            direction="BEARISH",
            timestamp="2026-07-23 07:20:00",
            events=[{
                "type": "BOS",
                "direction": "BEARISH",
                "time": "2026-07-23 07:20:00",
            }],
        )
        history = [{
            "direction": "SELL",
            "net_profit": -0.5,
            "close_time": "2026-07-23T07:27:45+00:00",
        }]

        ok, reason = RiskManager._check_same_thesis_reentry(
            "SELL", m5, history
        )

        self.assertFalse(ok)
        self.assertIn("Same-Thesis Re-entry", reason)

    def test_same_thesis_reentry_accepts_later_fresh_structure(self):
        m5 = _analysis(
            "M5",
            trend="BEARISH",
            direction="BEARISH",
            timestamp="2026-07-23 07:40:00",
            events=[{
                "type": "BOS",
                "direction": "BEARISH",
                "time": "2026-07-23 07:40:00",
            }],
        )
        history = [{
            "direction": "SELL",
            "net_profit": -0.5,
            "close_time": "2026-07-23T07:27:45+00:00",
        }]

        ok, reason = RiskManager._check_same_thesis_reentry(
            "SELL", m5, history
        )

        self.assertTrue(ok, reason)

    def test_same_thesis_reentry_accepts_later_verified_retest(self):
        m5 = _analysis(
            "M5",
            trend="BEARISH",
            direction="BEARISH",
            timestamp="2026-07-23 07:40:00",
        )
        m5["market_structure"]["retest_continuation"] = {
            "direction": "BEARISH",
            "time": "2026-07-23 07:40:00",
        }
        history = [{
            "direction": "SELL",
            "net_profit": 0.01,
            "close_time": "2026-07-23T07:27:45+00:00",
        }]

        ok, reason = RiskManager._check_same_thesis_reentry(
            "SELL", m5, history
        )

        self.assertTrue(ok, reason)

    def test_profitable_same_thesis_close_still_requires_fresh_structure(self):
        m5 = _analysis(
            "M5",
            trend="BULLISH",
            direction="BULLISH",
            timestamp="2026-07-23 07:30:00",
            events=[{
                "type": "BOS",
                "direction": "BULLISH",
                "time": "2026-07-23 07:25:00",
            }],
        )
        history = [{
            "direction": "BUY",
            "net_profit": 0.20,
            "close_time": "2026-07-23T07:29:30+00:00",
        }]

        ok, reason = RiskManager._check_same_thesis_reentry(
            "BUY", m5, history
        )

        self.assertFalse(ok)
        self.assertIn("closed with a profit", reason)

    def test_opposite_direction_close_does_not_block_new_thesis(self):
        m5 = _analysis(
            "M5",
            trend="BULLISH",
            direction="BULLISH",
            timestamp="2026-07-23 07:30:00",
        )
        history = [{
            "direction": "SELL",
            "net_profit": 0.20,
            "close_time": "2026-07-23T07:29:30+00:00",
        }]

        ok, reason = RiskManager._check_same_thesis_reentry(
            "BUY", m5, history
        )

        self.assertTrue(ok, reason)

    def test_cadjpy_weak_breakout_pattern_is_rejected(self):
        m5 = _analysis(
            "M5",
            trend="NEUTRAL",
            adx=20.7,
            breakout="BULLISH BREAKOUT",
            current=115.462,
            stoch=93.2,
            upper=115.457,
        )
        m15 = _analysis("M15", adx=8.7)
        h1 = _analysis("H1", adx=11.4)
        h4 = _analysis("H4", adx=29.0)

        ok, reason = RiskManager._check_entry_structure(
            "BUY", m5, m15, h1, h4, {"trend_align": False, "tf_agreement": False}
        )

        self.assertFalse(ok)
        self.assertIn("Breakout Confirmation", reason)

    def test_valid_bos_is_not_invalidated_by_breakout_annotation(self):
        m5 = _analysis(
            "M5",
            adx=26.0,
            breakout="BEARISH BREAKOUT",
            trend="BEARISH",
            direction="BEARISH",
            events=[{
                "type": "BOS",
                "direction": "BEARISH",
                "time": "2026-07-23 07:20:00",
            }],
        )
        m15 = _analysis(
            "M15", adx=19.0, trend="BEARISH", direction="BEARISH"
        )
        h1 = _analysis("H1", adx=25.0, trend="BEARISH", direction="BEARISH")
        h4 = _analysis("H4", adx=25.0, trend="BEARISH", direction="BEARISH")

        ok, reason = RiskManager._check_entry_structure(
            "SELL", m5, m15, h1, h4, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_strong_but_overextended_breakout_waits_for_retest(self):
        m5 = _analysis(
            "M5",
            trend="NEUTRAL",
            adx=30.0,
            breakout="BULLISH BREAKOUT",
            current=101.2,
            stoch=93.0,
            rsi=74.0,
            upper=101.0,
        )
        confirmations = [_analysis(tf, adx=22.0) for tf in ("M15", "H1", "H4")]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", m5, *confirmations, {"trend_align": False, "tf_agreement": False}
        )

        self.assertFalse(ok)
        self.assertIn("Overextension", reason)

    def test_recent_eurusd_breakout_loss_is_rejected_before_entry(self):
        """Regression for ticket 312865538's completed-candle snapshot."""
        m5 = _analysis(
            "M5",
            trend="BULLISH",
            direction="BULLISH",
            adx=26.0069,
            breakout="BULLISH BREAKOUT",
            current=1.15280,
            atr=0.000402426,
            stoch=95.8974,
            rsi=66.9079,
            upper=1.1529445,
            candle_range_atr=1.2922,
            candle_return_atr=0.6212,
            candle_body_atr_signed=0.6212,
        )
        m15 = _analysis("M15", adx=27.2)
        h1 = _analysis("H1", adx=12.7)
        h4 = _analysis("H4", adx=18.5)

        ok, reason = RiskManager._check_entry_structure(
            "BUY",
            m5,
            m15,
            h1,
            h4,
            strategy_mode="TREND_CONTINUATION",
        )

        self.assertFalse(ok)
        self.assertTrue(
            "Breakout Macro Strength" in reason
            or "Breakout Exhaustion" in reason,
            reason,
        )

    def test_exceptionally_strong_lower_breakout_can_bridge_marginal_macro(self):
        m5 = _analysis(
            "M5",
            adx=31.0,
            breakout="BULLISH BREAKOUT",
            stoch=70.0,
            rsi=60.0,
        )
        m15 = _analysis("M15", adx=32.0)
        h1 = _analysis("H1", adx=16.0)
        h4 = _analysis("H4", adx=18.5)

        ok, reason = RiskManager._check_entry_structure(
            "BUY",
            m5,
            m15,
            h1,
            h4,
            strategy_mode="TREND_CONTINUATION",
        )

        self.assertTrue(ok, reason)

    def test_breakout_uses_combined_m5_m15_momentum(self):
        m5 = _analysis(
            "M5",
            adx=22.9,
            breakout="BULLISH BREAKOUT",
            current=100.0,
            upper=101.0,
        )
        m15 = _analysis("M15", adx=35.4)
        h1 = _analysis("H1", adx=29.4)
        h4 = _analysis("H4", trend="NEUTRAL", direction="NEUTRAL", adx=16.6)

        ok, reason = RiskManager._check_entry_structure(
            "BUY", m5, m15, h1, h4, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_band_touch_with_one_extreme_oscillator_is_not_overextension(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0] = _analysis(
            "M5",
            current=101.2,
            upper=101.0,
            stoch=93.0,
            rsi=60.0,
            events=[{
                "type": "BOS",
                "direction": "BULLISH",
                "time": "2026-07-27 13:55:00",
            }],
        )

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_usdjpy_style_outer_band_entry_is_rejected_before_chasing(self):
        m5 = _analysis(
            "M5",
            adx=29.7,
            breakout="BULLISH BREAKOUT",
            current=163.75,
            atr=0.02811,
            stoch=89.1,
            rsi=73.4,
            upper=163.73488,
            lower=163.65252,
        )
        m5["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        confirmations = [
            _analysis("M15", adx=22.0),
            _analysis("H1", adx=28.8),
            _analysis("H4", adx=23.2),
        ]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", m5, *confirmations, strategy_mode="TREND_CONTINUATION"
        )

        self.assertFalse(ok)
        self.assertIn("Overextension", reason)

    def test_eurusd_style_sell_near_support_is_rejected(self):
        m5 = _analysis(
            "M5",
            trend="BEARISH",
            direction="BEARISH",
            adx=38.6,
            current=1.13800,
            atr=0.000338,
            stoch=25.5,
            rsi=34.5,
            upper=1.13948,
            lower=1.13771,
        )
        m5["market_structure"].update(
            support=1.13764,
            resistance=1.13954,
            structure_events=[{
                "type": "BOS",
                "direction": "BEARISH",
                "time": "2026-07-27 13:00:00",
            }],
        )
        confirmations = [
            _analysis("M15", trend="BEARISH", direction="BEARISH", adx=29.0),
            _analysis("H1", trend="BEARISH", direction="BEARISH", adx=25.0),
            _analysis("H4", trend="BEARISH", direction="BEARISH", adx=22.0),
        ]

        ok, reason = RiskManager._check_entry_structure(
            "SELL", m5, *confirmations, strategy_mode="TREND_CONTINUATION"
        )

        self.assertFalse(ok)
        self.assertIn("Opposing Zone", reason)

    def test_extended_m5_impulse_requires_retest(self):
        analyses = [
            _analysis(
                tf,
                candle_range_atr=1.61 if tf == "M5" else 1.0,
                candle_return_atr=1.61 if tf == "M5" else 0.0,
                candle_body_atr_signed=1.55 if tf == "M5" else 0.0,
            )
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertFalse(ok)
        self.assertIn("Entry Chase", reason)

    def test_long_wick_is_not_misclassified_as_entry_chase(self):
        analyses = [
            _analysis(
                tf,
                trend="BEARISH",
                direction="BEARISH",
                candle_range_atr=1.86 if tf == "M5" else 1.0,
                candle_return_atr=-0.36 if tf == "M5" else 0.0,
                candle_body_atr_signed=-0.24 if tf == "M5" else 0.0,
            )
            for tf in ("M5", "M15", "H1", "H4")
        ]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BEARISH",
            "time": "2026-07-27 13:55:00",
        }]

        ok, reason = RiskManager._check_entry_structure(
            "SELL", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_weak_h4_momentum_rejects_continuation(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        analyses[1]["indicators"]["adx_14"] = 19.0
        analyses[3]["indicators"]["adx_14"] = 14.1

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertFalse(ok)
        self.assertIn("Macro Momentum", reason)

    def test_fresh_bos_with_strong_lower_momentum_can_confirm_weak_h4(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        analyses[0]["indicators"]["adx_14"] = 26.0
        analyses[1]["indicators"]["adx_14"] = 22.0
        analyses[3]["indicators"]["adx_14"] = 14.1

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_neutral_m15_label_can_bridge_fresh_bos_when_macro_agrees(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        analyses[0]["indicators"]["adx_14"] = 22.0
        analyses[1]["market_structure"]["trend_state_direction"] = "NEUTRAL"
        analyses[1]["indicators"]["adx_14"] = 31.0

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertTrue(ok, reason)

    def test_opposing_m15_direction_cannot_use_neutral_bridge(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        analyses[0]["indicators"]["adx_14"] = 30.0
        analyses[1]["market_structure"]["trend_state_direction"] = "BEARISH"
        analyses[1]["indicators"]["adx_14"] = 31.0

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="TREND_CONTINUATION"
        )

        self.assertFalse(ok)
        self.assertIn("Structure Gate", reason)

    def test_verified_retest_can_confirm_weak_h4(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["retest_continuation"] = {
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
            "previous_state": "PULLBACK_IN_BULLISH_TREND",
            "current_state": "CONFIRMED_BULLISH",
        }
        analyses[0]["indicators"]["adx_14"] = 26.0
        analyses[1]["indicators"]["adx_14"] = 22.0
        analyses[3]["indicators"]["adx_14"] = 14.1

        ok, reason = RiskManager._check_entry_structure(
            "BUY", *analyses, strategy_mode="PULLBACK_RESUMPTION"
        )

        self.assertTrue(ok, reason)

    def test_live_buy_quote_beyond_band_rejects_completed_close_setup(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0] = _analysis(
            "M5",
            current=101.30,
            upper=101.0,
            stoch=93.0,
            rsi=74.0,
            events=[{
                "type": "BOS",
                "direction": "BULLISH",
                "time": "2026-07-27 13:55:00",
            }],
        )
        snapshot = SimpleNamespace(
            metrics=SimpleNamespace(bid=101.30, ask=101.31)
        )

        ok, reason = RiskManager._check_entry_structure(
            "BUY",
            *analyses,
            strategy_mode="TREND_CONTINUATION",
            market_snapshot=snapshot,
        )

        self.assertFalse(ok)
        self.assertIn("Overextension", reason)

    def test_live_quote_drift_rejects_stale_entry_price(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        snapshot = SimpleNamespace(
            metrics=SimpleNamespace(bid=100.29, ask=100.30)
        )

        ok, reason = RiskManager._check_entry_structure(
            "BUY",
            *analyses,
            strategy_mode="TREND_CONTINUATION",
            market_snapshot=snapshot,
        )

        self.assertFalse(ok)
        self.assertIn("Execution Drift", reason)

    def test_buy_spread_is_not_counted_as_execution_drift(self):
        analyses = [_analysis(tf) for tf in ("M5", "M15", "H1", "H4")]
        analyses[0]["market_structure"]["structure_events"] = [{
            "type": "BOS",
            "direction": "BULLISH",
            "time": "2026-07-27 13:55:00",
        }]
        snapshot = SimpleNamespace(
            metrics=SimpleNamespace(bid=100.0, ask=100.40)
        )

        ok, reason = RiskManager._check_entry_structure(
            "BUY",
            *analyses,
            strategy_mode="TREND_CONTINUATION",
            market_snapshot=snapshot,
        )

        self.assertTrue(ok, reason)

    def test_local_model_reversal_receives_deterministic_strategy_mode(self):
        m5 = _analysis(
            "M5",
            trend="BEARISH",
            direction="BULLISH",
            events=[
                {"type": "CHOCH", "direction": "BULLISH"},
                {"type": "BOS", "direction": "BULLISH"},
            ],
        )
        m5["market_structure"]["trend_state"] = "EARLY_BULLISH_REVERSAL"
        m15 = _analysis(
            "M15",
            trend="BEARISH",
            direction="BULLISH",
            events=[{"type": "CHOCH", "direction": "BULLISH"}],
        )
        h1 = _analysis("H1", trend="BEARISH", direction="BEARISH")
        h4 = _analysis("H4", trend="BEARISH", direction="BEARISH")

        mode = RiskManager._resolve_strategy_mode(
            "BUY", {}, m5, m15, h1, h4
        )

        self.assertEqual(mode, "CONFIRMED_REVERSAL")

    def test_reversal_inference_respects_adaptive_reversal_switch(self):
        m5 = _analysis(
            "M5",
            trend="BEARISH",
            direction="BULLISH",
            events=[
                {"type": "CHOCH", "direction": "BULLISH"},
                {"type": "BOS", "direction": "BULLISH"},
            ],
        )
        m5["market_structure"]["trend_state"] = "EARLY_BULLISH_REVERSAL"
        m15 = _analysis(
            "M15",
            trend="BEARISH",
            direction="BULLISH",
            events=[{"type": "CHOCH", "direction": "BULLISH"}],
        )
        h1 = _analysis("H1", trend="BEARISH", direction="BEARISH")
        h4 = _analysis("H4", trend="BEARISH", direction="BEARISH")

        with patch(
            "risk.manager.settings",
            replace(settings, adaptive_reversal_enabled=False),
        ):
            mode = RiskManager._resolve_strategy_mode(
                "BUY", {}, m5, m15, h1, h4
            )

        self.assertEqual(mode, "")

    def test_confirmed_reversal_bypasses_continuation_countertrend_gate(self):
        h1 = _analysis("H1", direction="BULLISH")
        h4 = _analysis("H4", direction="BEARISH")

        self.assertFalse(
            RiskManager._requires_strong_countertrend_exception(
                "CONFIRMED_REVERSAL", h1, h4
            )
        )
        self.assertTrue(
            RiskManager._requires_strong_countertrend_exception(
                "TREND_CONTINUATION", h1, h4
            )
        )

    def test_canonical_trend_direction_precedes_legacy_regime(self):
        analysis = _analysis(
            "H1", trend="NEUTRAL", direction="BEARISH"
        )

        self.assertEqual(
            RiskManager._trend_direction(analysis),
            "BEARISH",
        )

    def test_abnormal_completed_candle_triggers_market_shock_gate(self):
        m5 = _analysis("M5")
        m5["indicators"].update({"candle_range_atr": 3.0, "opening_gap_atr": 0.1})

        ok, reason = RiskManager._check_market_shock(m5)

        self.assertFalse(ok)
        self.assertIn("Market Shock", reason)

    def test_normal_completed_candle_passes_market_shock_gate(self):
        m5 = _analysis("M5")
        m5["indicators"].update({"candle_range_atr": 1.0, "opening_gap_atr": 0.1})

        ok, reason = RiskManager._check_market_shock(m5)

        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()

import unittest

from core.trend_state import classify_trend_state
from prompt_builder.generator import PromptGenerator


def _indicators(*, bullish=True, adx=30.0):
    return {
        "current_price": 1.1000,
        "ema_9": 1.1010 if bullish else 1.0990,
        "ema_21": 1.1000,
        "rsi_14": 58.0 if bullish else 42.0,
        "adx_14": adx,
        "atr_14_pips": 8.0,
        "macd": {"diff": 0.0002 if bullish else -0.0002},
        "stochastic": {"k": 55.0, "d": 50.0},
    }


def _analysis(regime, state, direction, event=None):
    return {
        "indicators": _indicators(bullish=direction == "BULLISH"),
        "market_structure": {
            "trend": regime,
            "regime_trend": regime,
            "trend_state": state,
            "trend_state_direction": direction,
            "trend_state_evidence": ["TEST"],
            "support": 1.09,
            "resistance": 1.11,
            "breakout_status": "None",
            "structure_events": [event] if event else [],
            "candlestick_patterns": [],
            "fair_value_gaps": [],
            "order_blocks": [],
        },
    }


class TrendStateTests(unittest.TestCase):
    def test_choch_marks_early_reversal_against_slow_regime(self):
        result = classify_trend_state(
            slow_trend="BEARISH",
            indicators=_indicators(bullish=True),
            structure_events=[{"type": "CHOCH", "direction": "BULLISH"}],
        )

        self.assertEqual(result["state"], "EARLY_BULLISH_REVERSAL")
        self.assertEqual(result["direction"], "BULLISH")
        self.assertIn("BULLISH_CHOCH", result["evidence"])

    def test_countermove_without_breakout_remains_pullback(self):
        result = classify_trend_state(
            slow_trend="BEARISH",
            indicators=_indicators(bullish=True),
            structure_events=[],
            breakout_status="None",
        )

        self.assertEqual(result["state"], "PULLBACK_IN_BEARISH_TREND")
        self.assertEqual(result["direction"], "BEARISH")

    def test_momentum_breakout_marks_early_reversal_without_waiting_for_slow_emas(self):
        result = classify_trend_state(
            slow_trend="BEARISH",
            indicators=_indicators(bullish=True, adx=25.0),
            structure_events=[],
            breakout_status="BULLISH BREAKOUT (test)",
        )

        self.assertEqual(result["state"], "EARLY_BULLISH_REVERSAL")
        self.assertIn("BULLISH_BREAKOUT", result["evidence"])

    def test_aligned_fast_and_slow_trends_are_confirmed(self):
        bullish = classify_trend_state(
            slow_trend="BULLISH", indicators=_indicators(bullish=True)
        )
        bearish = classify_trend_state(
            slow_trend="BEARISH", indicators=_indicators(bullish=False)
        )

        self.assertEqual(bullish["state"], "CONFIRMED_BULLISH")
        self.assertEqual(bearish["state"], "CONFIRMED_BEARISH")

    def test_prompt_exposes_all_timeframe_states_and_structure_events(self):
        choch = {"type": "CHOCH", "direction": "BULLISH", "level": 1.105}
        _, prompt = PromptGenerator.generate(
            symbol="USDCAD",
            timeframe="M5",
            analysis_data=_analysis(
                "BEARISH", "EARLY_BULLISH_REVERSAL", "BULLISH", choch
            ),
            m15_analysis=_analysis(
                "BEARISH", "PULLBACK_IN_BEARISH_TREND", "BEARISH"
            ),
            h1_analysis=_analysis("BEARISH", "CONFIRMED_BEARISH", "BEARISH"),
            h4_analysis=_analysis("BULLISH", "CONFIRMED_BULLISH", "BULLISH"),
            account_info={"balance": 15.17, "currency": "USD", "margin_free": 15.17},
            open_positions=[],
            trade_history=[],
            calendar_events=[],
        )

        self.assertIn("M5=EARLY_BULLISH_REVERSAL", prompt)
        self.assertIn("M15=PULLBACK_IN_BEARISH_TREND", prompt)
        self.assertIn("CHOCH-BULLISH@1.105", prompt)
        self.assertIn("a PULLBACK state is not a reversal", prompt)


if __name__ == "__main__":
    unittest.main()

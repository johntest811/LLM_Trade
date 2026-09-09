import unittest
from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from prompt_builder.generator import PromptGenerator


def _analysis(direction: str = "BEARISH") -> dict:
    bullish = direction == "BULLISH"
    return {
        "indicators": {
            "current_price": 1.405,
            "ema_9": 1.406 if bullish else 1.404,
            "ema_21": 1.405,
            "rsi_14": 58.0 if bullish else 42.0,
            "atr_14_pips": 6.0,
            "adx_14": 28.0,
            "macd": {"diff": 0.001 if bullish else -0.001},
            "stochastic": {"k": 55.0, "d": 50.0},
        },
        "market_structure": {
            "trend": direction,
            "regime_trend": direction,
            "trend_state": f"CONFIRMED_{direction}",
            "trend_state_direction": direction,
            "trend_state_evidence": ["BOS"],
            "structure_events": [
                {"type": "BOS", "direction": direction, "level": 1.404}
            ],
            "support": 1.403,
            "resistance": 1.407,
            "breakout_status": "CONFIRMED",
            "candlestick_patterns": [],
            "fair_value_gaps": [],
            "order_blocks": [],
        },
    }


class PromptIsolationTests(unittest.TestCase):
    def test_research_watch_cannot_bias_the_live_entry_prompt(self):
        analysis = _analysis()
        def generate():
            return PromptGenerator.generate(symbol="TEST", timeframe="M5", analysis_data=analysis,
                                            m15_analysis=analysis, h1_analysis=analysis, h4_analysis=analysis,
                                            account_info={}, open_positions=[], trade_history=[], calendar_events=None)
        before = generate()
        analysis["market_structure"]["reversal_watch"] = {
            "candidate": True, "live_eligible": False, "direction": "BUY",
            "reason": "RESEARCH_TEST_SENTINEL", "reference_price": 9,
        }
        self.assertEqual(generate(), before)

    def test_entry_direction_prompt_excludes_balance_and_prior_pnl(self):
        analysis = _analysis()
        _, prompt = PromptGenerator.generate(
            symbol="USDCAD",
            timeframe="M5",
            analysis_data=analysis,
            m15_analysis=analysis,
            h1_analysis=analysis,
            h4_analysis=analysis,
            account_info={
                "balance": 11.79,
                "margin_free": 11.79,
                "currency": "USD",
            },
            open_positions=[],
            trade_history=[
                {
                    "symbol": "USDCAD",
                    "action": "CLOSE",
                    "net_profit": -99.99,
                }
            ],
            calendar_events=None,
        )

        self.assertIn("USDCAD", prompt)
        self.assertIn("CHoCH/BOS events", prompt)
        self.assertIn("Allowed Evidence IDs", prompt)
        self.assertNotIn("Balance=", prompt)
        self.assertNotIn("FreeMargin=", prompt)
        self.assertNotIn("$-99.99", prompt)
        self.assertNotIn("Win Rate", prompt)
        self.assertNotIn("RiskCap=", prompt)
        self.assertIn("deliberately excluded", prompt)
        self.assertNotIn("StateEvidence=", prompt)
        self.assertIn("copy evidence_ids verbatim", prompt)

    def test_prompt_does_not_invite_disabled_counter_h4_continuations(self):
        analysis = _analysis("BULLISH")
        with patch(
            "prompt_builder.generator.settings",
            replace(settings, allow_strong_countertrend_entries=False),
        ):
            _, prompt = PromptGenerator.generate(
                symbol="USDCAD",
                timeframe="M5",
                analysis_data=analysis,
                m15_analysis=analysis,
                h1_analysis=analysis,
                h4_analysis=_analysis("BEARISH"),
                account_info={},
                open_positions=[],
                trade_history=[],
                calendar_events=[],
            )

        self.assertIn(
            "Do not choose an ordinary continuation entry against H4",
            prompt,
        )
        self.assertNotIn("A counter-H4 entry is exceptional", prompt)

    def test_prompt_explicitly_instructs_hold_when_no_m5_triggers_exist(self):
        no_structure_m5 = {
            "indicators": _analysis()["indicators"],
            "market_structure": {
                "trend": "BEARISH",
                "regime_trend": "BEARISH",
                "trend_state": "CONFIRMED_BEARISH",
                "trend_state_direction": "BEARISH",
                "structure_events": [],
                "breakout_status": "NONE",
                "candlestick_patterns": [],
                "fair_value_gaps": [],
                "order_blocks": [],
            },
        }
        _, prompt = PromptGenerator.generate(
            symbol="USDCAD",
            timeframe="M5",
            analysis_data=no_structure_m5,
            m15_analysis=_analysis(),
            h1_analysis=_analysis(),
            h4_analysis=_analysis(),
            account_info={},
            open_positions=[],
            trade_history=[],
            calendar_events=[],
        )

        self.assertIn("M5 Entry Triggers Status: NONE", prompt)
        self.assertIn("YOU MUST CHOOSE HOLD", prompt)


if __name__ == "__main__":
    unittest.main()

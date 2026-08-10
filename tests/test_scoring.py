import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.scoring import DecisionScoringEngine


def _analysis():
    return {
        "symbol": "EURUSD",
        "indicators": {
            "rsi_14": 55.0,
            "adx_14": 25.0,
            "macd": {"diff": 0.1},
            "current_price": 1.1,
            "atr_14_pips": 10.0,
            "bollinger_bands": {"width_pct": 0.1},
        },
        "market_structure": {
            "trend": "BULLISH",
            "order_blocks": [],
            "liquidity_zones": {},
            "structure_events": [],
            "breakout_status": "NONE",
        },
    }


def _analysis_with_directions(state_direction=None, legacy_trend=None):
    analysis = _analysis()
    structure = analysis["market_structure"]
    if state_direction is not None:
        structure["trend_state_direction"] = state_direction
    if legacy_trend is not None:
        structure["trend"] = legacy_trend
    return analysis


class RiskRewardScoringTests(unittest.TestCase):
    def _score(self, take_profit):
        analysis = _analysis()
        with patch(
            "core.scoring.settings",
            SimpleNamespace(min_risk_reward_ratio=1.47),
        ):
            return DecisionScoringEngine.calculate_trade_quality_score(
                "BUY",
                {
                    "action": "BUY",
                    "entry": 1.10000,
                    "stop_loss": 1.00000,
                    "take_profit": take_profit,
                },
                analysis,
                analysis,
                analysis,
                analysis,
                [],
            )

    def test_configured_147_boundary_receives_accepted_risk_score(self):
        self.assertEqual(self._score(1.24700)["risk_score"], 75.0)

    def test_value_below_configured_boundary_keeps_lower_risk_score(self):
        self.assertEqual(self._score(1.24600)["risk_score"], 40.0)


class DirectionAndSessionScoringTests(unittest.TestCase):
    def test_normal_mode_prefers_canonical_state_direction(self):
        bullish = _analysis_with_directions("BULLISH", "BEARISH")

        score, factors = DecisionScoringEngine.calculate_confluence_score(
            "BUY",
            bullish,
            bullish,
            bullish,
            bullish,
        )

        self.assertTrue(factors["trend_align"])
        self.assertTrue(factors["tf_agreement"])
        self.assertGreater(score, 0.0)

        quality = DecisionScoringEngine.calculate_trade_quality_score(
            "BUY",
            {
                "action": "BUY",
                "entry": 1.10000,
                "stop_loss": 1.00000,
                "take_profit": 1.25000,
            },
            bullish,
            bullish,
            bullish,
            bullish,
            [],
        )
        self.assertEqual(quality["trend_score"], 100.0)

    def test_reversal_mode_uses_canonical_state_direction(self):
        bullish = _analysis_with_directions("BULLISH", "BEARISH")

        _, factors = DecisionScoringEngine.calculate_confluence_score(
            "BUY",
            bullish,
            bullish,
            bullish,
            bullish,
            strategy_mode="CONFIRMED_REVERSAL",
        )

        self.assertTrue(factors["trend_align"])
        self.assertTrue(factors["tf_agreement"])

        quality = DecisionScoringEngine.calculate_trade_quality_score(
            "BUY",
            {
                "action": "BUY",
                "entry": 1.10000,
                "stop_loss": 1.00000,
                "take_profit": 1.25000,
                "_strategy": {"mode": "CONFIRMED_REVERSAL"},
            },
            bullish,
            bullish,
            bullish,
            bullish,
            [],
        )
        self.assertEqual(quality["trend_score"], 100.0)

    def test_legacy_trend_remains_a_fallback(self):
        bullish = _analysis_with_directions(legacy_trend="BULLISH")

        _, factors = DecisionScoringEngine.calculate_confluence_score(
            "BUY",
            bullish,
            bullish,
            bullish,
            bullish,
        )

        self.assertTrue(factors["trend_align"])
        self.assertTrue(factors["tf_agreement"])

    def test_configured_sessions_are_recognized(self):
        bullish = _analysis_with_directions("BULLISH", "BULLISH")

        for session in ("ASIA", "LONDON", "NEW_YORK", "NEWYORK", "US"):
            with self.subTest(session=session):
                _, factors = DecisionScoringEngine.calculate_confluence_score(
                    "BUY",
                    bullish,
                    bullish,
                    bullish,
                    bullish,
                    session_info=session,
                )

                self.assertTrue(factors["session_confluence"])

    def test_overlapping_supplied_sessions_are_recognized(self):
        bullish = _analysis_with_directions("BULLISH", "BULLISH")

        _, factors = DecisionScoringEngine.calculate_confluence_score(
            "BUY",
            bullish,
            bullish,
            bullish,
            bullish,
            session_info=["LONDON", "NEW_YORK"],
        )

        self.assertTrue(factors["session_confluence"])

    def test_unknown_session_has_no_confluence(self):
        bullish = _analysis_with_directions("BULLISH", "BULLISH")

        _, factors = DecisionScoringEngine.calculate_confluence_score(
            "BUY",
            bullish,
            bullish,
            bullish,
            bullish,
            session_info="UNKNOWN",
        )

        self.assertFalse(factors["session_confluence"])

    def test_weak_adx_direction_labels_do_not_receive_full_trend_score(self):
        m5 = _analysis_with_directions("BULLISH", "BULLISH")
        m15 = _analysis_with_directions("BULLISH", "BULLISH")
        h1 = _analysis_with_directions("BULLISH", "BULLISH")
        h4 = _analysis_with_directions("BULLISH", "BULLISH")
        m5["indicators"]["adx_14"] = 26.0
        m15["indicators"]["adx_14"] = 27.2
        h1["indicators"]["adx_14"] = 12.7
        h4["indicators"]["adx_14"] = 18.5

        quality = DecisionScoringEngine.calculate_trade_quality_score(
            "BUY",
            {
                "action": "BUY",
                "entry": 1.15291,
                "stop_loss": 1.15211,
                "take_profit": 1.15436,
            },
            m5,
            m15,
            h1,
            h4,
            [],
        )

        self.assertGreater(quality["trend_score"], 75.0)
        self.assertLess(quality["trend_score"], 100.0)


if __name__ == "__main__":
    unittest.main()

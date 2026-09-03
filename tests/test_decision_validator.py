import unittest
from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from core.validator import DecisionValidator


class DecisionValidatorTests(unittest.TestCase):
    def test_hold_does_not_require_prices(self):
        ok, decision, error = DecisionValidator.validate_decision(
            {"action": "HOLD", "confidence": 0.6, "reasoning": "mixed"}
        )
        self.assertTrue(ok, error)
        self.assertIsNone(decision["stop_loss"])

    def test_close_requires_exact_ticket(self):
        ok, _, _ = DecisionValidator.validate_decision(
            {"action": "CLOSE", "confidence": 0.8, "ticket_to_close": None}
        )
        self.assertFalse(ok)
        ok, decision, error = DecisionValidator.validate_decision(
            {"action": "CLOSE", "confidence": 0.8, "ticket_to_close": 123}
        )
        self.assertTrue(ok, error)
        self.assertEqual(decision["ticket_to_close"], 123)

    def test_model_lot_is_discarded(self):
        ok, decision, error = DecisionValidator.validate_decision({
            "action": "BUY", "confidence": 0.75, "entry": 100,
            "stop_loss": 99, "take_profit": 102, "lot_size": 50,
        })
        self.assertTrue(ok, error)
        self.assertNotIn("lot_size", decision)

    def test_model_risk_percentage_is_replaced_by_configuration(self):
        with patch("core.validator.settings", replace(settings, risk_percent=10.0)):
            ok, decision, error = DecisionValidator.validate_decision({
                "action": "SELL",
                "confidence": 0.85,
                "risk_percentage": 1.0,
                "reasoning": "Direction only",
            })

        self.assertTrue(ok, error)
        self.assertEqual(decision["risk_percentage"], 10.0)

    def test_entry_rejects_hallucinated_evidence_id(self):
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": [
                    "M5_BOS_BEARISH_20260723T0720",
                    "M15_TREND_BEARISH",
                ],
            },
            allowed_evidence_ids=["M15_TREND_BEARISH"],
        )

        self.assertFalse(ok)
        self.assertIn("unavailable evidence", error)

    def test_entry_requires_directional_m5_trigger_id(self):
        allowed = ["M5_TREND_BEARISH", "M15_TREND_BEARISH"]
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": allowed,
            },
            allowed_evidence_ids=allowed,
        )

        self.assertFalse(ok)
        self.assertIn("requires a supplied directional M5", error)

    def test_entry_accepts_verified_m5_trigger_id(self):
        allowed = [
            "M5_BOS_BEARISH_20260723T0720",
            "M15_TREND_BEARISH",
            "H1_TREND_BEARISH",
        ]
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": allowed,
            },
            allowed_evidence_ids=allowed,
        )

        self.assertTrue(ok, error)
        self.assertEqual(decision["evidence_ids"], allowed)

    def test_entry_cannot_leave_deterministic_direction_contract(self):
        allowed = [
            "M5_BOS_BULLISH_20260811T1340",
            "M5_BOS_BEARISH_20260811T1340",
        ]
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": ["M5_BOS_BEARISH_20260811T1340"],
            },
            allowed_evidence_ids=allowed,
            permitted_actions=("BUY",),
        )

        self.assertFalse(ok)
        self.assertIn("deterministic entry contract", error)

    def test_entry_resolves_descriptive_breakout_alias_to_allowed_m5_id(self):
        allowed = [
            "M5_TREND_BEARISH",
            "M5_BREAKOUT_BEARISH",
            "H4_BREAKOUT_BEARISH",
        ]
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": ["M5_TREND_BEARISH", "BEARISH_BREAKOUT"],
            },
            allowed_evidence_ids=allowed,
        )

        self.assertTrue(ok, error)
        self.assertEqual(
            decision["evidence_ids"],
            ["M5_TREND_BEARISH", "M5_BREAKOUT_BEARISH"],
        )

    def test_entry_resolves_reordered_timeframe_breakout_alias(self):
        allowed = ["M5_BREAKOUT_BULLISH"]
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "BUY",
                "confidence": 0.85,
                "evidence_ids": ["M5_BULLISH_BREAKOUT"],
            },
            allowed_evidence_ids=allowed,
        )

        self.assertTrue(ok, error)
        self.assertEqual(decision["evidence_ids"], allowed)

    def test_alias_cannot_create_evidence_that_analysis_did_not_supply(self):
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": ["BEARISH_BREAKOUT"],
            },
            allowed_evidence_ids=["M5_TREND_BEARISH"],
        )

        self.assertFalse(ok)
        self.assertIn("unavailable evidence", error)

    def test_ambiguous_undated_bos_alias_is_rejected(self):
        allowed = [
            "M5_BOS_BEARISH_20260729T0805",
            "M5_BOS_BEARISH_20260729T0810",
        ]
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": ["BEARISH_BOS"],
            },
            allowed_evidence_ids=allowed,
        )

        self.assertFalse(ok)
        self.assertIn("unavailable evidence", error)

    def test_hold_discards_noncanonical_evidence_commentary(self):
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "evidence_ids": ["rsi is bullish"],
            },
            allowed_evidence_ids=[],
        )

        self.assertTrue(ok, error)
        self.assertEqual(decision["evidence_ids"], [])

    def test_close_requires_verified_opposing_trigger(self):
        allowed = ["M5_TREND_BEARISH", "M15_TREND_BEARISH"]
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "CLOSE",
                "confidence": 0.8,
                "ticket_to_close": 123,
                "evidence_ids": allowed,
            },
            allowed_evidence_ids=allowed,
            close_position_side="BUY",
        )

        self.assertFalse(ok)
        self.assertIn("verified opposing M5", error)

    def test_close_accepts_verified_opposing_trigger(self):
        allowed = [
            "M5_CHOCH_BEARISH_20260723T0830",
            "M15_TREND_BEARISH",
        ]
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "CLOSE",
                "confidence": 0.8,
                "ticket_to_close": 123,
                "evidence_ids": allowed,
            },
            allowed_evidence_ids=allowed,
            close_position_side="BUY",
        )

        self.assertTrue(ok, error)
        self.assertEqual(decision["ticket_to_close"], 123)

    def test_fast_close_accepts_verified_opposing_m1_trigger(self):
        allowed = [
            "M1_CHOCH_BEARISH_20260723T0831",
            "M5_TREND_BULLISH",
        ]
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "CLOSE",
                "confidence": 0.8,
                "ticket_to_close": 123,
                "evidence_ids": [allowed[0]],
            },
            allowed_evidence_ids=allowed,
            close_position_side="BUY",
            close_trigger_timeframes=("M1", "M5"),
        )

        self.assertTrue(ok, error)
        self.assertEqual(decision["ticket_to_close"], 123)

    def test_m1_trigger_cannot_authorize_a_new_entry(self):
        allowed = [
            "M1_BOS_BEARISH_20260723T0831",
            "M15_TREND_BEARISH",
        ]
        ok, _, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.8,
                "evidence_ids": allowed,
            },
            allowed_evidence_ids=allowed,
        )

        self.assertFalse(ok)
        self.assertIn("directional M5", error)


if __name__ == "__main__":
    unittest.main()

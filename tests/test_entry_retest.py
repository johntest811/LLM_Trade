import unittest
from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from core.entry_retest import annotate_retest_continuation
from core.evidence import build_evidence_ids
from core.validator import DecisionValidator


def _analysis(
    timeframe,
    *,
    state="CONFIRMED_BEARISH",
    direction="BEARISH",
    adx=30.0,
    adx_delta=0.5,
    body=-0.25,
    candle_return=-0.20,
    candle_range=0.8,
):
    return {
        "timeframe": timeframe,
        "timestamp": "2026-07-29 15:05:00",
        "indicators": {
            "adx_14": adx,
            "adx_delta": adx_delta,
            "candle_body_atr_signed": body,
            "candle_return_atr": candle_return,
            "candle_range_atr": candle_range,
        },
        "market_structure": {
            "trend": direction,
            "trend_state": state,
            "trend_state_direction": direction,
            "structure_events": [],
            "breakout_status": "None",
        },
    }


class EntryRetestTests(unittest.TestCase):
    def test_exact_pullback_resumption_creates_verifiable_entry_evidence(self):
        m5 = _analysis("M5")
        m15 = _analysis("M15")
        h1 = _analysis("H1")
        configured = replace(
            settings,
            retest_continuation_enabled=True,
            retest_min_resumption_atr=0.10,
            entry_min_adx=20.0,
            confirmation_min_adx=18.0,
        )

        with patch("core.entry_retest.settings", configured):
            event = annotate_retest_continuation(
                m5,
                m15,
                h1,
                {"M5": "PULLBACK_IN_BEARISH_TREND"},
            )

        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "BEARISH")
        allowed = build_evidence_ids(
            {"M5": m5, "M15": m15, "H1": h1}
        )
        retest_id = next(
            value for value in allowed if value.startswith("M5_RETEST_BEARISH_")
        )
        ok, decision, error = DecisionValidator.validate_decision(
            {
                "action": "SELL",
                "confidence": 0.85,
                "evidence_ids": [retest_id],
                "reasoning": "Verified pullback resumption.",
            },
            allowed_evidence_ids=allowed,
        )
        self.assertTrue(ok, error)
        self.assertEqual(decision["evidence_ids"], [retest_id])

    def test_retest_does_not_persist_without_the_exact_transition(self):
        m5 = _analysis("M5")
        m5["market_structure"]["retest_continuation"] = {
            "direction": "BEARISH",
            "time": "old",
        }
        configured = replace(settings, retest_continuation_enabled=True)

        with patch("core.entry_retest.settings", configured):
            event = annotate_retest_continuation(
                m5,
                _analysis("M15"),
                _analysis("H1"),
                {"M5": "CONFIRMED_BEARISH"},
            )

        self.assertIsNone(event)
        self.assertNotIn(
            "retest_continuation", m5["market_structure"]
        )

    def test_weak_or_extended_resumption_does_not_create_evidence(self):
        configured = replace(
            settings,
            retest_continuation_enabled=True,
            retest_min_resumption_atr=0.10,
            entry_min_adx=20.0,
            entry_max_candle_range_atr=1.50,
        )
        previous = {"M5": "PULLBACK_IN_BEARISH_TREND"}

        weak = _analysis("M5", adx=19.9)
        extended = _analysis(
            "M5", body=-1.8, candle_return=-1.7, candle_range=2.0
        )
        with patch("core.entry_retest.settings", configured):
            self.assertIsNone(
                annotate_retest_continuation(
                    weak, _analysis("M15"), _analysis("H1"), previous
                )
            )
            self.assertIsNone(
                annotate_retest_continuation(
                    extended, _analysis("M15"), _analysis("H1"), previous
                )
            )


if __name__ == "__main__":
    unittest.main()

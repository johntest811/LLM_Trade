import unittest
from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from core.engine import TradingEngine
from core.evidence import (
    build_evidence_ids,
    has_directional_m5_trigger,
    has_directional_trigger,
)
from core.range_reversion import annotate_range_reversion
from core.scoring import DecisionScoringEngine
from core.validator import DecisionValidator
from risk.manager import RiskManager


def _analysis(
    timeframe="M5",
    *,
    direction="NEUTRAL",
    adx=14.0,
    price=99.05,
    rsi=32.0,
    stoch=18.0,
    body=0.20,
):
    return {
        "symbol": "EURUSD",
        "timeframe": timeframe,
        "timestamp": "2026-08-04 03:20:00",
        "indicators": {
            "current_price": price,
            "atr_14": 1.0,
            "atr_14_pips": 10.0,
            "adx_14": adx,
            "adx_delta": -0.2,
            "rsi_14": rsi,
            "candle_body_atr_signed": body,
            "candle_return_atr": body,
            "candle_range_atr": 0.8,
            "opening_gap_atr": 0.0,
            "stochastic": {"k": stoch, "d": stoch},
            "bollinger_bands": {
                "lower": 99.0,
                "middle": 100.0,
                "upper": 101.0,
                "width_pct": 0.2,
            },
            "macd": {"diff": 0.0},
            "ema_9": 99.2,
            "ema_21": 99.4,
        },
        "market_structure": {
            "trend": direction,
            "trend_state_direction": direction,
            "trend_state": "NEUTRAL" if direction == "NEUTRAL" else f"CONFIRMED_{direction}",
            "breakout_status": "NONE",
            "structure_events": [],
            "support": 98.8,
            "resistance": 101.2,
            "demand_zones": [],
            "supply_zones": [],
            "order_blocks": [],
            "fair_value_gaps": [],
            "liquidity_zones": {},
            "candlestick_patterns": [],
        },
    }


class RangeReversionTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            settings,
            range_reversion_enabled=True,
            entry_min_adx=20.0,
            range_band_tolerance_atr=0.10,
            range_max_band_overshoot_atr=0.35,
            range_buy_max_rsi=35.0,
            range_sell_min_rsi=65.0,
            range_buy_max_stoch=25.0,
            range_sell_min_stoch=75.0,
            range_min_reversal_body_atr=0.05,
            range_min_target_distance_atr=0.75,
            range_confirmation_max_adx=24.0,
            range_h4_opposition_min_adx=25.0,
            range_invalidation_atr=0.25,
        )

    def test_completed_inward_extreme_becomes_verified_buy_evidence(self):
        m5 = _analysis()
        confirmations = [_analysis(tf) for tf in ("M15", "H1", "H4")]
        with patch("core.range_reversion.settings", self.config):
            setup = annotate_range_reversion(m5, *confirmations)

        self.assertTrue(setup["candidate"])
        self.assertTrue(setup["eligible"])
        self.assertEqual(setup["direction"], "BUY")
        evidence = build_evidence_ids(
            dict(zip(("M5", "M15", "H1", "H4"), (m5, *confirmations)))
        )
        self.assertIn(setup["evidence_id"], evidence)
        self.assertTrue(has_directional_m5_trigger(evidence, "BUY"))
        self.assertFalse(
            has_directional_trigger(evidence, "BUY", timeframes=("M5",))
        )
        valid, decision, reason = DecisionValidator.validate_decision(
            {
                "action": "BUY",
                "confidence": 0.82,
                "evidence_ids": [setup["evidence_id"]],
                "reasoning": "Verified range reversal.",
                "trade_management": "Exit on invalidation.",
            },
            allowed_evidence_ids=evidence,
        )
        self.assertTrue(valid, reason)
        self.assertEqual(decision["action"], "BUY")

    def test_extreme_without_inward_candle_is_not_a_candidate(self):
        m5 = _analysis(body=-0.20)
        with patch("core.range_reversion.settings", self.config):
            setup = annotate_range_reversion(m5)
        self.assertFalse(setup["candidate"])
        self.assertFalse(setup["eligible"])

    def test_strong_m15_trend_prevents_range_eligibility(self):
        m5 = _analysis()
        m15 = _analysis("M15", direction="BEARISH", adx=29.0)
        with patch("core.range_reversion.settings", self.config):
            setup = annotate_range_reversion(
                m5, m15, _analysis("H1"), _analysis("H4")
            )
        self.assertTrue(setup["candidate"])
        self.assertFalse(setup["eligible"])
        self.assertIn("active trend strength", setup["reason"])

    def test_prefilter_allows_only_eligible_or_pending_range(self):
        m5 = _analysis()
        with patch("core.range_reversion.settings", self.config), patch(
            "core.engine.settings", self.config
        ):
            annotate_range_reversion(m5)
            self.assertEqual(
                TradingEngine._entry_prefilter_reason(
                    m5, allow_pending_range=True
                ),
                "",
            )
            self.assertIn("ADX Prefilter", TradingEngine._entry_prefilter_reason(m5))

    def test_structure_gate_accepts_verified_range_direction_only(self):
        m5 = _analysis()
        confirmations = [_analysis(tf) for tf in ("M15", "H1", "H4")]
        with patch("core.range_reversion.settings", self.config), patch(
            "risk.manager.settings", self.config
        ):
            annotate_range_reversion(m5, *confirmations)
            mode = RiskManager._resolve_strategy_mode(
                "BUY", {}, m5, *confirmations
            )
            ok, reason = RiskManager._check_entry_structure(
                "BUY", m5, *confirmations, strategy_mode=mode
            )
            wrong_ok, _ = RiskManager._check_entry_structure(
                "SELL", m5, *confirmations, strategy_mode=mode
            )

        self.assertEqual(mode, "RANGE_REVERSION")
        self.assertTrue(ok, reason)
        self.assertFalse(wrong_ok)

    def test_verified_range_clears_normal_quality_and_confluence_floors(self):
        m5 = _analysis()
        confirmations = [_analysis(tf) for tf in ("M15", "H1", "H4")]
        decision = {
            "action": "BUY",
            "confidence": 0.82,
            "entry": 99.05,
            "stop_loss": 98.70,
            "take_profit": 100.0,
            "_strategy": {"mode": "RANGE_REVERSION"},
        }
        with patch("core.range_reversion.settings", self.config), patch(
            "core.scoring.settings", self.config
        ):
            annotate_range_reversion(m5, *confirmations)
            confluence, _ = DecisionScoringEngine.calculate_confluence_score(
                "BUY",
                m5,
                *confirmations,
                session_info=["TOKYO"],
                strategy_mode="RANGE_REVERSION",
            )
            quality = DecisionScoringEngine.calculate_trade_quality_score(
                "BUY", decision, m5, *confirmations, calendar_events=None
            )

        self.assertGreaterEqual(confluence, 55.0)
        self.assertGreaterEqual(quality["overall_score"], 55.0)


if __name__ == "__main__":
    unittest.main()

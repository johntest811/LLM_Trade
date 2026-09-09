import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from core.continuation_quality import continuation_entry_check
from core.engine import TradingEngine
from core.opportunities import screen_opportunities
from core.scoring import order_block_proximity
from risk.manager import RiskManager
from tests.test_opportunity_scanning import frames_with_bos


class ContinuationQualityTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(settings, continuation_entry_guard_enabled=True,
                              continuation_min_clearance_atr=.15)
        for module in ("risk.manager", "core.engine"):
            context = patch(module + ".settings", self.config)
            context.start()
            self.addCleanup(context.stop)

    def test_positive_and_negative_cases_are_symmetric_and_scale_free(self):
        for action, direction in (("BUY", "BULLISH"), ("SELL", "BEARISH")):
            for scale in (.00001, .01, 1, 1000):
                with self.subTest(action=action, scale=scale):
                    frames = frames_with_bos(direction)
                    m5 = frames["M5"]
                    for key in ("current_price", "atr_14", "ema_9", "ema_21"):
                        m5["indicators"][key] *= scale
                    m5["market_structure"]["structure_events"][0]["level"] *= scale
                    self.assertTrue(continuation_entry_check(action, m5)[0])
                    m5["indicators"]["macd"]["diff"] *= -1
                    self.assertFalse(continuation_entry_check(action, m5)[0])

    def test_recorded_audusd_entry_is_rejected_for_both_independent_weaknesses(self):
        # Entry snapshot values, not a symbol-specific or date-specific production rule.
        frames = frames_with_bos()
        m5 = frames["M5"]
        m5["indicators"].update(current_price=.72289, atr_14=.00026609451964841894,
            ema_9=.7226036787666081, ema_21=.722425990149719, rsi_14=66.09201495050615,
            macd={"diff": -2.1060713496266563e-6})
        m5["market_structure"]["structure_events"][0]["level"] = .72286
        original = copy.deepcopy(m5)
        allowed, reason = continuation_entry_check("BUY", m5)
        self.assertFalse(allowed)
        self.assertIn("MACD", reason)
        self.assertEqual(m5, original)
        m5["indicators"]["macd"]["diff"] *= -1
        allowed, reason = continuation_entry_check("BUY", m5)
        self.assertFalse(allowed)
        self.assertIn("0.11 ATR", reason)

    def test_missing_or_nonfinite_data_and_unconfirmed_boundary_fail_closed(self):
        for issue in ("missing_macd", "nan", "missing_level", "negative_atr", "tiny_clearance"):
            with self.subTest(issue=issue):
                m5 = frames_with_bos()["M5"]
                if issue == "missing_macd": del m5["indicators"]["macd"]
                if issue == "nan": m5["indicators"]["ema_9"] = float("nan")
                if issue == "missing_level": del m5["market_structure"]["structure_events"][0]["level"]
                if issue == "negative_atr": m5["indicators"]["atr_14"] = -1
                if issue == "tiny_clearance": m5["market_structure"]["structure_events"][0]["level"] = 99.99
                self.assertFalse(continuation_entry_check("BUY", m5)[0])

    def test_discovery_fast_path_and_final_gate_all_reject_disagreeing_macd(self):
        frames = frames_with_bos()
        frames["M5"]["indicators"]["macd"]["diff"] = -.1
        screened = screen_opportunities(frames, {"capital_fit": True})
        self.assertFalse(screened["model_eligible"])
        self.assertIn("Continuation Confirmation", screened["opportunity_rejections"]["BUY"])
        self.assertIsNone(TradingEngine._deterministic_entry_fast_path({"analyses": frames}, ("BUY",)))
        allowed, reason = RiskManager._check_entry_structure("BUY", *(frames[tf] for tf in ("M5", "M15", "H1", "H4")),
                                                           strategy_mode="TREND_CONTINUATION", decision_confidence=.99)
        self.assertFalse(allowed)
        self.assertIn("Continuation Confirmation", reason)

    def test_positive_entry_keeps_opposing_zone_checks_active(self):
        frames = frames_with_bos()
        args = [frames[tf] for tf in ("M5", "M15", "H1", "H4")]
        self.assertTrue(RiskManager._check_entry_structure("BUY", *args, strategy_mode="TREND_CONTINUATION")[0])
        frames["M5"]["market_structure"]["resistance"] = 100.01
        allowed, reason = RiskManager._check_entry_structure("BUY", *args, strategy_mode="TREND_CONTINUATION")
        self.assertFalse(allowed)
        self.assertIn("Opposing Zone", reason)

    def test_verified_retest_does_not_need_to_rebreak_old_level(self):
        m5 = frames_with_bos()["M5"]
        m5["market_structure"]["retest_continuation"] = {"direction": "BULLISH"}
        m5["indicators"]["macd"]["diff"] = -.1
        self.assertTrue(continuation_entry_check("BUY", m5)[0])
        m5["market_structure"]["retest_continuation"]["direction"] = "BEARISH"
        self.assertFalse(continuation_entry_check("BUY", m5)[0])

    def test_order_block_proximity_is_atr_bounded_not_price_percentage(self):
        for action, direction, price in (("BUY", "BULLISH", 100.2), ("SELL", "BEARISH", 98.8)):
            for scale in (.00001, .01, 1, 1000):
                with self.subTest(action=action, scale=scale):
                    analysis = {"indicators": {"current_price": price*scale, "atr_14": scale}}
                    block = {"type": direction, "low": 99*scale, "high": 100*scale}
                    self.assertTrue(order_block_proximity(action, analysis, block))
                    analysis["indicators"]["current_price"] = (101.8 if action == "BUY" else 97.2)*scale
                    self.assertFalse(order_block_proximity(action, analysis, block))


if __name__ == "__main__":
    unittest.main()

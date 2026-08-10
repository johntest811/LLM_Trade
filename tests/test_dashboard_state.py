import json
import unittest

from ui.state import DashboardState


class DashboardStateTests(unittest.TestCase):
    def test_shadow_metrics_are_serialized(self):
        state = DashboardState()
        state.update_shadow(
            enabled=True,
            pending=2,
            resolved=4,
            wins=3,
            losses=1,
            win_rate_pct=75.0,
            expectancy_r=0.42,
        )

        payload = state.to_dict()

        self.assertTrue(payload["shadow"]["enabled"])
        self.assertEqual(payload["shadow"]["resolved"], 4)
        self.assertEqual(payload["shadow"]["expectancy_r"], 0.42)

    def test_symbol_decisions_serialize_independently(self):
        state = DashboardState()
        state.update_symbol_decision(
            "USDJPY",
            stage="DECIDED",
            action="HOLD",
            confidence=0.41,
            candle_time="2026-07-13T12:00:00+00:00",
        )
        state.update_symbol_decision(
            "GBPJPY",
            stage="RISK CHECK",
            action="BUY",
            confidence=0.78,
            gate_reason="Awaiting deterministic validation",
        )
        state.update_symbol_decision("USDJPY", stage="WAITING")

        payload = state.to_dict()
        serialized = json.dumps(payload)

        self.assertIn('"symbol_decisions"', serialized)
        self.assertEqual(set(payload["symbol_decisions"]), {"USDJPY", "GBPJPY"})
        self.assertEqual(payload["symbol_decisions"]["USDJPY"]["stage"], "WAITING")
        self.assertEqual(payload["symbol_decisions"]["USDJPY"]["action"], "HOLD")
        self.assertEqual(payload["symbol_decisions"]["GBPJPY"]["stage"], "RISK CHECK")
        self.assertEqual(payload["symbol_decisions"]["GBPJPY"]["action"], "BUY")
        self.assertTrue(payload["symbol_decisions"]["USDJPY"]["updated_at"])
        self.assertTrue(payload["symbol_decisions"]["GBPJPY"]["updated_at"])

    def test_market_scope_removes_old_session_quotes_and_fits(self):
        state = DashboardState()
        state.update_prices("USDJPY", 160.0, 160.01, 0.001)
        state.update_prices("ETHUSD", 1800.0, 1802.0, 0.01)
        state.update_market_fit("USDJPY", {"status": "CAPITAL FIT"})
        state.update_market_fit("ETHUSD", {"status": "CAPITAL FIT"})
        state.add_tick("USDJPY", 160.0, 160.01)
        state.add_tick("ETHUSD", 1800.0, 1802.0)

        state.retain_market_scope(["ETHUSD"])
        payload = state.to_dict()

        self.assertEqual(set(payload["prices"]), {"ETHUSD"})
        self.assertEqual(set(payload["market_fits"]), {"ETHUSD"})
        self.assertEqual(
            {tick["symbol"] for tick in payload["tick_stream"]},
            {"ETHUSD"},
        )


if __name__ == "__main__":
    unittest.main()

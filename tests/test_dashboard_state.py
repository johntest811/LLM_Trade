import json
import unittest

from ui.state import DashboardState


class DashboardStateTests(unittest.TestCase):
    def test_compact_wire_state_omits_engine_only_planner_payload(self):
        state = DashboardState()
        state.update_market_fit(
            "USDJPY",
            {
                "status": "CAPITAL FIT",
                "capital_fit": True,
                "reason": "Executable",
                "selection_score": 91.0,
                "directions": {
                    "BUY": {
                        "capital_fit": True,
                        "plan": {"entry": 150.0, "private": "engine-only"},
                    },
                    "SELL": {"capital_fit": False, "plan": {"entry": 149.9}},
                },
                "performance": {"trades": list(range(100))},
                "forex_context": {"private": "engine-only"},
            },
        )
        state.add_tick("USDJPY", 150.0, 150.01)

        payload = state.to_dict(compact=True, include_tick_stream=False)
        fit = payload["market_fits"]["USDJPY"]

        self.assertEqual(fit["selection_score"], 91.0)
        self.assertEqual(fit["directions"]["BUY"], {"capital_fit": True})
        self.assertNotIn("performance", fit)
        self.assertNotIn("forex_context", fit)
        self.assertEqual(payload["tick_stream"], [])

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
            gate_breakdown=[
                {
                    "gate": "Structure Gate",
                    "resolved": 4,
                    "wins": 1,
                    "losses": 3,
                    "expectancy_r": -0.25,
                }
            ],
            evidence_window_hours=48,
            direction_breakdown=[
                {
                    "action": "BUY",
                    "candidates": 12,
                    "rejected": 10,
                    "executed": 2,
                    "approval_rate_pct": 16.7,
                    "shadow_resolved": 8,
                    "shadow_positive": 5,
                    "shadow_expectancy_r": 0.12,
                }
            ],
        )

        payload = state.to_dict()

        self.assertTrue(payload["shadow"]["enabled"])
        self.assertEqual(payload["shadow"]["resolved"], 4)
        self.assertEqual(payload["shadow"]["expectancy_r"], 0.42)
        self.assertEqual(
            payload["shadow"]["gate_breakdown"][0]["gate"],
            "Structure Gate",
        )
        self.assertEqual(payload["shadow"]["evidence_window_hours"], 48)
        self.assertEqual(
            payload["shadow"]["direction_breakdown"][0]["action"],
            "BUY",
        )

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

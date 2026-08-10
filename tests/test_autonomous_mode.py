import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.engine import TradingEngine


ACCOUNT = {
    "login": 1001,
    "server": "Pepperstone-Demo",
    "company": "Pepperstone",
    "trade_mode": 0,
    "trade_mode_name": "DEMO",
    "account_trade_allowed": True,
    "expert_trading_allowed": True,
    "terminal_connected": True,
    "terminal_trade_allowed": True,
    "tradeapi_disabled": False,
}


class AutonomousModeTests(unittest.TestCase):
    def test_enable_requires_explicit_account_mode_phrase(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.is_running = True
        engine.conn = SimpleNamespace(
            get_account_info=AsyncMock(return_value=dict(ACCOUNT))
        )

        ok, message = asyncio.run(
            engine.enable_autonomous_mode("ENABLE AUTONOMOUS LIVE")
        )

        self.assertFalse(ok)
        self.assertEqual(message, "Type ENABLE AUTONOMOUS DEMO to confirm")

    def test_enable_persists_exact_account_identity(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.is_running = True
        engine.conn = SimpleNamespace(
            get_account_info=AsyncMock(return_value=dict(ACCOUNT))
        )
        engine.arm_entries = AsyncMock(return_value=(True, "Entries armed"))
        engine.db = SimpleNamespace(set_cache=AsyncMock(return_value=True))
        engine._update_execution_readiness = MagicMock()
        engine.log = MagicMock()
        engine.autonomous_enabled = False
        engine._autonomous_account_identity = None
        engine._last_autonomy_health_state = ""
        fake_dashboard = SimpleNamespace(update_automation=MagicMock())

        with patch("core.engine.dashboard_state", fake_dashboard):
            ok, message = asyncio.run(
                engine.enable_autonomous_mode("ENABLE AUTONOMOUS DEMO")
            )

        self.assertTrue(ok)
        self.assertIn("Autonomous DEMO mode enabled", message)
        engine.db.set_cache.assert_awaited_once()
        cache_key, payload = engine.db.set_cache.await_args.args
        self.assertEqual(
            cache_key,
            TradingEngine._autonomous_cache_key(ACCOUNT),
        )
        self.assertTrue(payload["enabled"])
        self.assertEqual(
            payload["account_identity"],
            TradingEngine._account_identity(ACCOUNT),
        )
        self.assertTrue(engine.autonomous_enabled)

    def test_restart_restore_rejects_different_account_identity(self):
        engine = TradingEngine.__new__(TradingEngine)
        other = {**ACCOUNT, "login": 2002}
        engine.db = SimpleNamespace(
            get_cache=AsyncMock(
                return_value={
                    "enabled": True,
                    "account_identity": TradingEngine._account_identity(other),
                }
            )
        )
        engine.autonomous_enabled = False
        engine._autonomous_account_identity = None
        fake_dashboard = SimpleNamespace(update_automation=MagicMock())

        with patch("core.engine.dashboard_state", fake_dashboard):
            asyncio.run(
                engine._restore_autonomous_authorization(
                    ACCOUNT, llm_ready=True
                )
            )

        self.assertFalse(engine.autonomous_enabled)
        self.assertIsNone(engine._autonomous_account_identity)

    def test_restart_restore_arms_only_after_health_checks_pass(self):
        identity = TradingEngine._account_identity(ACCOUNT)
        engine = TradingEngine.__new__(TradingEngine)
        engine.db = SimpleNamespace(
            get_cache=AsyncMock(
                return_value={
                    "enabled": True,
                    "account_identity": identity,
                }
            )
        )
        engine._history_ready = True
        engine._position_risk_healthy = True
        engine.autonomous_enabled = False
        engine._autonomous_account_identity = None
        engine._last_autonomy_health_state = ""
        engine._activate_autonomous_entries = MagicMock()
        engine.log = MagicMock()

        asyncio.run(
            engine._restore_autonomous_authorization(
                ACCOUNT, llm_ready=True
            )
        )

        self.assertTrue(engine.autonomous_enabled)
        self.assertEqual(engine._autonomous_account_identity, identity)
        engine._activate_autonomous_entries.assert_called_once()

    def test_operator_disarm_clears_persistent_autonomy(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.autonomous_enabled = True
        engine._autonomous_account_identity = TradingEngine._account_identity(
            ACCOUNT
        )
        engine._last_autonomy_health_state = "ACTIVE"
        engine.conn = SimpleNamespace(
            get_account_info=AsyncMock(return_value=dict(ACCOUNT))
        )
        engine.db = SimpleNamespace(set_cache=AsyncMock(return_value=True))
        engine.disarm_entries = MagicMock()
        engine._update_execution_readiness = MagicMock()
        engine.log = MagicMock()
        fake_dashboard = SimpleNamespace(update_automation=MagicMock())

        with patch("core.engine.dashboard_state", fake_dashboard):
            ok, _ = asyncio.run(engine.operator_disarm_entries())

        self.assertTrue(ok)
        self.assertFalse(engine.autonomous_enabled)
        engine.disarm_entries.assert_called_once()
        _, payload = engine.db.set_cache.await_args.args
        self.assertFalse(payload["enabled"])


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mt5.connection import MT5ConnectionManager


class MT5ConnectionLifecycleTests(unittest.TestCase):
    def test_stale_cached_connection_performs_native_reinitialize(self):
        manager = MT5ConnectionManager()
        manager._connected = True
        fake_settings = SimpleNamespace(
            mt5_path="",
            mt5_account=None,
            mt5_password="",
            mt5_server="",
            expected_broker="Pepperstone",
        )
        account = SimpleNamespace(
            trade_mode=2,
            company="Pepperstone Markets Limited",
            server="Pepperstone-Test",
            login=123456,
        )

        with (
            patch("mt5.connection.settings", fake_settings),
            patch(
                "mt5.connection.mt5.terminal_info",
                return_value=SimpleNamespace(connected=False),
            ),
            patch("mt5.connection.mt5.shutdown") as shutdown,
            patch("mt5.connection.mt5.initialize", return_value=True) as initialize,
            patch("mt5.connection.mt5.account_info", return_value=account),
        ):
            connected = asyncio.run(manager.initialize())

        self.assertTrue(connected)
        self.assertTrue(manager._connected)
        shutdown.assert_called_once_with()
        initialize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

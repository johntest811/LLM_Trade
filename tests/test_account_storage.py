import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from database.storage import TradingDatabase


class AccountScopedStorageTests(unittest.TestCase):
    def test_schema_initialization_closes_connection_when_migration_fails(self):
        connection = sqlite3.connect(":memory:")
        with patch("database.storage.sqlite3.connect", return_value=connection), patch.object(
            TradingDatabase,
            "_init_schema",
            side_effect=RuntimeError("migration failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "migration failed"):
                TradingDatabase("unused.db")

        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_same_position_id_is_isolated_between_accounts(self):
        row = {
            "position_id": 42,
            "open_time": "2026-01-01T00:00:00+00:00",
            "close_time": "2026-01-01T00:05:00+00:00",
            "symbol": "USDJPY",
            "direction": "BUY",
            "volume": 0.01,
            "open_price": 150.0,
            "close_price": 150.1,
            "gross_profit": 0.1,
            "commission": 0.0,
            "swap": 0.0,
            "fee": 0.0,
            "net_profit": 0.1,
            "magic": 202600,
            "close_reason": "TAKE_PROFIT",
            "profit_pips": 10.0,
            "initial_risk_pips": 8.0,
            "initial_risk_usd": 0.50,
            "mfe_pips": 12.0,
            "mae_pips": -3.0,
            "mfe_usd": 0.60,
            "mae_usd": -0.15,
            "rr_achieved": 0.20,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database = TradingDatabase(str(Path(temp_dir) / "accounts.db"))
            asyncio.run(database.upsert_closed_positions([row], account_login=1001))
            second = dict(row, net_profit=-0.2, close_reason="STOP_LOSS")
            asyncio.run(database.upsert_closed_positions([second], account_login=2002))

            first_rows = asyncio.run(
                database.get_closed_positions(account_login=1001)
            )
            second_rows = asyncio.run(
                database.get_closed_positions(account_login=2002)
            )

        self.assertEqual(len(first_rows), 1)
        self.assertEqual(len(second_rows), 1)
        self.assertEqual(first_rows[0]["net_profit"], 0.1)
        self.assertEqual(second_rows[0]["net_profit"], -0.2)
        self.assertEqual(first_rows[0]["mfe_pips"], 12.0)

    def test_initial_risk_baseline_is_insert_once_and_account_scoped(self):
        baseline = {
            "ticket": 77,
            "symbol": "USDJPY",
            "direction": "BUY",
            "entry_price": 150.0,
            "initial_sl": 149.9,
            "initial_risk_pips": 10.0,
            "initial_risk_usd": 0.67,
            "initial_volume": 0.01,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database = TradingDatabase(str(Path(temp_dir) / "risk.db"))
            first = asyncio.run(
                database.set_position_risk_baseline("Pepperstone|Demo|1001|0", baseline)
            )
            changed = dict(baseline, initial_sl=150.05, initial_risk_pips=5.0)
            canonical = asyncio.run(
                database.set_position_risk_baseline("Pepperstone|Demo|1001|0", changed)
            )
            other_account = asyncio.run(
                database.set_position_risk_baseline("Pepperstone|Live|1001|2", changed)
            )

        self.assertEqual(first["initial_risk_pips"], 10.0)
        self.assertEqual(canonical["initial_risk_pips"], 10.0)
        self.assertEqual(canonical["initial_sl"], 149.9)
        self.assertEqual(other_account["initial_risk_pips"], 5.0)

    def test_position_excursions_only_expand_mfe_and_mae(self):
        baseline = {
            "ticket": 77,
            "symbol": "USDJPY",
            "direction": "BUY",
            "entry_price": 150.0,
            "initial_sl": 149.9,
            "initial_risk_pips": 10.0,
            "initial_risk_usd": 0.67,
            "initial_volume": 0.01,
        }
        scope = "Pepperstone|Demo|1001|0"
        with tempfile.TemporaryDirectory() as temp_dir:
            database = TradingDatabase(str(Path(temp_dir) / "risk.db"))
            asyncio.run(database.set_position_risk_baseline(scope, baseline))
            asyncio.run(database.update_position_excursion(
                scope,
                77,
                profit_pips=4.0,
                profit_usd=0.20,
                observed_at="2026-01-01T00:01:00+00:00",
            ))
            asyncio.run(database.update_position_excursion(
                scope,
                77,
                profit_pips=-3.0,
                profit_usd=-0.15,
                observed_at="2026-01-01T00:02:00+00:00",
            ))
            asyncio.run(database.update_position_excursion(
                scope,
                77,
                profit_pips=1.0,
                profit_usd=0.05,
                observed_at="2026-01-01T00:03:00+00:00",
            ))
            row = asyncio.run(
                database.get_position_risk_baseline(scope, 77)
            )

        self.assertEqual(row["mfe_pips"], 4.0)
        self.assertEqual(row["mae_pips"], -3.0)
        self.assertEqual(row["mfe_usd"], 0.20)
        self.assertEqual(row["mae_usd"], -0.15)

    def test_strategy_close_reason_is_account_scoped_and_survives_sync(self):
        row = {
            "position_id": 88,
            "open_time": "2026-01-01T00:00:00+00:00",
            "close_time": "2026-01-01T00:05:00+00:00",
            "symbol": "AUDUSD",
            "direction": "BUY",
            "volume": 0.02,
            "open_price": 0.71752,
            "close_price": 0.71797,
            "gross_profit": 0.90,
            "commission": 0.0,
            "swap": 0.0,
            "fee": 0.0,
            "net_profit": 0.90,
            "magic": 202600,
            "close_reason": "EXPERT",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            database = TradingDatabase(str(Path(temp_dir) / "reasons.db"))
            self.assertTrue(asyncio.run(
                database.set_strategy_close_reason(1001, 88, "profit_giveback")
            ))
            reasons = asyncio.run(
                database.get_strategy_close_reasons(1001, [88])
            )
            other = asyncio.run(
                database.get_strategy_close_reasons(2002, [88])
            )
            first = dict(row, close_reason="PROFIT_GIVEBACK")
            asyncio.run(database.upsert_closed_positions([first], 1001))
            asyncio.run(database.upsert_closed_positions([row], 1001))
            closed = asyncio.run(
                database.get_closed_positions(account_login=1001)
            )

        self.assertEqual(reasons, {88: "PROFIT_GIVEBACK"})
        self.assertEqual(other, {})
        self.assertEqual(closed[0]["close_reason"], "PROFIT_GIVEBACK")


if __name__ == "__main__":
    unittest.main()

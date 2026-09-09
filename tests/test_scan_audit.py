import asyncio
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app_config.settings import settings
from core.engine import TradingEngine
from database.replay_logger import TradeReplayLogger
from ui.dashboard import app, get_scan_audit


class ScanAuditTests(unittest.TestCase):
    def test_cursor_pagination_has_no_duplicates_and_preserves_research(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TradeReplayLogger(str(Path(directory) / "scan.db"))
            watch = {"candidate": True, "live_eligible": False, "strategy_version": "test-v1"}
            logger.log_scan_observations("A", "DISCOVERY", [{"symbol": "TEST", "research_watch": watch} for _ in range(5)])
            first = logger.get_scan_observations("A", "TEST", 2)
            second = logger.get_scan_observations("A", "TEST", 2, first[-1]["id"])
            third = logger.get_scan_observations("A", "TEST", 2, second[-1]["id"])
            self.assertEqual(len({row["id"] for row in first + second + third}), 5)
            self.assertEqual(first[0]["snapshot"]["research_watch"], watch)
            self.assertEqual(len(first[0]["account_scope_id"]), 16)
            self.assertEqual(logger.get_scan_observations("B", "TEST", 2, first[-1]["id"]), [])
            with self.assertRaises(ValueError): logger.get_scan_observations("A", "", 2, 0)

    def test_endpoint_refuses_account_switch_during_read(self):
        engine = SimpleNamespace(_active_account_identity={"login": 1}, _account_scope=lambda account: "A")
        def reader(*args):
            engine._active_account_identity = {"login": 2}
            return [{"symbol": "OLD_ACCOUNT"}]
        engine.replay_logger = SimpleNamespace(get_scan_observations=reader)
        with patch.object(app.state, "engine", engine, create=True):
            response = asyncio.run(get_scan_audit("", 1, 20))
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("OLD_ACCOUNT", response.body.decode())

    def test_persistent_audit_is_account_scoped_and_keeps_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scan.db")
            logger = TradeReplayLogger(path)
            row = {"symbol":"EURJPY","opportunity_bar":"2032-02-29T03:25:00Z",
                   "status":"PREFILTERED","reason":"ADX decline","selected":False}
            self.assertTrue(logger.log_scan_observations("A", "DISCOVERY", [row]))
            self.assertTrue(logger.log_scan_observations("B", "DISCOVERY", [{**row,"reason":"Other account"}]))
            self.assertTrue(logger.log_scan_observations("A", "DISCOVERY", [{**row,"selected":True}]))
            records = TradeReplayLogger(path).get_scan_observations("A", "eurjpy")
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0]["snapshot"]["selected"])
            self.assertFalse(records[1]["snapshot"]["selected"])
            self.assertEqual(records[0]["config_fingerprint"], settings.config_fingerprint)
            self.assertEqual(logger.get_scan_observations("A", "EURJPY' OR 1=1 --"), [])
            self.assertEqual(logger.get_scan_observations("C"), [])

    def test_diagnostics_are_bounded_without_pruning_trade_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scan.db")
            logger = TradeReplayLogger(path)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("INSERT INTO trade_replay(time,symbol,action) VALUES ('2000-01-01','TEST','BUY')")
            policy = replace(settings, scan_audit_max_rows=2, scan_audit_retention_days=30)
            with patch("database.replay_logger.settings", policy):
                self.assertTrue(logger.log_scan_observations("A","DISCOVERY",[{"symbol":str(i)} for i in range(5)]))
                self.assertEqual(len(logger.get_scan_observations("A")), 2)
                with closing(sqlite3.connect(path)) as db, db:
                    db.execute("UPDATE scan_observation SET observed_at_utc = '2000-01-01T00:00:00+00:00'")
                logger.log_scan_observations("A","ENTRY",[{"symbol":"NEW"}])
            self.assertEqual(len(logger.get_scan_observations("A")), 1)
            with closing(sqlite3.connect(path)) as db, db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM trade_replay").fetchone()[0], 1)

    def test_engine_deduplicates_and_retries_failed_writes(self):
        engine = TradingEngine.__new__(TradingEngine)
        writer = Mock(return_value=True)
        engine.replay_logger = SimpleNamespace(log_scan_observations=writer)
        row = {"symbol":"TEST","opportunity_bar":"2030-01-01T00:00:00Z","status":"BLOCKED",
               "research_watch":{"candidate":True,"live_eligible":False}}
        policy = replace(settings, scan_audit_enabled=True)
        with patch("core.engine.settings", policy):
            for _ in range(10):
                asyncio.run(engine._record_scan_audit([row], "DISCOVERY", {"login":1}))
            self.assertEqual(writer.call_count, 1)
            self.assertTrue(writer.call_args.args[2][0]["research_watch"]["candidate"])
            writer.return_value = False
            changed = {**row,"status":"READY FOR REVIEW"}
            asyncio.run(engine._record_scan_audit([changed], "DISCOVERY", {"login":1}))
            writer.return_value = True
            asyncio.run(engine._record_scan_audit([changed], "DISCOVERY", {"login":1}))
            self.assertEqual(writer.call_count, 3)
            asyncio.run(engine._record_scan_audit([changed], "DISCOVERY", {"login":2}))
            self.assertEqual(writer.call_count, 4)
            writer.side_effect = RuntimeError("storage unavailable")
            asyncio.run(engine._record_scan_audit([row], "ENTRY", {"login":2}))

    def test_read_endpoint_limits_scope_and_validates_limits(self):
        reader = Mock(return_value=[{"symbol":"EURJPY"}])
        engine = SimpleNamespace(
            _active_account_identity={"login":42},
            _account_scope=Mock(return_value="A"),
            replay_logger=SimpleNamespace(get_scan_observations=reader),
        )
        with patch.object(app.state, "engine", engine, create=True):
            response = asyncio.run(get_scan_audit("EURJPY", 20))
            self.assertEqual(response["observations"], [{"symbol":"EURJPY"}])
            reader.assert_called_once_with("A", "EURJPY", 20)
            self.assertEqual(asyncio.run(get_scan_audit("", 201)).status_code, 422)
            self.assertEqual(asyncio.run(get_scan_audit("X"*101, 1)).status_code, 422)
            reader.side_effect = RuntimeError("temporary database lock")
            response = asyncio.run(get_scan_audit("", 10))
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("temporary database lock", json.loads(response.body)["error"])

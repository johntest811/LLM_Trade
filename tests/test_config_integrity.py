import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app_config.paths import ENV_PATH
from app_config.settings import duplicate_env_keys, settings
from database.replay_logger import TradeReplayLogger


class ConfigIntegrityTests(unittest.TestCase):
    def test_workspace_env_has_unique_keys(self):
        self.assertEqual(duplicate_env_keys(ENV_PATH), {})

    def test_duplicate_detector_reports_key_line_numbers_without_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text(
                "RISK_PERCENT=1\n# comment\nRISK_PERCENT=2\n",
                encoding="utf-8",
            )

            self.assertEqual(duplicate_env_keys(path), {"RISK_PERCENT": [1, 3]})

    def test_fingerprint_is_stable_sensitive_and_sanitized(self):
        self.assertEqual(settings.config_fingerprint, settings.config_fingerprint)
        changed = replace(settings, risk_percent=settings.risk_percent + 0.01)
        self.assertNotEqual(settings.config_fingerprint, changed.config_fingerprint)

        snapshot = settings.sanitized_snapshot()
        for excluded in (
            "mt5_account",
            "mt5_password",
            "mt5_path",
            "openai_api_key",
            "openai_organization",
            "openai_project",
        ):
            self.assertNotIn(excluded, snapshot)

    def test_replay_rows_record_effective_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audit.db"
            replay = TradeReplayLogger(str(db_path))
            row_id = replay.log_decision_trace(
                symbol="USDJPY",
                candle_time="2026-08-10T05:30:00+00:00",
                telemetry={
                    "trace_id": "config-trace",
                    "provider": "local",
                    "model": "test-model",
                    "latency_seconds": 1.0,
                    "success": True,
                },
                decision={"action": "HOLD", "confidence": 0.0},
                validation_status="VALID",
            )
            self.assertIsNotNone(row_id)
            replay_id = replay.log_replay_attempt(
                symbol="USDJPY",
                action="HOLD",
                prompt_text="test",
                llm_json={"action": "HOLD", "confidence": 0.0},
                indicators={},
                market_structure={},
                quality_score=0.0,
                confluence_score=0.0,
                status="DECIDED",
            )
            self.assertIsNotNone(replay_id)

            conn = sqlite3.connect(db_path)
            try:
                decision_value = conn.execute(
                    "SELECT config_fingerprint FROM decision_trace "
                    "WHERE trace_id='config-trace'"
                ).fetchone()[0]
                replay_value = conn.execute(
                    "SELECT config_fingerprint FROM trade_replay WHERE id=?",
                    (replay_id,),
                ).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(decision_value, settings.config_fingerprint)
            self.assertEqual(replay_value, settings.config_fingerprint)


if __name__ == "__main__":
    unittest.main()

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app_config.paths import ENV_PATH
from app_config.settings import duplicate_env_keys, settings
from database.replay_logger import TradeReplayLogger


class ConfigIntegrityTests(unittest.TestCase):
    def test_requested_protection_profile_is_the_default_contract(self):
        self.assertEqual(settings.profit_lock_trigger_usd, 0.35)
        self.assertEqual(settings.profit_lock_trigger_r, 0.50)
        self.assertEqual(settings.profit_lock_floor_usd, 0.08)
        self.assertEqual(settings.profit_lock_mid_trigger_usd, 0.60)
        self.assertEqual(settings.profit_lock_mid_trigger_r, 0.75)
        self.assertEqual(settings.profit_lock_mid_fraction, 0.35)
        self.assertEqual(settings.profit_lock_final_trigger_usd, 1.15)
        self.assertEqual(settings.profit_lock_final_floor_usd, 1.00)
        self.assertEqual(settings.breakeven_trigger_r, 0.75)
        self.assertEqual(settings.trailing_trigger_r, 1.00)
        self.assertEqual(settings.failed_thesis_reversal_min_confidence, 0.60)
        self.assertEqual(settings.same_thesis_reentry_min_bars, 2)
        self.assertFalse(hasattr(settings, "early_profit_lock_enabled"))

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

    def test_strategy_close_reason_survives_generic_broker_reconciliation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audit.db"
            replay = TradeReplayLogger(str(db_path))
            replay_id = replay.log_replay_attempt(
                symbol="NZDUSD",
                action="BUY",
                prompt_text="test",
                llm_json={"action": "BUY", "confidence": 0.85},
                indicators={},
                market_structure={},
                quality_score=75.0,
                confluence_score=75.0,
                status="OPENED",
                ticket=12345,
            )
            self.assertIsNotNone(replay_id)

            self.assertTrue(
                replay.update_replay_outcome(
                    12345,
                    -0.42,
                    close_reason="STAGNATION_EXIT",
                )
            )
            self.assertTrue(
                replay.update_replay_outcome(
                    12345,
                    -0.42,
                    close_reason="EXPERT",
                    mfe_usd=0.40,
                    mae_usd=-0.46,
                )
            )

            conn = sqlite3.connect(db_path)
            try:
                reason, mfe_usd, mae_usd = conn.execute(
                    "SELECT close_reason, mfe_usd, mae_usd "
                    "FROM trade_replay WHERE id = ?",
                    (replay_id,),
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(reason, "STAGNATION_EXIT")
            self.assertAlmostEqual(mfe_usd, 0.40)
            self.assertAlmostEqual(mae_usd, -0.46)

    def test_broker_close_reason_populates_empty_replay_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "audit.db"
            replay = TradeReplayLogger(str(db_path))
            replay.log_replay_attempt(
                symbol="CADJPY",
                action="BUY",
                prompt_text="test",
                llm_json={"action": "BUY", "confidence": 0.85},
                indicators={},
                market_structure={},
                quality_score=80.0,
                confluence_score=80.0,
                status="OPENED",
                ticket=67890,
            )
            self.assertTrue(
                replay.update_replay_outcome(
                    67890,
                    2.38,
                    close_reason="TAKE_PROFIT",
                )
            )
            conn = sqlite3.connect(db_path)
            try:
                reason = conn.execute(
                    "SELECT close_reason FROM trade_replay WHERE ticket = ?",
                    (67890,),
                ).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(reason, "TAKE_PROFIT")


if __name__ == "__main__":
    unittest.main()

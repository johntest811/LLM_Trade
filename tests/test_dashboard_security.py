import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app_config.settings import settings
from ui.dashboard import _is_local_origin, save_config


class DashboardSecurityTests(unittest.TestCase):
    def test_browser_origin_must_be_loopback(self):
        self.assertTrue(_is_local_origin("http://127.0.0.1:8080"))
        self.assertTrue(_is_local_origin("http://localhost:8080"))
        self.assertTrue(_is_local_origin("http://[::1]:8080"))
        self.assertFalse(_is_local_origin("https://example.com"))
        self.assertFalse(_is_local_origin("http://localhost.example.com:8080"))

    def test_config_rejects_newline_injection(self):
        response = asyncio.run(
            save_config({"TRADING_SYMBOLS": "ETHUSD\nDRY_RUN=False"})
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("newlines", json.loads(response.body)["error"])

    def test_config_rejects_invalid_numeric_value(self):
        response = asyncio.run(save_config({"RISK_PERCENT": "not-a-number"}))
        self.assertEqual(response.status_code, 422)
        self.assertIn("numeric", json.loads(response.body)["error"])

    def test_config_rejects_fractional_integer_setting(self):
        response = asyncio.run(save_config({"MAX_OPEN_POSITIONS": "1.5"}))
        self.assertEqual(response.status_code, 422)
        self.assertIn("whole number", json.loads(response.body)["error"])

    def test_config_rejects_lookalike_local_llm_host(self):
        response = asyncio.run(
            save_config({"LOCAL_LLM_URL": "http://localhost.example.com:1234/v1"})
        )
        self.assertEqual(response.status_code, 422)

    def test_config_never_accepts_openai_api_key(self):
        response = asyncio.run(save_config({"OPENAI_API_KEY": "do-not-persist"}))
        self.assertEqual(response.status_code, 400)
        self.assertIn("No permitted", json.loads(response.body)["error"])

    def test_config_rejects_unknown_local_quantization(self):
        response = asyncio.run(
            save_config({"LOCAL_LLM_REQUIRED_QUANTIZATION": "Q5_K"})
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("AUTO, Q6_K, or Q8_0", json.loads(response.body)["error"])

    def test_config_persists_147_minimum_risk_reward(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "MIN_RISK_REWARD_RATIO=1.5\n",
                encoding="utf-8",
            )
            with patch(
                "ui.dashboard._config_env_path", return_value=env_path
            ):
                response = asyncio.run(
                    save_config({"MIN_RISK_REWARD_RATIO": "1.47"})
                )

            self.assertEqual(response["status"], "saved")
            self.assertIn(
                "MIN_RISK_REWARD_RATIO=1.47",
                env_path.read_text(encoding="utf-8"),
            )

    def test_config_accepts_current_micro_account_risk_and_new_entry_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "RISK_PERCENT=10.0\n"
                "ENTRY_ADX_DECLINE_TOLERANCE=0.0\n"
                "PLAN_MAX_COST_TARGET_EXTENSION_R=0.0\n",
                encoding="utf-8",
            )
            with patch(
                "ui.dashboard._config_env_path", return_value=env_path
            ):
                response = asyncio.run(
                    save_config(
                        {
                            "RISK_PERCENT": "10.5",
                            "ENTRY_ADX_DECLINE_TOLERANCE": "0.50",
                            "PLAN_MAX_COST_TARGET_EXTENSION_R": "0.50",
                        }
                    )
                )

            self.assertEqual(response["status"], "saved")
            saved = env_path.read_text(encoding="utf-8")
            self.assertIn("RISK_PERCENT=10.5", saved)
            self.assertIn("ENTRY_ADX_DECLINE_TOLERANCE=0.50", saved)
            self.assertIn("PLAN_MAX_COST_TARGET_EXTENSION_R=0.50", saved)

    def test_config_rejects_early_profit_floor_without_headroom(self):
        response = asyncio.run(
            save_config(
                {
                    "EARLY_PROFIT_LOCK_TRIGGER_USD": "0.12",
                    "EARLY_PROFIT_LOCK_FLOOR_USD": "0.12",
                }
            )
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn(
            "must be lower",
            json.loads(response.body)["error"],
        )

    def test_config_persists_retest_and_default_off_micro_profit_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "RETEST_CONTINUATION_ENABLED=false\n"
                "RETEST_MIN_RESUMPTION_ATR=0.20\n"
                "MICRO_PROFIT_PROTECTION_ENABLED=true\n",
                encoding="utf-8",
            )
            with patch(
                "ui.dashboard._config_env_path", return_value=env_path
            ):
                response = asyncio.run(
                    save_config(
                        {
                            "RETEST_CONTINUATION_ENABLED": "true",
                            "RETEST_MIN_RESUMPTION_ATR": "0.10",
                            "MICRO_PROFIT_PROTECTION_ENABLED": "false",
                        }
                    )
                )

            self.assertEqual(response["status"], "saved")
            saved = env_path.read_text(encoding="utf-8")
            self.assertIn("RETEST_CONTINUATION_ENABLED=true", saved)
            self.assertIn("RETEST_MIN_RESUMPTION_ATR=0.10", saved)
            self.assertIn("MICRO_PROFIT_PROTECTION_ENABLED=false", saved)

    def test_micro_profit_toggle_is_hot_applied_for_refresh_consistency(self):
        original = settings.micro_profit_protection_enabled
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                env_path = Path(temp_dir) / ".env"
                env_path.write_text(
                    "MICRO_PROFIT_PROTECTION_ENABLED=false\n",
                    encoding="utf-8",
                )
                with patch(
                    "ui.dashboard._config_env_path", return_value=env_path
                ):
                    response = asyncio.run(
                        save_config(
                            {"MICRO_PROFIT_PROTECTION_ENABLED": "true"}
                        )
                    )

                self.assertEqual(response["status"], "saved")
                self.assertFalse(response["restart_required"])
                self.assertIn(
                    "MICRO_PROFIT_PROTECTION_ENABLED",
                    response["hot_applied"],
                )
                self.assertTrue(settings.micro_profit_protection_enabled)
                self.assertIn(
                    "MICRO_PROFIT_PROTECTION_ENABLED=true",
                    env_path.read_text(encoding="utf-8"),
                )
        finally:
            object.__setattr__(
                settings, "micro_profit_protection_enabled", original
            )

    def test_config_persists_auto_quantization_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "LOCAL_LLM_REQUIRED_QUANTIZATION=Q8_0\n"
                "LLM_MAX_CONCURRENCY=1\n",
                encoding="utf-8",
            )
            with patch(
                "ui.dashboard._config_env_path", return_value=env_path
            ):
                response = asyncio.run(
                    save_config(
                        {
                            "LOCAL_LLM_REQUIRED_QUANTIZATION": "AUTO",
                            "LLM_MAX_CONCURRENCY": "1",
                        }
                    )
                )

            self.assertEqual(response["status"], "saved")
            saved = env_path.read_text(encoding="utf-8")
            self.assertIn(
                "LOCAL_LLM_REQUIRED_QUANTIZATION=AUTO", saved
            )
            self.assertIn("LLM_MAX_CONCURRENCY=1", saved)

    def test_config_persists_supported_q8_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "LOCAL_LLM_REQUIRED_QUANTIZATION=Q6_K\n"
                "LLM_MAX_CONCURRENCY=2\n",
                encoding="utf-8",
            )
            with patch(
                "ui.dashboard._config_env_path", return_value=env_path
            ):
                response = asyncio.run(
                    save_config(
                        {
                            "LOCAL_LLM_REQUIRED_QUANTIZATION": "Q8_0",
                            "LLM_MAX_CONCURRENCY": "1",
                        }
                    )
                )

            self.assertEqual(response["status"], "saved")
            saved = env_path.read_text(encoding="utf-8")
            self.assertIn(
                "LOCAL_LLM_REQUIRED_QUANTIZATION=Q8_0", saved
            )
            self.assertIn("LLM_MAX_CONCURRENCY=1", saved)


if __name__ == "__main__":
    unittest.main()

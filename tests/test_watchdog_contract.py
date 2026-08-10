from pathlib import Path
import unittest


class WatchdogContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).parents[1] / "run_unattended.ps1"
        ).read_text(encoding="utf-8")

    def test_monitors_engine_broker_and_protection_progress(self):
        self.assertIn("broker_poll_heartbeat_utc", self.script)
        self.assertIn('"LOOP", "PROTECTION"', self.script)
        self.assertIn('"CONNECTION"', self.script)

    def test_external_model_failure_does_not_trigger_restart(self):
        self.assertIn('"LLM"', self.script)
        self.assertIn("decision_provider_inference_ready", self.script)
        self.assertIn("dependency is unavailable (no process restart)", self.script)

    def test_duplicate_and_restart_loop_guards_are_present(self):
        self.assertIn("Multiple matching engine processes were found", self.script)
        self.assertIn("A second engine will not ", self.script)
        self.assertIn('"be started."', self.script)
        self.assertIn("MaxRestartsPerWindow", self.script)
        self.assertIn("Restart circuit opened", self.script)
        self.assertIn("replacementAuthorized", self.script)

    def test_watchdog_does_not_mutate_trading_risk_settings(self):
        self.assertNotIn("/api/risk/", self.script)
        self.assertNotIn("RISK_PERCENT", self.script)
        self.assertNotIn("MAX_DAILY_LOSS", self.script)

    def test_recovered_process_preserves_runtime_diagnostics(self):
        self.assertIn("RedirectStandardOutput", self.script)
        self.assertIn("RedirectStandardError", self.script)
        self.assertIn("runtime_stdout.log", self.script)
        self.assertIn("runtime_stderr.log", self.script)


if __name__ == "__main__":
    unittest.main()

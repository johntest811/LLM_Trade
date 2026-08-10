import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.engine import TradingEngine


class ScanStatusAggregationTests(unittest.TestCase):
    @staticmethod
    def _engine(*symbols: str) -> TradingEngine:
        engine = TradingEngine.__new__(TradingEngine)
        engine._scan_status_lock = threading.RLock()
        engine._active_scan_symbols = tuple(symbols)
        engine._symbol_scan_states = {symbol: "WAITING" for symbol in symbols}
        return engine

    def test_stale_symbols_do_not_overwrite_active_or_completed_scan(self):
        engine = self._engine("USDJPY", "CADJPY", "ETHUSD")
        fake_dashboard = SimpleNamespace(update_automation=MagicMock())

        with patch("core.engine.dashboard_state", fake_dashboard):
            engine._set_symbol_scan_state("USDJPY", "STALE")
            engine._set_symbol_scan_state("CADJPY", "STALE")
            active = engine._set_symbol_scan_state("ETHUSD", "LLM_INFERENCE")
            late_stale = engine._set_symbol_scan_state("USDJPY", "STALE")
            completed = engine._set_symbol_scan_state("ETHUSD", "COMPLETE")

        self.assertEqual(active, "LLM INFERENCE · ETHUSD · 2 STALE")
        self.assertEqual(late_stale, active)
        self.assertEqual(completed, "MONITORING M5 · 1 READY · 2 STALE")
        self.assertEqual(
            engine._symbol_scan_states,
            {"USDJPY": "STALE", "CADJPY": "STALE", "ETHUSD": "COMPLETE"},
        )

    def test_aggregate_only_includes_currently_active_configured_symbols(self):
        engine = self._engine("USDJPY", "CADJPY", "ETHUSD")
        engine._symbol_scan_states.update({"USDJPY": "STALE", "CADJPY": "ERROR"})
        fake_dashboard = SimpleNamespace(update_automation=MagicMock())

        with patch("core.engine.dashboard_state", fake_dashboard):
            engine._set_active_scan_symbols(["ETHUSD"])

        published = fake_dashboard.update_automation.call_args.kwargs["scan_status"]
        self.assertEqual(published, "WAITING FOR NEXT M5 CLOSE")
        self.assertEqual(engine._symbol_scan_states["USDJPY"], "STALE")
        self.assertEqual(engine._symbol_scan_states["CADJPY"], "ERROR")


if __name__ == "__main__":
    unittest.main()

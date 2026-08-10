import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ui.dashboard import _live_payload, _send_bounded


class _Socket:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.messages = []

    async def send_text(self, payload):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.messages.append(payload)


class DashboardStreamTests(unittest.TestCase):
    def test_live_payload_sends_one_current_sample_per_market(self):
        prices = {
            "EURUSD": SimpleNamespace(
                bid=1.1,
                ask=1.1001,
                spread_pips=1.0,
                spread_value=1.0,
                spread_unit="pips",
                asset_class="FX/CFD",
                trend="BULLISH",
                adx=25.0,
                updated_at="2026-07-29T05:00:00+00:00",
            ),
            "USDJPY": SimpleNamespace(
                bid=160.0,
                ask=160.01,
                spread_pips=1.0,
                spread_value=1.0,
                spread_unit="pips",
                asset_class="FX/CFD",
                trend="BEARISH",
                adx=30.0,
                updated_at="2026-07-29T05:00:01+00:00",
            ),
        }
        fake_state = SimpleNamespace(
            _lock=threading.RLock(),
            prices=prices,
            tick_stream=[{"symbol": "EURUSD"}] * 50,
            system=SimpleNamespace(
                cpu_pct=10.0,
                ram_pct=20.0,
                ram_used_gb=6.0,
                ram_total_gb=32.0,
                gpu_util_pct=30.0,
                gpu_mem_pct=40.0,
                gpu_mem_used_mb=4000.0,
                gpu_mem_total_mb=12000.0,
                gpu_temp_c=55.0,
            ),
        )

        with patch("ui.dashboard.dashboard_state", fake_state):
            payload = _live_payload()

        self.assertEqual(len(payload["tick_stream"]), 2)
        self.assertEqual(set(payload["prices"]), {"EURUSD", "USDJPY"})
        self.assertIn("updated_at", payload["prices"]["EURUSD"])

    def test_stalled_socket_times_out_without_blocking_other_clients(self):
        fast = _Socket()
        slow = _Socket(delay=0.05)

        async def exercise():
            return await asyncio.gather(
                _send_bounded(fast, "payload", timeout=0.01),
                _send_bounded(slow, "payload", timeout=0.01),
            )

        results = asyncio.run(exercise())

        self.assertTrue(results[0][1])
        self.assertFalse(results[1][1])
        self.assertEqual(fast.messages, ["payload"])


if __name__ == "__main__":
    unittest.main()

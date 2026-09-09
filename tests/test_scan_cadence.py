import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd

from core.engine import TradingEngine
from app_config.settings import settings as runtime_settings

# Model-lane cadence cases deliberately exercise the model, not the separately
# tested deterministic fast path (test_engine_resilience.py).
settings = replace(runtime_settings, deterministic_entry_fast_path_enabled=False)


class _Reader:
    def __init__(self, stale=False, live_tick=False):
        self.stale = stale
        self.live_tick = live_tick
        self.calls = {"M5": 0, "M15": 0, "H1": 0, "H4": 0}

    async def get_ohlcv(self, symbol, timeframe, count):
        self.calls[timeframe] += 1
        frame = pd.DataFrame({"time": [f"{timeframe}-closed-bar"]})
        if timeframe == "M5":
            frame.attrs.update(
                is_stale=self.stale,
                age_after_close_seconds=900.0 if self.stale else 0.0,
            )
        return frame

    async def get_live_tick(self, symbol, assume_connected=False):
        del assume_connected
        return {"bid": 1.1000, "ask": 1.1001} if self.live_tick else None


class _Analyzer:
    def analyze(self, symbol, timeframe, candles):
        return {
            "indicators": {"adx_14": 30.0, "adx_delta": 1.0, "current_price": 100.0,
                           "atr_14": 1.0, "ema_9": 99.8, "ema_21": 100.0,
                           "rsi_14": 45.0, "macd": {"diff": -.1}},
            "market_structure": {
                "trend": "BEARISH",
                "trend_state": "EARLY_BULLISH_REVERSAL",
                "structure_events": [
                    {
                        "type": "BOS",
                        "direction": "BEARISH",
                        "level": 100.5,
                        "time": "2026-08-12 05:10:00+00:00",
                    }
                ],
            },
            "timeframe": timeframe,
        }


class _WeakAnalyzer(_Analyzer):
    def analyze(self, symbol, timeframe, candles):
        result = super().analyze(symbol, timeframe, candles)
        result["indicators"]["adx_14"] = 12.0
        return result


class _NoTriggerAnalyzer(_Analyzer):
    def analyze(self, symbol, timeframe, candles):
        result = super().analyze(symbol, timeframe, candles)
        result["market_structure"]["structure_events"] = []
        return result


class _Database:
    async def get_closed_positions(self, **kwargs):
        return []


class _LLM:
    def __init__(self):
        self.calls = 0
        self.last_error = ""

    async def get_trading_decision(self, system_prompt, user_prompt):
        self.calls += 1
        return {"action": "HOLD", "confidence": 0.5, "reasoning": "No setup"}


class _InvalidLLM(_LLM):
    async def get_trading_decision(self, system_prompt, user_prompt):
        self.calls += 1
        return {"action": "NOT_AN_ORDER", "confidence": 0.5}


class ScanCadenceTests(unittest.TestCase):
    def setUp(self):
        context = patch("core.engine.settings", settings)
        context.start()
        self.addCleanup(context.stop)

    def test_model_latency_excludes_serial_queue_wait(self):
        telemetry = {"trace_id": "trace-1", "latency_seconds": 3.75}
        self.assertEqual(
            TradingEngine._reported_inference_latency(telemetry, 21.5), 3.75
        )

    def test_disarmed_signal_is_released_for_fresh_post_arm_rescan(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.last_bar_times = {"AUDCAD": "bar-1", "USDCAD": "bar-1"}
        engine.last_scan_times = {
            "AUDCAD": pd.Timestamp("2026-07-22 08:40:00").to_pydatetime(),
            "USDCAD": pd.Timestamp("2026-07-22 08:40:00").to_pydatetime(),
        }
        engine._signals_waiting_for_rearm = set()

        engine._defer_disarmed_signal("audcad")
        released = engine._release_deferred_signals_for_rescan()

        self.assertEqual(released, ("AUDCAD",))
        self.assertNotIn("AUDCAD", engine.last_bar_times)
        self.assertNotIn("AUDCAD", engine.last_scan_times)
        self.assertEqual(engine.last_bar_times["USDCAD"], "bar-1")
        self.assertFalse(engine._signals_waiting_for_rearm)

    def test_managed_open_position_is_prioritized_over_ranked_candidates(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine._symbols_for_current_market = lambda: ["AUDUSD", "USDJPY"]
        positions = [
            {"symbol": "USDCAD", "bot_owned": True},
            {"symbol": "EURJPY", "bot_owned": False},
        ]

        with patch(
            "core.engine.settings",
            replace(settings, manage_external_positions=False),
        ):
            symbols = engine._symbols_with_position_priority(positions)

        self.assertEqual(symbols, ["USDCAD", "AUDUSD", "USDJPY"])

    def test_live_quote_uses_completed_analysis_state_after_analysis_is_created(self):
        llm = _LLM()
        engine = TradingEngine(object(), _Reader(live_tick=True), object(), llm, _Database(), object())
        engine.analyzer = _Analyzer()
        engine.entries_armed = False
        engine.log = lambda *args, **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.mt5.symbol_info", return_value=SimpleNamespace(point=0.00001)),
            patch("core.engine.PromptGenerator.generate", return_value=("system", "user")),
            patch(
                "core.engine.DeterministicTradePlanner.assess_capital_fit",
                return_value={"capital_fit": True, "status": "CAPITAL FIT"},
            ),
        ):
            asyncio.run(engine._evaluate_symbol("USDCAD", {"balance": 15.17}, []))

        self.assertEqual(llm.calls, 1)

    def test_disarmed_entries_do_not_disable_llm_monitoring(self):
        llm = _LLM()
        engine = TradingEngine(object(), _Reader(), object(), llm, _Database(), object())
        engine.analyzer = _Analyzer()
        engine.entries_armed = False
        engine.log = lambda *args, **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.PromptGenerator.generate", return_value=("system", "user")),
        ):
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))

        self.assertEqual(llm.calls, 1)
        self.assertEqual(engine.last_bar_times["USDJPY"], "M5-closed-bar")

    def test_completed_bar_is_only_sent_once(self):
        llm = _LLM()
        reader = _Reader()
        engine = TradingEngine(object(), reader, object(), llm, _Database(), object())
        engine.analyzer = _Analyzer()
        engine.entries_armed = False
        engine.log = lambda *args, **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.PromptGenerator.generate", return_value=("system", "user")),
        ):
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))
            engine.last_scan_times.clear()
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))

        self.assertEqual(llm.calls, 1)
        self.assertEqual(reader.calls, {"M5": 2, "M15": 1, "H1": 1, "H4": 1})

    def test_processed_bar_outcome_is_not_overwritten_as_missed_later(self):
        class _OldCompletedBarReader(_Reader):
            async def get_ohlcv(self, symbol, timeframe, count):
                frame = await super().get_ohlcv(symbol, timeframe, count)
                if timeframe == "M5":
                    frame.attrs["age_after_close_seconds"] = 70.0
                return frame

        llm = _LLM()
        reader = _OldCompletedBarReader()
        engine = TradingEngine(
            object(), reader, object(), llm, _Database(), object()
        )
        engine.last_bar_times["USDJPY"] = "M5-closed-bar"
        engine.log = lambda *args, **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch(
                "core.engine.dashboard_state.update_symbol_decision"
            ) as update_decision,
        ):
            asyncio.run(
                engine._evaluate_symbol("USDJPY", {"balance": 15.0}, [])
            )

        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            reader.calls, {"M5": 1, "M15": 0, "H1": 0, "H4": 0}
        )
        self.assertFalse(
            any(
                call.kwargs.get("stage") == "MISSED CANDLE"
                for call in update_decision.call_args_list
            )
        )

    def test_invalid_model_output_is_not_retried_on_the_same_candle(self):
        llm = _InvalidLLM()
        reader = _Reader()
        engine = TradingEngine(
            object(), reader, object(), llm, _Database(), object()
        )
        engine.analyzer = _Analyzer()
        engine.entries_armed = False
        engine.log = lambda *args, **kwargs: None
        engine.replay_logger.log_replay_attempt = lambda **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.PromptGenerator.generate", return_value=("system", "user")),
            patch(
                "core.engine.DeterministicTradePlanner.assess_capital_fit",
                return_value={"capital_fit": True, "status": "CAPITAL FIT"},
            ),
        ):
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))
            engine.last_scan_times.clear()
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))

        self.assertEqual(llm.calls, 1)
        self.assertEqual(engine.last_bar_times["USDJPY"], "M5-closed-bar")

    def test_stale_m5_bar_never_reaches_llm(self):
        llm = _LLM()
        reader = _Reader(stale=True)
        engine = TradingEngine(object(), reader, object(), llm, _Database(), object())
        engine.analyzer = _Analyzer()
        engine.log = lambda *args, **kwargs: None

        with patch("core.engine.is_weekend", return_value=False):
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))

        self.assertEqual(llm.calls, 0)
        self.assertEqual(reader.calls, {"M5": 1, "M15": 0, "H1": 0, "H4": 0})

    def test_market_watch_polls_all_candidates_not_only_active_symbols(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.is_running = True
        engine.conn = SimpleNamespace(is_connected=AsyncMock(return_value=True))
        engine.reader = SimpleNamespace(
            get_live_tick=AsyncMock(
                side_effect=lambda symbol, assume_connected=False: {
                    "time_msc": {
                        "ETHUSD": 1,
                        "LTCUSD": 2,
                        "XRPUSD": 3,
                    }[symbol],
                    "bid": 1.0,
                    "ask": 1.1,
                }
            )
        )
        engine._market_candidates_for_current_market = lambda: [
            "ETHUSD", "LTCUSD", "XRPUSD"
        ]
        engine._symbols_for_current_market = lambda: ["ETHUSD"]
        engine._last_tick_error_log_monotonic = 0.0
        engine.log = lambda *args, **kwargs: None

        async def stop_after_iteration(_seconds):
            engine.is_running = False

        with (
            patch("core.engine.asyncio.sleep", side_effect=stop_after_iteration),
            patch(
                "core.engine.mt5.symbol_info",
                return_value=SimpleNamespace(point=0.01),
            ),
            patch("core.engine.dashboard_state.update_prices"),
            patch("core.engine.dashboard_state.add_tick"),
        ):
            asyncio.run(engine._tick_loop())

        self.assertEqual(engine.reader.get_live_tick.await_count, 3)
        self.assertEqual(
            [call.args[0] for call in engine.reader.get_live_tick.await_args_list],
            ["ETHUSD", "LTCUSD", "XRPUSD"],
        )
        self.assertTrue(
            all(
                call.kwargs.get("assume_connected") is True
                for call in engine.reader.get_live_tick.await_args_list
            )
        )

    def test_deterministic_entry_prefilter_skips_impossible_model_request(self):
        llm = _LLM()
        reader = _Reader()
        engine = TradingEngine(
            object(), reader, object(), llm, _Database(), object()
        )
        engine.analyzer = _WeakAnalyzer()
        engine.log = lambda *args, **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch(
                "core.engine.DeterministicTradePlanner.assess_capital_fit",
                return_value={"capital_fit": True, "status": "CAPITAL FIT"},
            ),
        ):
            asyncio.run(
                engine._evaluate_symbol("USDJPY", {"balance": 15.0}, [])
            )

        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            engine.last_bar_times["USDJPY"], "M5-closed-bar"
        )
        self.assertEqual(
            reader.calls, {"M5": 1, "M15": 1, "H1": 1, "H4": 1}
        )

    def test_no_actionable_m5_evidence_does_not_consume_model_capacity(self):
        llm = _LLM()
        reader = _Reader()
        engine = TradingEngine(
            object(), reader, object(), llm, _Database(), object()
        )
        engine.analyzer = _NoTriggerAnalyzer()
        engine.log = lambda *args, **kwargs: None

        with (
            patch("core.engine.is_weekend", return_value=False),
            patch(
                "core.engine.DeterministicTradePlanner.assess_capital_fit",
                return_value={"capital_fit": True, "status": "CAPITAL FIT"},
            ),
            patch(
                "core.engine.dashboard_state.update_symbol_decision"
            ) as update_decision,
        ):
            asyncio.run(
                engine._evaluate_symbol("USDJPY", {"balance": 15.0}, [])
            )

        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            engine.last_bar_times["USDJPY"], "M5-closed-bar"
        )
        self.assertTrue(
            any(
                call.kwargs.get("stage") == "NO ENTRY EVIDENCE"
                for call in update_decision.call_args_list
            )
        )

    def test_exhausted_bos_uses_neither_model_calls_nor_admission_slots(self):
        class ExhaustedAnalyzer(_Analyzer):
            def analyze(self, symbol, timeframe, candles):
                result = super().analyze(symbol, timeframe, candles)
                result["timestamp"] = "2026-08-12 05:10:00+00:00"
                result["market_structure"].update(
                    trend_state="CONFIRMED_BEARISH", trend_state_direction="BEARISH",
                    breakout_status="BEARISH BREAKOUT" if timeframe == "M5" else "NONE",
                )
                if timeframe != "M5":
                    result["market_structure"]["structure_events"] = []
                result["indicators"].update(
                    rsi_14=25.0, stochastic={"k": 1.0, "d": 1.0},
                    current_price=100.0, atr_14=1.0,
                    bollinger_bands={"lower": 99.0, "upper": 101.0},
                )
                return result

        llm = _LLM()
        engine = TradingEngine(object(), _Reader(), object(), llm, _Database(), object())
        engine.analyzer = ExhaustedAnalyzer()
        engine.log = lambda *args, **kwargs: None
        with (
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.DeterministicTradePlanner.assess_capital_fit", return_value={"capital_fit": True}),
            patch("core.engine.dashboard_state.update_symbol_decision") as update_decision,
        ):
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))
        self.assertEqual(llm.calls, 0)
        self.assertFalse(engine._entry_model_admissions)
        self.assertTrue(any(call.kwargs.get("stage") == "PREFILTERED" for call in update_decision.call_args_list))

    def test_older_ranking_cannot_veto_a_fresh_setup_with_unused_capacity(self):
        llm = _LLM()
        engine = TradingEngine(object(), _Reader(), object(), llm, _Database(), object())
        engine.analyzer = _Analyzer()
        engine.entries_armed = False
        engine.log = lambda *args, **kwargs: None
        engine._market_rankings["USDJPY"] = {"model_selection_rank": 20, "model_eligible": False}
        with (
            patch("core.engine.is_weekend", return_value=False),
            patch("core.engine.DeterministicTradePlanner.assess_capital_fit", return_value={"capital_fit": True}),
            patch("core.engine.PromptGenerator.generate", return_value=("system", "user")),
        ):
            asyncio.run(engine._evaluate_symbol("USDJPY", {"balance": 15.0}, []))
        self.assertEqual(llm.calls, 1)

    def test_entry_prefilter_ignores_small_adx_measurement_noise(self):
        reason = TradingEngine._entry_prefilter_reason(
            {"indicators": {"adx_14": 25.0, "adx_delta": -0.25}}
        )

        self.assertEqual(reason, "")

    def test_entry_prefilter_blocks_material_adx_decline(self):
        reason = TradingEngine._entry_prefilter_reason(
            {"indicators": {"adx_14": 25.0, "adx_delta": -0.75}}
        )

        self.assertIn("ADX Prefilter", reason)

    def test_entry_prefilter_allows_bounded_aligned_structure_decline(self):
        configured = replace(
            settings,
            entry_min_adx=15.0,
            breakout_min_adx=25.0,
            confirmation_min_adx=20.0,
            entry_adx_decline_tolerance=0.5,
            entry_aligned_adx_decline_tolerance=1.5,
        )
        for direction in ("BULLISH", "BEARISH"):
            with self.subTest(direction=direction):
                analyses = []
                for timeframe in ("M5", "M15", "H1"):
                    analyses.append({
                        "indicators": {
                            "adx_14": 30.0,
                            "adx_delta": -1.0 if timeframe == "M5" else 0.5,
                        },
                        "market_structure": {
                            "trend_state_direction": direction,
                            "structure_events": ([{
                                "type": "BOS",
                                "direction": direction,
                            }] if timeframe == "M5" else []),
                        },
                    })

                with patch("core.engine.settings", configured):
                    reason = TradingEngine._entry_prefilter_reason(*analyses)

                self.assertEqual(reason, "")

    def test_entry_prefilter_does_not_bridge_opposing_confirmation(self):
        m5 = {
            "indicators": {"adx_14": 30.0, "adx_delta": -1.0},
            "market_structure": {
                "trend_state_direction": "BULLISH",
                "structure_events": [{
                    "type": "BOS",
                    "direction": "BULLISH",
                }],
            },
        }
        m15 = {
            "indicators": {"adx_14": 30.0},
            "market_structure": {"trend_state_direction": "BEARISH"},
        }
        h1 = {
            "indicators": {"adx_14": 30.0},
            "market_structure": {"trend_state_direction": "BULLISH"},
        }
        configured = replace(
            settings,
            entry_adx_decline_tolerance=0.5,
            entry_aligned_adx_decline_tolerance=1.5,
            breakout_min_adx=25.0,
            confirmation_min_adx=20.0,
        )

        with patch("core.engine.settings", configured):
            reason = TradingEngine._entry_prefilter_reason(m5, m15, h1)

        self.assertIn("ADX Prefilter", reason)

    def test_engine_progress_health_detects_stalled_broker_poll(self):
        engine = TradingEngine.__new__(TradingEngine)
        engine.is_running = True
        engine._main_loop_heartbeat_monotonic = 100.0
        engine._broker_poll_heartbeat_monotonic = 70.0

        with (
            patch(
                "core.engine.settings",
                replace(settings, engine_heartbeat_stale_seconds=20.0),
            ),
            patch("core.engine.time.monotonic", return_value=101.0),
        ):
            healthy, reason = engine._engine_progress_health()

        self.assertFalse(healthy)
        self.assertIn("stale", reason)

    def test_entry_decision_deadline_rejects_old_analysis(self):
        with (
            patch(
                "core.engine.settings",
                replace(settings, max_entry_decision_age_seconds=45.0),
            ),
            patch("core.engine.time.monotonic", return_value=100.0),
        ):
            fresh, reason = TradingEngine._entry_decision_fresh(40.0)

        self.assertFalse(fresh)
        self.assertIn("60.0s", reason)

    def test_entry_inference_budget_uses_tighter_bar_deadline(self):
        with (
            patch(
                "core.engine.settings",
                replace(
                    settings,
                    max_entry_decision_age_seconds=45.0,
                    max_entry_bar_age_seconds=45.0,
                ),
            ),
            patch("core.engine.time.monotonic", return_value=100.0),
        ):
            remaining = TradingEngine._entry_inference_budget_seconds(
                80.0,
                10.0,
            )

        self.assertEqual(remaining, 15.0)

    def test_late_scan_keeps_q8_inference_window_inside_same_m5_candle(self):
        with (
            patch(
                "core.engine.settings",
                replace(
                    settings,
                    max_entry_decision_age_seconds=75.0,
                    max_entry_bar_age_seconds=120.0,
                ),
            ),
            patch("core.engine.time.monotonic", return_value=100.0),
        ):
            remaining = TradingEngine._entry_inference_budget_seconds(
                100.0,
                72.0,
            )

        self.assertEqual(remaining, 48.0)

    def test_entry_waiting_on_model_queue_expires_without_calling_model(self):
        async def scenario():
            llm = _LLM()
            engine = TradingEngine(
                object(),
                _Reader(),
                object(),
                llm,
                _Database(),
                object(),
            )
            engine.analyzer = _Analyzer()
            engine.entries_armed = False
            engine.log = lambda *args, **kwargs: None
            engine._decision_semaphore = asyncio.Semaphore(1)
            await engine._decision_semaphore.acquire()
            try:
                await engine._evaluate_symbol(
                    "USDJPY", {"balance": 15.0}, []
                )
            finally:
                engine._decision_semaphore.release()
            return engine, llm

        deadline_settings = replace(
            settings,
            max_entry_decision_age_seconds=0.05,
            max_entry_bar_age_seconds=45.0,
        )
        with (
            patch("core.engine.settings", deadline_settings),
            patch("core.engine.is_weekend", return_value=False),
            patch(
                "core.engine.PromptGenerator.generate",
                return_value=("system", "user"),
            ),
            patch(
                "core.engine.DeterministicTradePlanner.assess_capital_fit",
                return_value={
                    "capital_fit": True,
                    "status": "CAPITAL FIT",
                },
            ),
        ):
            engine, llm = asyncio.run(scenario())

        self.assertEqual(llm.calls, 0)
        self.assertEqual(
            engine.last_bar_times["USDJPY"], "M5-closed-bar"
        )


if __name__ == "__main__":
    unittest.main()

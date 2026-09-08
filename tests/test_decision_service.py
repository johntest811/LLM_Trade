import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from database.replay_logger import TradeReplayLogger
from llm.client import (
    DecisionResponse,
    DecisionTelemetry,
    DeterministicDecisionProvider,
    LLMClient,
    LocalDecisionProvider,
    OpenAIResponsesProvider,
)


def _openai_settings(api_key: str = "test-key") -> SimpleNamespace:
    return SimpleNamespace(
        openai_model="gpt-5.6-terra",
        openai_api_key=api_key,
        openai_base_url="https://api.openai.com/v1",
        openai_reasoning_effort="low",
        openai_timeout=20.0,
        openai_max_retries=1,
        openai_organization="",
        openai_project="",
    )


def _analysis(
    direction: str,
    *,
    adx: float = 30.0,
    state_direction: str | None = None,
    state_name: str | None = None,
    rsi: float | None = None,
    events: list[dict] | None = None,
    breakout: str = "None",
) -> dict:
    bullish = direction == "BULLISH"
    return {
        "indicators": {
            "current_price": 1.20,
            "ema_9": 1.21 if bullish else 1.19,
            "ema_21": 1.20,
            "rsi_14": rsi if rsi is not None else (58.0 if bullish else 42.0),
            "adx_14": adx,
            "macd": {"diff": 0.001 if bullish else -0.001},
            "stochastic": {"k": 55.0 if bullish else 45.0},
            "bollinger_bands": {"upper": 1.30, "lower": 1.10},
            "candle_range_atr": 1.0,
            "opening_gap_atr": 0.0,
        },
        "market_structure": {
            "trend": direction,
            "trend_state_direction": state_direction or direction,
            "trend_state": state_name or f"CONFIRMED_{direction}",
            "structure_events": events or [],
            "breakout_status": breakout,
        },
    }


def _context(direction: str = "BULLISH", *, has_open_position: bool = False) -> dict:
    return {
        "symbol": "USDCAD",
        "completed_bar": "2026-07-21 10:00:00",
        "has_open_position": has_open_position,
        "open_positions": [],
        "analyses": {
            timeframe: _analysis(direction)
            for timeframe in ("M5", "M15", "H1", "H4")
        },
    }


class DeterministicDecisionProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_aligned_bullish_context_returns_buy_without_network(self):
        provider = DeterministicDecisionProvider()
        with (
            patch("llm.client.requests.post") as post,
            patch("llm.client.requests.get") as get,
        ):
            response = await provider.request("rules", "USDCAD", context=_context())
            health = await provider.health_check()

        self.assertEqual(response.decision["action"], "BUY")
        self.assertGreaterEqual(response.decision["confidence"], 0.70)
        self.assertTrue(response.telemetry.success)
        self.assertEqual(response.telemetry.provider, "deterministic")
        self.assertEqual(response.telemetry.model, "rules-v2-adaptive")
        self.assertTrue(health["available"])
        post.assert_not_called()
        get.assert_not_called()

    async def test_aligned_bearish_context_returns_sell(self):
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=_context("BEARISH")
        )
        self.assertEqual(response.decision["action"], "SELL")
        self.assertGreaterEqual(response.decision["confidence"], 0.70)

    async def test_conflicting_m15_state_holds(self):
        context = _context()
        context["analyses"]["M15"] = _analysis(
            "BULLISH", state_direction="BEARISH"
        )
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=context
        )
        self.assertEqual(response.decision["action"], "HOLD")
        self.assertIn("No continuation", response.decision["reasoning"])

    async def test_weak_adx_holds(self):
        context = _context()
        context["analyses"]["M5"] = _analysis("BULLISH", adx=12.0)
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=context
        )
        self.assertEqual(response.decision["action"], "HOLD")
        self.assertIn("No continuation", response.decision["reasoning"])

    async def test_completed_pullback_resumption_returns_buy(self):
        context = _context()
        context["previous_trend_states"] = {
            "M5": "PULLBACK_IN_BULLISH_TREND"
        }
        context["analyses"]["M5"]["market_structure"][
            "retest_continuation"
        ] = {
            "direction": "BULLISH",
            "time": context["completed_bar"],
            "previous_state": "PULLBACK_IN_BULLISH_TREND",
            "current_state": "CONFIRMED_BULLISH",
        }
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=context
        )

        self.assertEqual(response.decision["action"], "BUY")
        self.assertEqual(
            response.decision["_strategy"]["mode"], "PULLBACK_RESUMPTION"
        )

    async def test_confirmed_reversal_requires_choch_and_secondary_structure(self):
        context = _context("BEARISH")
        bullish_events = [
            {"type": "CHOCH", "direction": "BULLISH"},
            {"type": "BOS", "direction": "BULLISH"},
        ]
        context["analyses"]["M5"] = _analysis(
            "BEARISH",
            state_direction="BULLISH",
            state_name="EARLY_BULLISH_REVERSAL",
            adx=30.0,
            rsi=58.0,
            events=bullish_events,
        )
        context["analyses"]["M15"] = _analysis(
            "BEARISH",
            state_direction="BULLISH",
            state_name="EARLY_BULLISH_REVERSAL",
            adx=25.0,
            rsi=58.0,
            events=[{"type": "CHOCH", "direction": "BULLISH"}],
        )
        context["analyses"]["M5"]["indicators"].update(
            ema_9=1.21,
            ema_21=1.20,
            rsi_14=58.0,
            macd={"diff": 0.001},
        )
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=context
        )

        self.assertEqual(response.decision["action"], "BUY")
        self.assertEqual(response.decision["_strategy"]["mode"], "CONFIRMED_REVERSAL")

    async def test_market_shock_holds_even_when_trends_align(self):
        context = _context()
        context["analyses"]["M5"]["indicators"]["candle_range_atr"] = 3.0
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=context
        )

        self.assertEqual(response.decision["action"], "HOLD")
        self.assertIn("MARKET_SHOCK", response.decision["reasoning"])

    async def test_open_position_holds_for_existing_protection_engine(self):
        response = await DeterministicDecisionProvider().request(
            "rules", "USDCAD", context=_context(has_open_position=True)
        )
        self.assertEqual(response.decision["action"], "HOLD")
        self.assertIn("already open", response.decision["reasoning"])

    async def test_missing_context_fails_closed(self):
        response = await DeterministicDecisionProvider().request("rules", "USDCAD")
        self.assertIsNone(response.decision)
        self.assertFalse(response.telemetry.success)
        self.assertIn("context is missing", response.telemetry.error)


class OpenAIDecisionProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_key_fails_closed_without_network(self):
        with patch("llm.client.settings", _openai_settings(api_key="")):
            provider = OpenAIResponsesProvider()
            with patch("llm.client.requests.post") as post:
                response = await provider.request("system", "market")

        self.assertIsNone(response.decision)
        self.assertFalse(response.telemetry.success)
        self.assertIn("OPENAI_API_KEY", response.telemetry.error)
        post.assert_not_called()

    async def test_responses_request_is_structured_and_decision_only(self):
        api_response = Mock()
        api_response.headers = {"x-request-id": "req_test"}
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "id": "resp_test",
            "model": "gpt-5.6-terra-2026-07-01",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps(
                                {
                                    "action": "HOLD",
                                    "confidence": 0.61,
                                    "ticket_to_close": None,
                                    "reasoning": "Signals conflict.",
                                    "trade_management": "Wait for confirmation.",
                                }
                            ),
                        }
                    ],
                }
            ],
        }

        with patch("llm.client.settings", _openai_settings()):
            provider = OpenAIResponsesProvider()
            with patch("llm.client.requests.post", return_value=api_response) as post:
                response = await provider.request("system", "market")

        self.assertEqual(response.decision["action"], "HOLD")
        self.assertTrue(response.telemetry.success)
        self.assertEqual(response.telemetry.request_id, "req_test")
        _, kwargs = post.call_args
        self.assertEqual(kwargs["json"]["reasoning"], {"effort": "low"})
        self.assertFalse(kwargs["json"]["store"])
        self.assertTrue(kwargs["json"]["text"]["format"]["strict"])
        self.assertNotIn("tools", kwargs["json"])


class LocalDecisionProviderHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_pins_seed_and_strict_schema(self):
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-4b",
            local_llm_temperature=0.0,
            local_llm_top_p=0.8,
            local_llm_seed=42,
            local_llm_max_retries=1,
            local_llm_max_tokens=220,
            local_llm_structured_output=True,
        )
        response = {
            "id": "local-test",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "HOLD",
                                "confidence": 0.0,
                                "ticket_to_close": None,
                                "reasoning": "Mixed evidence.",
                                "trade_management": "Wait for confirmation.",
                            }
                        )
                    }
                }
            ],
        }

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch.object(provider, "_post", return_value=response) as post:
                result = await provider.request("system", "user")

        self.assertTrue(result.telemetry.success)
        payload = post.call_args.args[0]
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 220)
        self.assertEqual(payload["model"], "qwen/qwen3.5-4b")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    async def test_qwen35_4b_catalog_id_resolves_loaded_short_alias(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "qwen3.5-4b",
                    "quantization": {"name": "Q8_0"},
                    "loaded_instances": [
                        {
                            "id": "qwen3.5-4b",
                            "config": {
                                "context_length": 10240,
                                "parallel": 4,
                                "flash_attention": True,
                            },
                        }
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-4b",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=4096,
            local_llm_required_quantization="AUTO",
            llm_max_concurrency=1,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["available"])
        self.assertEqual(result["configured_model"], "qwen/qwen3.5-4b")
        self.assertEqual(result["resolved_model"], "qwen3.5-4b")
        self.assertEqual(provider.model, "qwen3.5-4b")
        self.assertTrue(result["context_matches"])
        self.assertTrue(result["parallel_matches"])
        self.assertTrue(result["quantization_matches"])

    async def test_qwen35_4b_requested_quantized_ids_are_ready(self):
        for model_id, quantization in (
            ("qwen3.5-4b@q4_k_s", "Q4_K_S"),
            ("qwen3.5-4b@iq4_xs", "IQ4_XS"),
        ):
            with self.subTest(model_id=model_id):
                api_response = Mock()
                api_response.raise_for_status.return_value = None
                api_response.json.return_value = {
                    "models": [
                        {
                            "key": model_id,
                            "quantization": {"name": quantization},
                            "loaded_instances": [
                                {
                                    "id": model_id,
                                    "config": {
                                        "context_length": 8192,
                                        "parallel": 1,
                                    },
                                }
                            ],
                        }
                    ]
                }
                local_settings = SimpleNamespace(
                    local_llm_model=model_id,
                    local_llm_url=(
                        "http://127.0.0.1:1234/v1/chat/completions"
                    ),
                    local_llm_context_size=4096,
                    local_llm_required_quantization="AUTO",
                    llm_max_concurrency=1,
                )

                with patch("llm.client.settings", local_settings):
                    provider = LocalDecisionProvider()
                    with patch(
                        "llm.client.requests.get", return_value=api_response
                    ):
                        result = await provider.health_check()

                self.assertTrue(result["available"])
                self.assertEqual(result["resolved_model"], model_id)
                self.assertTrue(result["quantization_matches"])
                self.assertIsNone(result["supported_quantizations"])

    async def test_qwen35_quantized_catalog_alias_resolves_publisher_prefix(self):
        model_id = "qwen3.5-4b@iq4_xs"
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": model_id,
                    "quantization": {"name": "IQ4_XS"},
                    "loaded_instances": [
                        {
                            "id": model_id,
                            "config": {
                                "context_length": 4096,
                                "parallel": 1,
                            },
                        }
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-4b@iq4_xs",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=4096,
            local_llm_required_quantization="AUTO",
            llm_max_concurrency=1,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["available"])
        self.assertEqual(result["resolved_model"], model_id)

    async def test_bonsai_request_disables_thinking_and_uses_strict_schema(self):
        local_settings = SimpleNamespace(
            local_llm_model="prism-ml/bonsai-27b",
            local_llm_temperature=0.0,
            local_llm_top_p=0.8,
            local_llm_seed=42,
            local_llm_max_retries=1,
            local_llm_max_tokens=220,
            local_llm_structured_output=True,
        )
        response = {
            "id": "bonsai-test",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "action": "HOLD",
                                "confidence": 0.0,
                                "ticket_to_close": None,
                                "evidence_ids": [],
                                "reasoning": "No entry evidence.",
                                "trade_management": "Wait.",
                            }
                        )
                    }
                }
            ],
        }

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch.object(provider, "_post", return_value=response) as post:
                result = await provider.request("system", "user")

        self.assertTrue(result.telemetry.success)
        payload = post.call_args.args[0]
        self.assertEqual(payload["model"], "prism-ml/bonsai-27b")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["max_tokens"], 220)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    async def test_bonsai_auto_profile_accepts_its_native_q1(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "prism-ml/bonsai-27b",
                    "quantization": {"name": "Q1_0"},
                    "loaded_instances": [
                        {
                            "id": "prism-ml/bonsai-27b",
                            "config": {
                                "context_length": 10240,
                                "parallel": 4,
                                "flash_attention": True,
                            },
                        }
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="prism-ml/bonsai-27b",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=4096,
            local_llm_required_quantization="AUTO",
            llm_max_concurrency=1,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["available"])
        self.assertEqual(result["resolved_model"], "prism-ml/bonsai-27b")
        self.assertEqual(result["quantization"], "Q1_0")
        self.assertTrue(result["quantization_matches"])
        self.assertIsNone(result["supported_quantizations"])

    async def test_any_model_auto_profile_accepts_q1(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "example/ordinary-model",
                    "quantization": {"name": "Q1_0"},
                    "loaded_instances": [
                        {
                            "id": "example/ordinary-model",
                            "config": {"context_length": 4096, "parallel": 1},
                        }
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="example/ordinary-model",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=4096,
            local_llm_required_quantization="AUTO",
            llm_max_concurrency=1,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["available"])
        self.assertTrue(result["quantization_matches"])
        self.assertIsNone(result["supported_quantizations"])

    async def test_native_health_reports_loaded_context_and_quantization(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "google/gemma-4-12b-qat",
                    "quantization": {"name": "Q4_0"},
                    "loaded_instances": [
                        {
                            "config": {
                                "context_length": 8192,
                                "parallel": 2,
                                "flash_attention": True,
                            }
                        }
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="google/gemma-4-12b-qat",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=8192,
            local_llm_required_quantization="Q4_0",
            llm_max_concurrency=2,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response) as get:
                result = await provider.health_check()

        self.assertTrue(result["online"])
        self.assertTrue(result["available"])
        self.assertTrue(result["loaded"])
        self.assertTrue(result["context_matches"])
        self.assertTrue(result["parallel_matches"])
        self.assertTrue(result["quantization_matches"])
        self.assertEqual(result["loaded_context_length"], 8192)
        self.assertEqual(result["quantization"], "Q4_0")
        self.assertIn("/api/v1/models", get.call_args.args[0])

    async def test_native_health_rejects_installed_but_unloaded_model(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "qwen/qwen3.5-9b",
                    "quantization": {"name": "Q6_K"},
                    "loaded_instances": [],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-9b",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=8192,
            local_llm_required_quantization="Q6_K",
            llm_max_concurrency=2,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["online"])
        self.assertTrue(result["installed"])
        self.assertFalse(result["loaded"])
        self.assertFalse(result["available"])
        self.assertIn("not loaded", result["error"])

    async def test_native_health_rejects_mismatched_runtime_shape(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "qwen/qwen3.5-9b",
                    "quantization": {"name": "Q8_0"},
                    "loaded_instances": [
                        {"config": {"context_length": 4096, "parallel": 1}}
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-9b",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=8192,
            local_llm_required_quantization="Q6_K",
            llm_max_concurrency=2,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertFalse(result["available"])
        self.assertFalse(result["context_matches"])
        self.assertFalse(result["parallel_matches"])
        self.assertFalse(result["quantization_matches"])
        self.assertIn("loaded context", result["error"])

    async def test_native_health_accepts_larger_loaded_context(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "qwen/qwen3.5-9b",
                    "quantization": {"name": "Q6_K"},
                    "loaded_instances": [
                        {"config": {"context_length": 10240, "parallel": 2}}
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-9b",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=8192,
            local_llm_required_quantization="Q6_K",
            llm_max_concurrency=2,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["available"])
        self.assertTrue(result["context_matches"])
        self.assertEqual(result["loaded_context_length"], 10240)

    async def test_native_health_auto_profile_accepts_supported_variants(self):
        for quantization in ("Q4_K_M", "Q6_K", "Q8_0", "IQ4_XS", "F16", "BF16", "NEW_QUANT", None):
            with self.subTest(quantization=quantization):
                api_response = Mock()
                api_response.raise_for_status.return_value = None
                api_response.json.return_value = {
                    "models": [
                        {
                            "key": "qwen/qwen3.5-9b",
                            "quantization": {"name": quantization},
                            "loaded_instances": [
                                {
                                    "config": {
                                        "context_length": 10240,
                                        "parallel": 2,
                                    }
                                }
                            ],
                        }
                    ]
                }
                local_settings = SimpleNamespace(
                    local_llm_model="qwen/qwen3.5-9b",
                    local_llm_url=(
                        "http://127.0.0.1:1234/v1/chat/completions"
                    ),
                    local_llm_context_size=8192,
                    local_llm_required_quantization="AUTO",
                    llm_max_concurrency=1,
                )

                with patch("llm.client.settings", local_settings):
                    provider = LocalDecisionProvider()
                    with patch(
                        "llm.client.requests.get",
                        return_value=api_response,
                    ):
                        result = await provider.health_check()

                self.assertTrue(result["available"])
                self.assertTrue(result["quantization_matches"])
                self.assertEqual(
                    result["quantization_mode"], "FOLLOW_LOADED"
                )
                self.assertIsNone(result["supported_quantizations"])

    async def test_native_health_auto_profile_accepts_other_quantization(self):
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "models": [
                {
                    "key": "qwen/qwen3.5-9b",
                    "quantization": {"name": "Q5_K_M"},
                    "loaded_instances": [
                        {
                            "config": {
                                "context_length": 10240,
                                "parallel": 2,
                            }
                        }
                    ],
                }
            ]
        }
        local_settings = SimpleNamespace(
            local_llm_model="qwen/qwen3.5-9b",
            local_llm_url="http://127.0.0.1:1234/v1/chat/completions",
            local_llm_context_size=8192,
            local_llm_required_quantization="AUTO",
            llm_max_concurrency=1,
        )

        with patch("llm.client.settings", local_settings):
            provider = LocalDecisionProvider()
            with patch("llm.client.requests.get", return_value=api_response):
                result = await provider.health_check()

        self.assertTrue(result["available"])
        self.assertTrue(result["quantization_matches"])
        self.assertNotIn("error", result)


class LLMClientReadinessTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _client(provider) -> LLMClient:
        client = LLMClient.__new__(LLMClient)
        client.provider = provider
        client.last_latency_seconds = 0.0
        client.last_error = ""
        client.last_telemetry = None
        return client

    async def test_local_readiness_requires_successful_hold_completion(self):
        class Provider:
            name = "local"
            model = "qwen/qwen3.5-9b"

            async def health_check(self):
                return {"online": True, "available": True}

            async def request(self, system_prompt, user_prompt, context=None):
                return DecisionResponse(
                    decision={"action": "HOLD"},
                    telemetry=DecisionTelemetry(
                        trace_id="probe-ok",
                        provider=self.name,
                        model=self.model,
                        started_at_utc="2026-07-24T00:00:00+00:00",
                        latency_seconds=0.25,
                        success=True,
                    ),
                )

        result = await self._client(Provider()).readiness_probe()

        self.assertTrue(result["inference_ready"])
        self.assertEqual(result["probe_latency_seconds"], 0.25)

    async def test_local_readiness_fails_closed_when_completion_fails(self):
        class Provider:
            name = "local"
            model = "qwen/qwen3.5-9b"

            async def health_check(self):
                return {"online": True, "available": True}

            async def request(self, system_prompt, user_prompt, context=None):
                return DecisionResponse(
                    decision=None,
                    telemetry=DecisionTelemetry(
                        trace_id="probe-failed",
                        provider=self.name,
                        model=self.model,
                        started_at_utc="2026-07-24T00:00:00+00:00",
                        latency_seconds=0.1,
                        success=False,
                        error="worker unavailable",
                    ),
                )

        result = await self._client(Provider()).readiness_probe()

        self.assertFalse(result["inference_ready"])
        self.assertEqual(result["error"], "worker unavailable")


class DecisionTraceTests(unittest.TestCase):
    def test_hold_decision_is_persisted_for_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "trace.db")
            logger = TradeReplayLogger(db_path)
            row_id = logger.log_decision_trace(
                symbol="USDJPY",
                candle_time="2026-07-21 10:00:00",
                telemetry={
                    "trace_id": "trace-1",
                    "started_at_utc": "2026-07-21T10:00:01+00:00",
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                    "latency_seconds": 1.25,
                    "success": True,
                    "request_id": "req-1",
                    "prompt_sha256": "abc",
                    "prompt_tokens_estimate": 400,
                    "error": "",
                },
                decision={"action": "HOLD", "confidence": 0.72, "_agent": {}},
                validation_status="VALID",
            )

            self.assertIsNotNone(row_id)
            with closing(sqlite3.connect(db_path)) as conn:
                row = conn.execute(
                    "SELECT provider, model, action, confidence, validation_status "
                    "FROM decision_trace WHERE trace_id = ?",
                    ("trace-1",),
                ).fetchone()
            self.assertEqual(row[:3], ("openai", "gpt-5.6-terra", "HOLD"))
            self.assertAlmostEqual(row[3], 0.72)
            self.assertEqual(row[4], "VALID")


if __name__ == "__main__":
    unittest.main()

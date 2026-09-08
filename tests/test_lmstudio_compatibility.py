"""Compatibility regressions using LM Studio catalog and error response shapes."""

import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import requests

from app_config.settings import settings
from llm.client import LocalDecisionProvider
from llm.lmstudio import reasoning_effort, select_model


def model(key, quantization="IQ4_XS", loaded=True, identifier=None, **extra):
    return {
        "type": "llm", "key": key, "quantization": {"name": quantization},
        "loaded_instances": [{
            "id": identifier or key,
            "config": {"context_length": 8192, "parallel": 1},
        }] if loaded else [],
        **extra,
    }


def response(status=200, payload=None):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(payload or {}).encode()
    return result


class ModelSelectionTests(unittest.TestCase):
    def test_custom_identifier_selects_its_actual_instance(self):
        item = model("publisher/model", identifier="first")
        item["loaded_instances"].append({"id": "my-trader", "config": {}})
        selected, instance, error = select_model("my-trader", [item])
        self.assertIs(selected, item)
        self.assertEqual(instance["id"], "my-trader")
        self.assertFalse(error)

    def test_generic_name_prefers_loaded_variant(self):
        installed = model("qwen3.5-4b@q4_k_s", "Q4_K_S", loaded=False)
        loaded = model("qwen3.5-4b@iq4_xs")
        self.assertIs(select_model("qwen/qwen3.5-4b", [installed, loaded])[0], loaded)

    def test_explicit_variant_never_switches_to_different_loaded_variant(self):
        installed = model("qwen3.5-4b@q4_k_s", "Q4_K_S", loaded=False)
        loaded = model("qwen3.5-4b@iq4_xs")
        selected, instance, error = select_model("qwen3.5-4b@q4_k_s", [loaded, installed])
        self.assertIs(selected, installed)
        self.assertIsNone(instance)
        self.assertFalse(error)

    def test_explicit_quantization_matches_catalog_metadata(self):
        item = model("prism-ml/bonsai-27b", "Q1_0")
        self.assertIs(select_model("prism-ml/bonsai-27b@q1_0", [item])[0], item)
        self.assertIsNone(select_model("prism-ml/bonsai-27b@q4_k_s", [item])[0])

    def test_exact_publisher_wins_and_short_name_ties_are_reported(self):
        items = [model("alice/model"), model("bob/model")]
        self.assertIs(select_model("bob/model", items)[0], items[1])
        self.assertIn("ambiguous", select_model("model", items)[2])

    def test_two_loaded_variants_require_explicit_identifier(self):
        items = [model("model@q4_k_s"), model("model@iq4_xs")]
        self.assertIn("ambiguous", select_model("model", items)[2])

    def test_embedding_is_not_a_chat_model(self):
        self.assertIn("embedding", select_model("embed", [model("embed", type="embedding")])[2])

    def test_reasoning_uses_advertised_options(self):
        for options, expected in [(["off", "on"], "none"), (["low", "high"], "low"), (["on"], None), ([], None)]:
            with self.subTest(options=options):
                self.assertEqual(reasoning_effort({"capabilities": {"reasoning": {"allowed_options": options}}}), expected)


class RequestCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = replace(settings, local_llm_model="custom-chat", local_llm_max_retries=1,
                              local_llm_required_quantization="AUTO", llm_max_concurrency=1,
                              local_llm_context_size=4096)

    async def test_custom_model_capabilities_disable_thinking(self):
        item = model("vendor/future-model", identifier="custom-chat",
                     capabilities={"reasoning": {"allowed_options": ["off", "on"]}})
        with patch("llm.client.settings", self.config), patch("llm.client.requests.get", return_value=response(payload={"models": [item]})):
            provider = LocalDecisionProvider()
            self.assertTrue((await provider.health_check())["available"])
            with patch.object(provider, "_post", return_value={"choices": [{"message": {"content": '{"action":"HOLD"}'}}]}) as post:
                await provider.request("system", "user")
            self.assertEqual(post.call_args.args[0]["reasoning_effort"], "none")

    async def test_reasoning_json_is_not_used_as_final_decision(self):
        for content, finish, succeeds in [
            ('<think>{"action":"BUY"}</think>{"action":"HOLD"}', "stop", True),
            ('<think>{"action":"BUY"}', "stop", False),
            ('{"action":"HOLD"}', "length", False),
            (None, "stop", False),
        ]:
            with self.subTest(content=content, finish=finish), patch("llm.client.settings", self.config):
                provider = LocalDecisionProvider()
                with patch.object(provider, "_post", return_value={"choices": [{"message": {"content": content}, "finish_reason": finish}]}):
                    result = await provider.request("system", "user")
                self.assertEqual(result.telemetry.success, succeeds)
                if succeeds:
                    self.assertEqual(result.decision["action"], "HOLD")

    async def test_native_bad_metadata_does_not_bypass_checks(self):
        item = model("custom-chat")
        item["loaded_instances"][0]["config"]["context_length"] = "invalid"
        with patch("llm.client.settings", self.config), patch("llm.client.requests.get", return_value=response(payload={"models": [item]})) as get:
            result = await LocalDecisionProvider().health_check()
        self.assertFalse(result["available"])
        self.assertEqual(get.call_count, 1)

    async def test_legacy_model_list_reports_missing_and_preserves_pins(self):
        for configured, pin, ready in [("custom-chat", "AUTO", True), ("missing", "AUTO", False), ("custom-chat", "Q8_0", False)]:
            with self.subTest(configured=configured, pin=pin), patch("llm.client.settings", replace(self.config, local_llm_model=configured, local_llm_required_quantization=pin)):
                with patch("llm.client.requests.get", side_effect=[response(404), response(payload={"data": [{"id": "custom-chat"}]})]):
                    result = await LocalDecisionProvider().health_check()
            self.assertEqual(result["available"], ready)
            if not ready:
                self.assertTrue(result["error"])

    def test_unsupported_schema_fallback_is_bounded_and_preserves_json_instruction(self):
        payload = {"model": "custom-chat", "messages": [{"role": "user", "content": "Return HOLD"}], "response_format": {"type": "json_schema"}}
        with patch("llm.client.settings", self.config), patch("llm.client.requests.post", side_effect=[response(400, {"error": "Structured output is not supported"}), response(payload={"choices": []})]) as post:
            LocalDecisionProvider._post(payload)
        self.assertEqual(post.call_count, 2)
        sent = post.call_args.kwargs["json"]
        self.assertNotIn("response_format", sent)
        self.assertIn("ticket_to_close", sent["messages"][-1]["content"])
        self.assertLessEqual(post.call_args.kwargs["timeout"], self.config.local_llm_timeout)
        self.assertIn("response_format", payload)
        self.assertEqual(payload["messages"][0]["content"], "Return HOLD")

    def test_server_errors_do_not_trigger_parameter_fallback(self):
        with patch("llm.client.settings", self.config), patch("llm.client.requests.post", return_value=response(500, {"error": "out of memory"})) as post:
            with self.assertRaises(requests.HTTPError):
                LocalDecisionProvider._post({"model": "custom-chat"})
        self.assertEqual(post.call_count, 1)

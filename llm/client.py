"""Provider-aware, decision-only model service.

The model can propose BUY, SELL, HOLD, or CLOSE. It never receives an order
execution tool; deterministic planning, risk validation, and MT5 execution
remain in the trading engine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol
from urllib.parse import quote

import requests

from app_config.settings import settings
from core.evidence import build_evidence_ids

logger = logging.getLogger("TradingSystem.DecisionService")

LOCAL_LLM_AUTO_QUANTIZATION = "AUTO"
SUPPORTED_LOCAL_LLM_QUANTIZATIONS = ("Q4_K_M", "Q6_K", "Q8_0")


DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "CLOSE"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "ticket_to_close": {"type": ["integer", "null"]},
        "evidence_ids": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 8,
        },
        "reasoning": {"type": "string", "maxLength": 500},
        "trade_management": {"type": "string", "maxLength": 500},
    },
    "required": [
        "action",
        "confidence",
        "ticket_to_close",
        "evidence_ids",
        "reasoning",
        "trade_management",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class DecisionTelemetry:
    trace_id: str
    provider: str
    model: str
    started_at_utc: str
    latency_seconds: float
    success: bool
    request_id: str = ""
    error: str = ""
    prompt_sha256: str = ""
    prompt_tokens_estimate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionResponse:
    decision: Optional[Dict[str, Any]]
    telemetry: DecisionTelemetry


class DecisionProvider(Protocol):
    name: str
    model: str

    async def request(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResponse:
        ...

    async def health_check(self) -> Dict[str, Any]:
        ...


def _prompt_metadata(system_prompt: str, user_prompt: str) -> tuple[str, int]:
    rendered = f"{system_prompt}\n\0\n{user_prompt}"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest(), max(1, len(rendered) // 4)


def _safe_market_score(context: Dict[str, Any]) -> float:
    try:
        value = float((context.get("market") or {}).get("selection_score", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return round(value, 1) if math.isfinite(value) else 0.0


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _local_model_aliases(model_id: object) -> set[str]:
    """Return stable LM Studio aliases for a configured or loaded model ID.

    LM Studio's catalog uses IDs such as ``qwen/qwen3.5-4b``, while a locally
    imported GGUF can expose the shorter API identifier ``qwen3.5-4b``. Those
    names refer to the same selectable model and must not fail readiness solely
    because one side includes the publisher namespace.
    """
    normalized = str(model_id or "").strip().casefold()
    if not normalized:
        return set()
    aliases = {normalized}
    if "/" in normalized:
        aliases.add(normalized.rsplit("/", 1)[-1])
    return aliases


def _local_model_ids_match(configured: object, candidate: object) -> bool:
    return bool(
        _local_model_aliases(configured) & _local_model_aliases(candidate)
    )


def _is_qwen35_model(model_id: object) -> bool:
    return any(alias.startswith("qwen3.5-") for alias in _local_model_aliases(model_id))


def _is_bonsai27_model(model_id: object) -> bool:
    """Recognize Prism/LM Studio aliases for the binary Bonsai 27B model."""
    return any(
        alias == "bonsai-27b" or alias.startswith("bonsai-27b-")
        for alias in _local_model_aliases(model_id)
    )


def _requires_nonthinking_response(model_id: object) -> bool:
    # Both model families default to a reasoning pass in LM Studio. Trading
    # decisions are bounded JSON, so hidden reasoning must not consume the
    # completion budget before the schema-constrained answer is emitted.
    return _is_qwen35_model(model_id) or _is_bonsai27_model(model_id)


def _supported_quantizations_for_model(model_id: object) -> tuple[str, ...]:
    # Q1_0 is normally too lossy for the decision lane. Bonsai 27B is an
    # explicitly trained binary model whose native, validated GGUF is Q1_0, so
    # admit that quantization only for this named family.
    if _is_bonsai27_model(model_id):
        return (*SUPPORTED_LOCAL_LLM_QUANTIZATIONS, "Q1_0")
    return SUPPORTED_LOCAL_LLM_QUANTIZATIONS


class LocalDecisionProvider:
    name = "local"

    def __init__(self) -> None:
        self.configured_model = settings.local_llm_model
        self.model = self.configured_model

    async def request(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResponse:
        trace_id = uuid.uuid4().hex
        prompt_hash, prompt_tokens = _prompt_metadata(system_prompt, user_prompt)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        last_error = ""

        for attempt in range(1, settings.local_llm_max_retries + 1):
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": settings.local_llm_temperature,
                "top_p": settings.local_llm_top_p,
                "seed": settings.local_llm_seed,
                "max_tokens": settings.local_llm_max_tokens,
                "stream": False,
            }
            # Supported hybrid-thinking models default to reasoning in LM
            # Studio. Keep this short deterministic lane non-thinking so the
            # bounded completion can always reach its JSON answer.
            if _requires_nonthinking_response(self.model):
                payload["reasoning_effort"] = "none"
            if settings.local_llm_structured_output:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trading_decision",
                        "strict": True,
                        "schema": DECISION_SCHEMA,
                    },
                }
            try:
                response = await asyncio.to_thread(self._post, payload)
                choices = response.get("choices") or []
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                decision = _parse_json(content)
                if decision is not None:
                    telemetry = DecisionTelemetry(
                        trace_id=trace_id,
                        provider=self.name,
                        model=self.model,
                        started_at_utc=started_at,
                        latency_seconds=time.monotonic() - started,
                        success=True,
                        request_id=str(response.get("id", "")),
                        prompt_sha256=prompt_hash,
                        prompt_tokens_estimate=prompt_tokens,
                    )
                    return DecisionResponse(decision=decision, telemetry=telemetry)
                last_error = "Model returned invalid JSON"
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning(
                    "Local decision request %s/%s failed: %s",
                    attempt,
                    settings.local_llm_max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.exception("Unexpected local inference failure")
            if attempt < settings.local_llm_max_retries:
                await asyncio.sleep(min(attempt, 2))

        return DecisionResponse(
            decision=None,
            telemetry=DecisionTelemetry(
                trace_id=trace_id,
                provider=self.name,
                model=self.model,
                started_at_utc=started_at,
                latency_seconds=time.monotonic() - started,
                success=False,
                error=last_error or "Local decision request failed",
                prompt_sha256=prompt_hash,
                prompt_tokens_estimate=prompt_tokens,
            ),
        )

    @staticmethod
    def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            settings.local_llm_url,
            json=payload,
            timeout=settings.local_llm_timeout,
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        base_url = settings.local_llm_url.split("/v1/", 1)[0]

        def _get(url: str) -> Dict[str, Any]:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            return response.json()

        try:
            # The native endpoint distinguishes downloaded models from loaded
            # inference instances and reports the context actually allocated.
            payload = await asyncio.to_thread(_get, f"{base_url}/api/v1/models")
            models = payload.get("models") or []
            model_info = next(
                (
                    item
                    for item in models
                    if _local_model_ids_match(
                        self.configured_model, item.get("key")
                    )
                ),
                None,
            )
            if model_info is None:
                return {
                    "online": True,
                    "available": False,
                    "loaded": False,
                    "selected_model": self.model,
                    "error": "Configured model is not installed in LM Studio",
                }
            instances = model_info.get("loaded_instances") or []
            resolved_model = str(
                (
                    instances[0].get("id")
                    if instances
                    else model_info.get("key")
                )
                or self.configured_model
            )
            self.model = resolved_model
            instance_config = (instances[0].get("config") or {}) if instances else {}
            loaded_context = instance_config.get("context_length")
            loaded_parallel = instance_config.get("parallel")
            configured_context = settings.local_llm_context_size
            loaded_quantization = (model_info.get("quantization") or {}).get("name")
            required_quantization = str(
                settings.local_llm_required_quantization or ""
            ).strip().upper()
            loaded_quantization_normalized = str(
                loaded_quantization or ""
            ).strip().upper()
            supported_quantizations = _supported_quantizations_for_model(
                self.configured_model
            )
            # LM Studio may allocate a larger context window than this client
            # needs.  That is compatible: the configured value is the minimum
            # capacity required by the trading prompts, not an exact runtime
            # shape that must be duplicated by the server.
            context_matches = (
                int(loaded_context) >= int(configured_context)
                if loaded_context is not None
                else None
            )
            parallel_matches = (
                int(loaded_parallel) >= int(settings.llm_max_concurrency)
                if loaded_parallel is not None
                else None
            )
            quantization_mode = (
                "FOLLOW_LOADED"
                if required_quantization == LOCAL_LLM_AUTO_QUANTIZATION
                else "PINNED" if required_quantization else "UNRESTRICTED"
            )
            if quantization_mode == "FOLLOW_LOADED":
                quantization_matches = (
                    loaded_quantization_normalized
                    in supported_quantizations
                )
            else:
                quantization_matches = (
                    loaded_quantization_normalized == required_quantization
                    if required_quantization
                    else True
                )
            ready = bool(instances) and context_matches is not False \
                and parallel_matches is not False and quantization_matches
            result = {
                "online": True,
                "installed": True,
                "available": ready,
                "loaded": bool(instances),
                "selected_model": self.model,
                "configured_model": self.configured_model,
                "resolved_model": resolved_model,
                "loaded_context_length": loaded_context,
                "configured_context_length": configured_context,
                "context_matches": context_matches,
                "loaded_parallel": loaded_parallel,
                "configured_parallel": settings.llm_max_concurrency,
                "parallel_matches": parallel_matches,
                "flash_attention": instance_config.get("flash_attention"),
                "quantization": loaded_quantization,
                "required_quantization": required_quantization or None,
                "quantization_mode": quantization_mode,
                "supported_quantizations": list(supported_quantizations),
                "quantization_matches": quantization_matches,
            }
            if not instances:
                result["error"] = "Configured model is installed but not loaded in LM Studio"
            elif context_matches is False:
                result["error"] = (
                    f"LM Studio loaded context is {loaded_context}, below the project "
                    f"minimum of {configured_context}"
                )
            elif parallel_matches is False:
                result["error"] = (
                    f"LM Studio permits {loaded_parallel} parallel prediction(s), but "
                    f"the project is configured for {settings.llm_max_concurrency}"
                )
            elif not quantization_matches:
                if quantization_mode == "FOLLOW_LOADED":
                    supported = " or ".join(
                        supported_quantizations
                    )
                    result["error"] = (
                        "LM Studio loaded quantization is "
                        f"{loaded_quantization or 'unknown'}, but automatic "
                        f"switching supports {supported}"
                    )
                else:
                    result["error"] = (
                        "LM Studio loaded quantization is "
                        f"{loaded_quantization or 'unknown'}, but the project "
                        f"requires {required_quantization}"
                    )
            return result
        except Exception as native_exc:
            # Older LM Studio releases expose only the OpenAI-compatible model
            # list. That confirms installation but not loaded-instance details.
            try:
                payload = await asyncio.to_thread(_get, f"{base_url}/v1/models")
                ids = [item.get("id") for item in payload.get("data", [])]
                resolved_model = next(
                    (
                        model_id
                        for model_id in ids
                        if _local_model_ids_match(
                            self.configured_model, model_id
                        )
                    ),
                    None,
                )
                if resolved_model:
                    self.model = str(resolved_model)
                return {
                    "online": True,
                    "available": resolved_model is not None,
                    "loaded": None,
                    "selected_model": self.model,
                    "configured_model": self.configured_model,
                    "resolved_model": resolved_model,
                    "configured_context_length": settings.local_llm_context_size,
                }
            except Exception as fallback_exc:
                return {
                    "online": False,
                    "available": False,
                    "loaded": False,
                    "selected_model": self.model,
                    "error": str(fallback_exc or native_exc),
                }


class DeterministicDecisionProvider:
    """Millisecond trend-following policy built from completed-candle evidence."""

    name = "deterministic"
    model = "rules-v2-adaptive"

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _structure(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return (analysis or {}).get("market_structure", {}) or {}

    @classmethod
    def _direction(cls, analysis: Optional[Dict[str, Any]], *, state: bool) -> str:
        structure = cls._structure(analysis)
        value = (
            structure.get("trend_state_direction") or structure.get("trend")
            if state
            else structure.get("trend")
        )
        direction = str(value or "NEUTRAL").upper()
        return direction if direction in {"BULLISH", "BEARISH"} else "NEUTRAL"

    @staticmethod
    def _hold(reason: str) -> Dict[str, Any]:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "ticket_to_close": None,
            "evidence_ids": [],
            "reasoning": reason[:500],
            "trade_management": "Wait for a newly completed M5 candle with aligned evidence.",
        }

    @classmethod
    def _momentum(cls, analysis: Dict[str, Any], action: str) -> tuple[int, Dict[str, bool]]:
        indicators = analysis.get("indicators", {}) or {}
        ema9 = cls._number(indicators.get("ema_9"))
        ema21 = cls._number(indicators.get("ema_21"))
        rsi = cls._number(indicators.get("rsi_14"))
        macd = cls._number((indicators.get("macd") or {}).get("diff"))
        votes = {
            "EMA": bool(
                ema9 is not None
                and ema21 is not None
                and (ema9 > ema21 if action == "BUY" else ema9 < ema21)
            ),
            "RSI": bool(
                rsi is not None
                and (50.0 <= rsi <= 68.0 if action == "BUY" else 32.0 <= rsi <= 50.0)
            ),
            "MACD": bool(
                macd is not None and (macd > 0.0 if action == "BUY" else macd < 0.0)
            ),
        }
        return sum(votes.values()), votes

    @classmethod
    def _structure_flags(
        cls, analysis: Dict[str, Any], expected: str
    ) -> tuple[bool, bool, bool]:
        structure = cls._structure(analysis)
        matching = [
            event
            for event in structure.get("structure_events", [])
            if isinstance(event, dict)
            and str(event.get("direction", "")).upper() == expected
        ]
        has_bos = any(str(event.get("type", "")).upper() == "BOS" for event in matching)
        has_choch = any(str(event.get("type", "")).upper() == "CHOCH" for event in matching)
        breakout = str(structure.get("breakout_status", "")).upper()
        return has_bos, has_choch, expected in breakout and "BREAKOUT" in breakout

    @classmethod
    def _decide(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("has_open_position"):
            return cls._hold(
                "A position is already open; deterministic SL/TP, break-even, and "
                "trailing protection remain in control."
            )

        analyses = context.get("analyses") or {}
        if not all(analyses.get(key) for key in ("M5", "M15", "H1", "H4")):
            return cls._hold("Complete M5, M15, H1, and H4 analysis is required.")

        m5 = analyses["M5"]
        indicators = m5.get("indicators", {}) or {}
        range_atr = cls._number(indicators.get("candle_range_atr")) or 0.0
        gap_atr = cls._number(indicators.get("opening_gap_atr")) or 0.0
        if (
            range_atr >= settings.market_shock_range_atr
            or gap_atr >= settings.market_shock_gap_atr
        ):
            return cls._hold(
                f"MARKET_SHOCK: M5 range={range_atr:.2f} ATR, gap={gap_atr:.2f} ATR; "
                "wait for completed post-shock structure."
            )

        timeframes = ("M5", "M15", "H1", "H4")
        slow = {key: cls._direction(analyses[key], state=False) for key in timeframes}
        state_direction = {key: cls._direction(analyses[key], state=True) for key in timeframes}
        state_name = {
            key: str(cls._structure(analyses[key]).get("trend_state", "NEUTRAL")).upper()
            for key in timeframes
        }
        previous = {
            str(key).upper(): str(value).upper()
            for key, value in (context.get("previous_trend_states") or {}).items()
        }
        m5_adx = cls._number(indicators.get("adx_14")) or 0.0
        m15_adx = cls._number(
            (analyses["M15"].get("indicators", {}) or {}).get("adx_14")
        ) or 0.0

        candidates: list[Dict[str, Any]] = []
        for expected, action in (("BULLISH", "BUY"), ("BEARISH", "SELL")):
            momentum_votes, _ = cls._momentum(m5, action)
            has_bos, has_choch, has_breakout = cls._structure_flags(m5, expected)
            m15_bos, m15_choch, _ = cls._structure_flags(analyses["M15"], expected)
            retest = cls._structure(m5).get("retest_continuation")
            has_retest = bool(
                isinstance(retest, dict)
                and str(retest.get("direction", "")).upper() == expected
            )

            reversal = bool(
                settings.adaptive_reversal_enabled
                and state_name["M5"] == f"EARLY_{expected}_REVERSAL"
                and state_direction["M15"] == expected
                and has_choch
                and (has_bos or m15_bos or m15_choch or has_breakout)
                and momentum_votes == 3
                and m5_adx >= settings.adaptive_reversal_min_adx
                and m15_adx >= settings.confirmation_min_adx
            )
            resumed_pullback = bool(
                has_retest
                and previous.get("M5") == f"PULLBACK_IN_{expected}_TREND"
                and state_name["M5"] == f"CONFIRMED_{expected}"
                and slow["M15"] == expected
                and slow["H1"] == expected
                and state_direction["M15"] == expected
                and momentum_votes >= 2
                and (has_bos or has_breakout or momentum_votes >= 2)
                and m5_adx >= 20.0
                and m15_adx >= settings.confirmation_min_adx
            )
            continuation = bool(
                slow["M5"] == slow["M15"] == slow["H1"] == expected
                and state_direction["M5"] == expected
                and state_direction["M15"] == expected
                and not state_name["M5"].startswith("PULLBACK_")
                and momentum_votes >= 2
                and m5_adx >= 20.0
                and m15_adx >= settings.confirmation_min_adx
            )

            mode = (
                "CONFIRMED_REVERSAL"
                if reversal
                else "PULLBACK_RESUMPTION"
                if resumed_pullback
                else "TREND_CONTINUATION"
                if continuation
                else ""
            )
            if not mode:
                continue

            # Continuations and resumptions against H4 need the existing
            # high-strength structural exception. Confirmed reversals are
            # already held to CHoCH plus secondary structure confirmation.
            h4_conflict = slow["H4"] not in {expected, "NEUTRAL"}
            rsi = cls._number(indicators.get("rsi_14"))
            safe_rsi = bool(
                rsi is not None
                and (
                    rsi <= settings.countertrend_buy_max_rsi
                    if action == "BUY"
                    else rsi >= settings.countertrend_sell_min_rsi
                )
            )
            if h4_conflict and mode != "CONFIRMED_REVERSAL":
                if not (
                    settings.allow_strong_countertrend_entries
                    and m5_adx >= settings.countertrend_min_adx
                    and safe_rsi
                    and (has_bos or has_choch or has_retest)
                ):
                    continue

            stochastic = indicators.get("stochastic") or {}
            bands = indicators.get("bollinger_bands") or {}
            stoch_k = cls._number(stochastic.get("k"))
            rsi = cls._number(indicators.get("rsi_14"))
            atr = cls._number(indicators.get("atr_14"))
            current = cls._number(indicators.get("current_price"))
            outer = cls._number(
                bands.get("upper") if action == "BUY" else bands.get("lower")
            )
            if all(
                value is not None for value in (stoch_k, rsi, atr, current, outer)
            ) and atr > 0:
                overshoot_atr = max(
                    0.0,
                    (current - outer) / atr
                    if action == "BUY"
                    else (outer - current) / atr,
                )
                if (
                    action == "BUY"
                    and overshoot_atr
                    >= settings.overextension_min_band_overshoot_atr
                    and stoch_k >= settings.overextension_stoch_high
                    and rsi >= settings.overextension_rsi_high
                ) or (
                    action == "SELL"
                    and overshoot_atr
                    >= settings.overextension_min_band_overshoot_atr
                    and stoch_k <= settings.overextension_stoch_low
                    and rsi <= settings.overextension_rsi_low
                ):
                    continue

            base = {
                "TREND_CONTINUATION": 0.66,
                "PULLBACK_RESUMPTION": 0.70,
                "CONFIRMED_REVERSAL": 0.73,
            }[mode]
            confidence = base + momentum_votes * 0.025
            confidence += min(0.05, max(0.0, (m5_adx - 20.0) / 200.0))
            confidence += 0.035 if has_bos else 0.0
            confidence += 0.035 if has_choch else 0.0
            confidence += 0.025 if has_breakout else 0.0
            confidence += 0.03 if slow["H4"] == expected else 0.0
            candidates.append(
                {
                    "action": action,
                    "expected": expected,
                    "mode": mode,
                    "confidence": round(min(0.95, confidence), 3),
                    "momentum_votes": momentum_votes,
                    "has_bos": has_bos,
                    "has_choch": has_choch,
                    "has_breakout": has_breakout,
                    "has_retest": has_retest,
                }
            )

        if not candidates:
            return cls._hold(
                "No continuation, pullback-resumption, or confirmed-reversal state "
                "has enough completed-candle confirmation."
            )
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        best = candidates[0]
        labels = [
            label
            for label, active in (
                ("BOS", best["has_bos"]),
                ("CHoCH", best["has_choch"]),
                ("breakout", best["has_breakout"]),
                ("verified retest", best["has_retest"]),
            )
            if active
        ]
        structure_summary = "/".join(labels) if labels else "state alignment"
        return {
            "action": best["action"],
            "confidence": best["confidence"],
            "ticket_to_close": None,
            "evidence_ids": list(build_evidence_ids(analyses))[:8],
            "reasoning": (
                f"{best['mode']}: {best['action']} with momentum "
                f"{best['momentum_votes']}/3, ADX {m5_adx:.1f}/{m15_adx:.1f}, "
                f"structure={structure_summary}."
            )[:500],
            "trade_management": (
                "Planner and risk engine own entry, stop, target, size, margin, spread, "
                "shock filtering, and broker validation."
            ),
            "_strategy": {
                "mode": best["mode"],
                "expected_direction": best["expected"],
                "market_selection_score": _safe_market_score(context),
            },
        }

    async def request(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResponse:
        trace_id = uuid.uuid4().hex
        prompt_hash, prompt_tokens = _prompt_metadata(system_prompt, user_prompt)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        if context is None:
            return DecisionResponse(
                decision=None,
                telemetry=DecisionTelemetry(
                    trace_id=trace_id,
                    provider=self.name,
                    model=self.model,
                    started_at_utc=started_at,
                    latency_seconds=time.perf_counter() - started,
                    success=False,
                    error="Deterministic decision context is missing",
                    prompt_sha256=prompt_hash,
                    prompt_tokens_estimate=prompt_tokens,
                ),
            )
        decision = self._decide(context)
        return DecisionResponse(
            decision=decision,
            telemetry=DecisionTelemetry(
                trace_id=trace_id,
                provider=self.name,
                model=self.model,
                started_at_utc=started_at,
                latency_seconds=time.perf_counter() - started,
                success=True,
                prompt_sha256=prompt_hash,
                prompt_tokens_estimate=prompt_tokens,
            ),
        )

    async def health_check(self) -> Dict[str, Any]:
        return {
            "online": True,
            "available": True,
            "configured": True,
            "selected_model": self.model,
        }


class OpenAIResponsesProvider:
    name = "openai"

    def __init__(self) -> None:
        self.model = settings.openai_model

    @staticmethod
    def _headers() -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        if settings.openai_organization:
            headers["OpenAI-Organization"] = settings.openai_organization
        if settings.openai_project:
            headers["OpenAI-Project"] = settings.openai_project
        return headers

    async def request(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResponse:
        trace_id = uuid.uuid4().hex
        prompt_hash, prompt_tokens = _prompt_metadata(system_prompt, user_prompt)
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        last_error = ""

        if not settings.openai_api_key:
            return DecisionResponse(
                decision=None,
                telemetry=DecisionTelemetry(
                    trace_id=trace_id,
                    provider=self.name,
                    model=self.model,
                    started_at_utc=started_at,
                    latency_seconds=0.0,
                    success=False,
                    error="OPENAI_API_KEY is not configured",
                    prompt_sha256=prompt_hash,
                    prompt_tokens_estimate=prompt_tokens,
                ),
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": settings.openai_reasoning_effort},
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "trading_decision",
                    "strict": True,
                    "schema": DECISION_SCHEMA,
                },
            },
            "max_output_tokens": 350,
            "store": False,
        }

        for attempt in range(1, settings.openai_max_retries + 1):
            try:
                response, request_id = await asyncio.to_thread(self._post, payload)
                decision = _parse_json(self._extract_output_text(response))
                if decision is not None:
                    telemetry = DecisionTelemetry(
                        trace_id=trace_id,
                        provider=self.name,
                        model=str(response.get("model") or self.model),
                        started_at_utc=started_at,
                        latency_seconds=time.monotonic() - started,
                        success=True,
                        request_id=request_id or str(response.get("id", "")),
                        prompt_sha256=prompt_hash,
                        prompt_tokens_estimate=prompt_tokens,
                    )
                    return DecisionResponse(decision=decision, telemetry=telemetry)
                last_error = "OpenAI response did not contain valid decision JSON"
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning(
                    "OpenAI decision request %s/%s failed: %s",
                    attempt,
                    settings.openai_max_retries,
                    exc,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.exception("Unexpected OpenAI inference failure")
            if attempt < settings.openai_max_retries:
                await asyncio.sleep(min(attempt, 2))

        return DecisionResponse(
            decision=None,
            telemetry=DecisionTelemetry(
                trace_id=trace_id,
                provider=self.name,
                model=self.model,
                started_at_utc=started_at,
                latency_seconds=time.monotonic() - started,
                success=False,
                error=last_error or "OpenAI decision request failed",
                prompt_sha256=prompt_hash,
                prompt_tokens_estimate=prompt_tokens,
            ),
        )

    @classmethod
    def _post(cls, payload: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        response = requests.post(
            f"{settings.openai_base_url}/responses",
            headers=cls._headers(),
            json=payload,
            timeout=settings.openai_timeout,
        )
        response.raise_for_status()
        return response.json(), str(response.headers.get("x-request-id", ""))

    @staticmethod
    def _extract_output_text(payload: Dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        chunks: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
        return "".join(chunks)

    async def health_check(self) -> Dict[str, Any]:
        if not settings.openai_api_key:
            return {
                "online": False,
                "available": False,
                "configured": False,
                "selected_model": self.model,
                "error": "OPENAI_API_KEY is not configured",
            }

        def _get() -> None:
            response = requests.get(
                f"{settings.openai_base_url}/models/{quote(self.model, safe='')}",
                headers=self._headers(),
                timeout=min(settings.openai_timeout, 8.0),
            )
            response.raise_for_status()

        try:
            await asyncio.to_thread(_get)
            return {
                "online": True,
                "available": True,
                "configured": True,
                "selected_model": self.model,
            }
        except Exception as exc:
            return {
                "online": False,
                "available": False,
                "configured": True,
                "selected_model": self.model,
                "error": str(exc),
            }


class LLMClient:
    """Facade retained for compatibility with the existing engine and API."""

    def __init__(self) -> None:
        if settings.llm_provider == "openai":
            self.provider: DecisionProvider = OpenAIResponsesProvider()
        elif settings.llm_provider == "local":
            self.provider = LocalDecisionProvider()
        else:
            self.provider = DeterministicDecisionProvider()
        self.last_latency_seconds = 0.0
        self.last_error = ""
        self.last_telemetry: Optional[DecisionTelemetry] = None

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def model_name(self) -> str:
        return self.provider.model

    async def request_decision(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DecisionResponse:
        response = await self.provider.request(
            system_prompt, user_prompt, context=context
        )
        self.last_telemetry = response.telemetry
        self.last_latency_seconds = response.telemetry.latency_seconds
        self.last_error = response.telemetry.error
        decision = dict(response.decision) if response.decision is not None else None
        if decision is not None:
            decision["_agent"] = response.telemetry.to_dict()
        logger.info(
            "Decision trace=%s provider=%s model=%s success=%s latency=%.2fs",
            response.telemetry.trace_id,
            response.telemetry.provider,
            response.telemetry.model,
            response.telemetry.success,
            response.telemetry.latency_seconds,
        )
        return DecisionResponse(decision=decision, telemetry=response.telemetry)

    async def get_trading_decision(
        self,
        system_prompt: str,
        user_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible method used by older integrations and tests."""
        return (
            await self.request_decision(
                system_prompt, user_prompt, context=context
            )
        ).decision

    async def health_check(self) -> Dict[str, Any]:
        result = await self.provider.health_check()
        return {
            **result,
            "provider": self.provider_name,
            "selected_model": self.model_name,
            "max_concurrency": settings.llm_max_concurrency,
        }

    async def readiness_probe(self) -> Dict[str, Any]:
        """Verify that a loaded local provider can complete one response.

        LM Studio can report a model as loaded before its inference worker is
        ready.  The bounded HOLD probe keeps entry authorization fail-closed
        during that warm-up window.  Remote and deterministic providers retain
        their existing health checks so this method adds no paid request.
        """
        health = await self.health_check()
        metadata_ready = bool(
            health.get("online") and health.get("available")
        )
        if not metadata_ready:
            return {
                **health,
                "inference_ready": False,
                "probe_latency_seconds": 0.0,
            }
        if self.provider_name != "local":
            return {
                **health,
                "inference_ready": True,
                "probe_latency_seconds": 0.0,
            }

        response = await self.provider.request(
            (
                "You are a local inference readiness probe. Return only the "
                "required JSON object and never propose a trade."
            ),
            (
                'Return {"action":"HOLD","confidence":0,'
                '"ticket_to_close":null,"evidence_ids":[],'
                '"reasoning":"ready","trade_management":"none"}.'
            ),
            context=None,
        )
        decision = response.decision or {}
        inference_ready = bool(
            response.telemetry.success
            and str(decision.get("action", "")).upper() == "HOLD"
        )
        return {
            **health,
            "inference_ready": inference_ready,
            "probe_latency_seconds": response.telemetry.latency_seconds,
            "error": (
                ""
                if inference_ready
                else response.telemetry.error or "Inference probe failed"
            ),
        }

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Compatibility hook retained for existing callers."""
        return _parse_json(text)

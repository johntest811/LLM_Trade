"""Broker-derived cross-pair context for autonomous FX decisions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple


KNOWN_CURRENCIES = {
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
    "USD",
}


def split_fx_symbol(symbol: str) -> Optional[Tuple[str, str]]:
    normalized = "".join(
        character for character in str(symbol).upper() if character.isalpha()
    )
    if len(normalized) < 6:
        return None
    base, quote = normalized[:3], normalized[3:6]
    if base not in KNOWN_CURRENCIES or quote not in KNOWN_CURRENCIES:
        return None
    return base, quote


def active_sessions(now_utc: Optional[datetime] = None) -> list[str]:
    """Return broad UTC liquidity sessions without pretending DST precision."""
    now_utc = now_utc or datetime.now(timezone.utc)
    hour = now_utc.astimezone(timezone.utc).hour
    sessions: list[str] = []
    if 0 <= hour < 9:
        sessions.append("ASIA")
    if 7 <= hour < 16:
        sessions.append("LONDON")
    if 12 <= hour < 21:
        sessions.append("NEW_YORK")
    return sessions or ["ROLLOVER"]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def build_currency_context(
    rows: Iterable[Dict[str, Any]],
    *,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate normalized M5 movement across the configured FX universe."""
    totals: Dict[str, float] = defaultdict(float)
    samples: Dict[str, int] = defaultdict(int)
    prepared: list[tuple[Dict[str, Any], str, str, float]] = []
    for row in rows:
        pair = split_fx_symbol(str(row.get("symbol", "")))
        if pair is None:
            continue
        base, quote = pair
        impulse = max(-2.0, min(2.0, _finite(row.get("m5_impulse_atr"))))
        regime = str(row.get("selection_regime", "") or "").upper()
        regime_vote = (
            0.5
            if "BULLISH" in regime
            else -0.5
            if "BEARISH" in regime
            else 0.0
        )
        signal = impulse + regime_vote
        totals[base] += signal
        totals[quote] -= signal
        samples[base] += 1
        samples[quote] += 1
        prepared.append((row, base, quote, signal))

    sessions = active_sessions(now_utc)
    contexts: Dict[str, Dict[str, Any]] = {}
    for row, base, quote, own_signal in prepared:
        # Leave the target pair out. Its own impulse/regime cannot be reused as
        # supposedly independent currency-strength confirmation.
        base_samples = max(0, samples.get(base, 0) - 1)
        quote_samples = max(0, samples.get(quote, 0) - 1)
        base_strength = (
            (totals.get(base, 0.0) - own_signal) / base_samples
            if base_samples
            else 0.0
        )
        quote_strength = (
            (totals.get(quote, 0.0) + own_signal) / quote_samples
            if quote_samples
            else 0.0
        )
        pair_strength = base_strength - quote_strength
        reliable = base_samples >= 1 and quote_samples >= 1
        bias = (
            "BULLISH"
            if reliable and pair_strength >= 0.35
            else "BEARISH"
            if reliable and pair_strength <= -0.35
            else "NEUTRAL"
        )
        regime = str(row.get("selection_regime", "") or "").upper()
        aligned = (
            ("BULLISH" in regime and bias == "BULLISH")
            or ("BEARISH" in regime and bias == "BEARISH")
        )
        contexts[str(row.get("symbol", "")).upper()] = {
            "base_currency": base,
            "quote_currency": quote,
            "base_strength": round(base_strength, 3),
            "quote_strength": round(quote_strength, 3),
            "pair_strength": round(pair_strength, 3),
            "bias": bias,
            "reliable": reliable,
            "aligned_with_regime": aligned,
            "currency_samples": {
                base: base_samples,
                quote: quote_samples,
            },
            "active_sessions_utc": sessions,
            "source": "BROKER_M5_CROSS_PAIR",
        }
    return contexts

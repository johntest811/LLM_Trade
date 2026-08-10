"""Helpers for normalizing broker-server timestamps to UTC.

MetaQuotes documents Python timestamps as UTC, but some broker terminals expose
the server wall-clock value as the epoch. Pepperstone uses GMT+3 during US DST
and GMT+2 otherwise, so an untreated timestamp can look several hours ahead of
the operating-system clock. These helpers detect only a plausible *positive*,
whole-hour offset; genuinely old ticks are never shifted forward and disguised
as current data.
"""

from __future__ import annotations

import math
import threading
import time


MAX_SERVER_OFFSET_HOURS = 14
OFFSET_MATCH_TOLERANCE_SECONDS = 5 * 60
MIN_OFFSET_DETECTION_SECONDS = 30 * 60

_offset_lock = threading.RLock()
_server_offsets: dict[str, int] = {}


def remember_server_offset_seconds(symbol: str, offset_seconds: int) -> int:
    """Remember a verified broker clock offset for later quote validation."""
    key = str(symbol or "").strip().upper()
    offset = max(0, int(offset_seconds))
    if key:
        with _offset_lock:
            _server_offsets[key] = offset
    return offset


def remembered_server_offset_seconds(symbol: str) -> int | None:
    """Return the last verified offset for a symbol, if one is known."""
    key = str(symbol or "").strip().upper()
    if not key:
        return None
    with _offset_lock:
        return _server_offsets.get(key)


def infer_positive_server_offset_seconds(
    broker_epoch: float,
    *,
    now_epoch: float | None = None,
) -> int:
    """Return a plausible positive whole-hour broker offset, otherwise zero."""
    try:
        broker_value = float(broker_epoch)
        now_value = float(time.time() if now_epoch is None else now_epoch)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(broker_value) or broker_value <= 0:
        return 0
    if not math.isfinite(now_value) or now_value <= 0:
        return 0

    delta = broker_value - now_value
    if delta < MIN_OFFSET_DETECTION_SECONDS:
        return 0
    hours = int(round(delta / 3600.0))
    if hours < 1 or hours > MAX_SERVER_OFFSET_HOURS:
        return 0
    candidate = hours * 3600
    if abs(delta - candidate) > OFFSET_MATCH_TOLERANCE_SECONDS:
        return 0
    return candidate


def normalized_broker_epoch(broker_epoch: float, offset_seconds: int) -> float:
    """Convert a broker wall-clock epoch to UTC epoch seconds."""
    return float(broker_epoch) - max(0, int(offset_seconds))


def broker_tick_age_seconds(
    broker_epoch: float,
    *,
    now_epoch: float | None = None,
    offset_seconds: int | None = None,
    symbol: str = "",
) -> float:
    """Return tick age after normalizing a detected broker-server offset."""
    now_value = float(time.time() if now_epoch is None else now_epoch)
    if offset_seconds is not None:
        offset = remember_server_offset_seconds(symbol, offset_seconds)
    else:
        detected = infer_positive_server_offset_seconds(
            broker_epoch, now_epoch=now_value
        )
        if detected:
            offset = remember_server_offset_seconds(symbol, detected)
        else:
            remembered = remembered_server_offset_seconds(symbol)
            offset = 0 if remembered is None else remembered

    normalized = normalized_broker_epoch(broker_epoch, offset)
    # A future quote which does not match a known whole-hour broker offset is
    # ambiguous. Fail closed instead of treating it as zero seconds old.
    if normalized > now_value + OFFSET_MATCH_TOLERANCE_SECONDS:
        return math.inf
    return max(0.0, now_value - normalized)

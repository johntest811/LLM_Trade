"""Bounded, rotating discovery of broker instruments (never submits orders)."""

import math
from typing import Any, Iterable


def tradable_symbols(instruments: Iterable[Any]) -> list[str]:
    """Keep real instruments with usable contract metadata and entry permission."""
    instruments = list(instruments)
    spellings = {}
    for info in instruments:
        name = str(getattr(info, "name", "") or "")
        spellings.setdefault(name.upper(), set()).add(name)
    result = []
    for info in instruments:
        name = str(getattr(info, "name", "") or "")
        if not name or name != name.strip() or getattr(info, "custom", False):
            continue
        if len(spellings[name.upper()]) != 1:
            continue
        try:
            if int(info.trade_mode) not in (1, 2, 4):
                continue
            if not all(math.isfinite(float(value)) and float(value) > 0 for value in (
                info.point, info.trade_tick_size, info.volume_min, info.volume_step,
            )):
                continue
            # Market orders and broker SL/TP must all be supported.
            if (int(info.order_mode) & 49) != 49:
                continue
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        result.append(name)
    visible = {getattr(info, "name", "") for info in instruments if getattr(info, "visible", False)}
    return sorted(set(result), key=lambda name: (name not in visible, name.upper(), name))


def discovery_batch(
    configured: list[str], selected: Iterable[str], universe: list[str],
    cursor: int, batch_size: int,
) -> tuple[list[str], int]:
    """Always revisit configured/active markets; rotate additional instruments."""
    priority = list(dict.fromkeys(name.strip().upper() for name in [*configured, *selected] if name.strip()))
    known = set(priority)
    if not universe:
        return priority, 0
    # Anchor the cursor in the complete catalog. Removing newly selected
    # markets before indexing would otherwise skip other markets each cycle.
    start = cursor % len(universe)
    extras = []
    visited = 0
    while visited < len(universe) and len(extras) < max(1, batch_size):
        name = universe[(start + visited) % len(universe)].strip().upper()
        visited += 1
        if name and name not in known:
            extras.append(name)
            known.add(name)
    return priority + extras, (start + visited) % len(universe)

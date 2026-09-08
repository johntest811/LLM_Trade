"""Deterministic, account-currency profit retention; no broker side effects."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class RetentionState:
    peak_net_usd: float = 0.0
    floor_usd: float = 0.0
    volume: float = 0.0
    reference_volume: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "RetentionState":
        if not isinstance(value, Mapping):
            return cls()
        try:
            state = cls(**{
                key: float(value.get(key, 0.0)) for key in cls.__dataclass_fields__
            })
        except (TypeError, ValueError, OverflowError):
            return cls()
        if (
            any(not math.isfinite(v) or v < 0 for v in state.to_dict().values())
            or state.floor_usd > state.peak_net_usd
            or state.volume <= 0
            or state.reference_volume < state.volume
        ):
            return cls()
        return state


def advance_retention(
    previous: RetentionState,
    *,
    net_profit_usd: float,
    volume: float,
    initial_risk_usd: float,
    policy: Any,
) -> RetentionState:
    """Ratchet only observed net peaks, preserving headroom and remaining size.

    The cash milestone is a minimum, not a fixed take-profit. Missing USD risk
    uses only the cash milestone; valid risk must also meet the R threshold.
    A partial close scales protection to the remaining position. An addition
    resets the monetary baseline; the executor still never loosens broker SL.
    """
    try:
        net, size, risk = map(float, (net_profit_usd, volume, initial_risk_usd))
        trigger = float(policy.profit_retention_trigger_usd)
        trigger_r = float(policy.profit_retention_trigger_r)
        mature_r = float(policy.profit_retention_mature_trigger_r)
        keep = float(policy.profit_retention_keep_fraction)
        mature_keep = float(policy.profit_retention_mature_keep_fraction)
        cash_floor = float(policy.profit_lock_final_floor_usd)
        headroom = float(policy.profit_retention_min_headroom_usd)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return previous
    values = (net, size, risk, trigger, trigger_r, mature_r, keep,
              mature_keep, cash_floor, headroom)
    if (
        not all(math.isfinite(v) for v in values)
        or size <= 0 or risk < 0 or trigger <= 0 or trigger_r <= 0
        or mature_r < trigger_r or not 0 < keep <= mature_keep < 1
        or cash_floor < 0 or headroom <= 0
        or trigger < cash_floor + headroom
    ):
        return previous

    peak, floor = previous.peak_net_usd, previous.floor_usd
    reference = previous.reference_volume or size
    if previous.volume > 0 and size < previous.volume - 1e-9:
        ratio = size / previous.volume
        peak *= ratio
        floor *= ratio
    elif size > previous.volume + 1e-9:
        peak, floor, reference = 0.0, 0.0, size
    peak = max(peak, net, 0.0)
    remaining_risk = risk * size / reference
    peak_r = peak / remaining_risk if remaining_risk > 0 else 0.0
    # Once armed, the cash trigger is not re-required after a partial close.
    eligible = floor > 0 or (
        peak + 1e-9 >= trigger
        and (remaining_risk <= 0 or peak_r + 1e-9 >= trigger_r)
    )
    if eligible:
        fraction = mature_keep if remaining_risk > 0 and peak_r >= mature_r else keep
        milestone = cash_floor * size / reference
        candidate = min(max(milestone, peak * fraction), peak - headroom * size / reference)
        # Never round the requested floor above available profit/headroom.
        floor = max(floor, math.floor((candidate + 1e-9) * 100) / 100)
    return RetentionState(peak, max(0.0, floor), size, reference)

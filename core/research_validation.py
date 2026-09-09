"""Read-only, cost-aware forward audit of recorded reversal-watch observations.

Run python -m core.research_validation --help. This is NOT a broker executor,
an LLM backtest, or an automatic strategy-promotion mechanism. Input M1 OHLC
must be BID prices with UTC bar-open timestamps. Costs are explicit assumptions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ValidationCosts:
    spread_price: float
    slippage_price: float
    commission_r: float

    def __post_init__(self):
        if not all(math.isfinite(value) and value >= 0 for value in asdict(self).values()):
            raise ValueError("Costs must be finite, nonnegative, and explicitly supplied")


def _utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing timestamp")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _validated_bars(bars, as_of):
    required = ["time", "open", "high", "low", "close"]
    if not set(required).issubset(bars.columns):
        raise ValueError("M1 BID CSV requires time, open, high, low, close")
    frame = bars[required].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if (frame.empty or frame["time"].isna().any() or frame["time"].duplicated().any()
            or not frame["time"].is_monotonic_increasing
            or not frame["time"].eq(frame["time"].dt.floor("min")).all()):
        raise ValueError("M1 timestamps must be unique, ascending minute opens")
    for key in required[1:]:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if not frame[key].map(lambda value: math.isfinite(value) and value > 0).all():
            raise ValueError("OHLC prices must be finite and positive")
    if ((frame.low > frame[["open", "close"]].min(axis=1)).any()
            or (frame.high < frame[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Invalid OHLC bounds")
    # Never resolve a trade using a partly formed candle, including in replay.
    return frame.loc[frame.time + pd.Timedelta(minutes=1) <= as_of].reset_index(drop=True)


def _simulate(watch, observed, frame, costs, horizon):
    signal = _utc(watch["signal_time_utc"])
    candle = _utc(watch["candle_time"])
    if signal != candle + pd.Timedelta(minutes=5) or candle != candle.floor("5min"):
        raise ValueError("Signal must follow its completed M5 candle")
    if observed < signal:
        raise ValueError("Observation precedes signal availability")
    if observed - signal > pd.Timedelta(minutes=2):
        return {"status": "STALE_SIGNAL", "net_r": None}
    start = observed.ceil("min")
    deadline = start + pd.Timedelta(minutes=horizon)
    direction = watch["direction"]
    if direction not in {"BUY", "SELL"}:
        raise ValueError("Invalid research direction")
    sign = 1 if direction == "BUY" else -1
    reference, stop, target, atr = (float(watch[key]) for key in
                                    ("reference_price", "stop_loss", "take_profit", "atr"))
    if (not all(math.isfinite(v) and v > 0 for v in (reference, stop, target, atr))
            or sign * (reference - stop) <= 0 or sign * (target - reference) <= 0):
        raise ValueError("Invalid research plan geometry")
    first = frame.time.searchsorted(start)
    if first >= len(frame) or frame.iloc[first].time != start:
        return {"status": "INCOMPLETE_DATA", "net_r": None}
    entry = float(frame.iloc[first].open) + (costs.spread_price if sign > 0 else 0) + sign * costs.slippage_price
    if (sign * (entry - stop) <= 0 or sign * (target - entry) <= 0
            or abs(entry - reference) > 0.35 * atr):
        return {"status": "UNFILLABLE_OR_DRIFT", "net_r": None}
    risk = abs(entry - stop)
    base = {"entry_time": start.isoformat(), "entry_price": entry,
            "direction": direction, "net_r": None, "exit_time": deadline.isoformat()}
    expected = start
    for row in frame.iloc[first:first + horizon].itertuples(index=False):
        if row.time != expected:
            return {**base, "status": "DATA_GAP"}
        expected += pd.Timedelta(minutes=1)
        # Sell exits execute at ASK; buy exits at BID. Slippage is adverse.
        exit_spread = costs.spread_price if sign < 0 else 0
        opened, high, low, close = (float(getattr(row, key)) + exit_spread
                                    for key in ("open", "high", "low", "close"))
        stop_hit = low <= stop if sign > 0 else high >= stop
        target_hit = high >= target if sign > 0 else low <= target
        status, exit_price = "", 0.0
        if stop_hit:
            # Unknown intrabar ordering is pessimistic, never discarded as a
            # conveniently missing loss. Stop gaps can lose more than 1 R.
            status = "BOTH_TOUCHED_STOP_FIRST" if target_hit else "STOP"
            exit_price = (min(stop, opened) if sign > 0 else max(stop, opened)) - sign * costs.slippage_price
        elif target_hit:
            status, exit_price = "TARGET", target - sign * costs.slippage_price
        elif expected == deadline:
            status, exit_price = "HORIZON", close - sign * costs.slippage_price
        if status:
            return {**base, "status": status, "exit_time": expected.isoformat(),
                    "exit_price": exit_price,
                    "net_r": sign * (exit_price - entry) / risk - costs.commission_r}
    return {**base, "status": "INCOMPLETE_DATA"}


def _stats(rows):
    values = [row["net_r"] for row in rows if row.get("net_r") is not None]
    gain = sum(max(0.0, value) for value in values)
    loss = -sum(min(0.0, value) for value in values)
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {"trades": len(values), "net_r": round(sum(values), 6),
            "expectancy_r": round(sum(values) / len(values), 6) if values else None,
            "profit_factor": round(gain / loss, 6) if loss else None,
            "max_closed_drawdown_r": round(drawdown, 6),
            "win_rate": sum(value > 0 for value in values) / len(values) if values else None}


def validate_observations(observations, bars, symbol, costs, *, as_of=None,
                          folds=3, horizon_minutes=60, min_trades_per_fold=20):
    """Chronological, version-separated simulation; never returns live approval.

    No fitting is performed. Holdout boundaries are time-based; trades crossing
    one are purged from per-fold metrics. Each version/symbol permits only one
    hypothetical open position, preventing overlapping observations inflating N.
    """
    if not 2 <= folds <= 12 or not 1 <= horizon_minutes <= 240 or min_trades_per_fold < 1:
        raise ValueError("Use 2-12 folds, 1-240 horizon minutes, and a positive sample requirement")
    as_of = _utc(as_of or datetime.now(timezone.utc))
    if not isinstance(observations, list) or any(not isinstance(row, dict) for row in observations):
        raise ValueError("Observations must be a list of scan objects")
    if "symbol" in bars and set(bars.symbol.astype(str).str.upper()) != {symbol.upper()}:
        raise ValueError("CSV symbol does not match the requested market")
    frame = _validated_bars(bars, as_of)
    unique, invalid = {}, 0
    for observation in observations:
        if str(observation.get("symbol", "")).upper() != symbol.upper():
            continue
        snapshot = observation.get("snapshot") or {}
        if not isinstance(snapshot, dict):
            invalid += 1
            continue
        watch = snapshot.get("research_watch") or {}
        if not isinstance(watch, dict):
            invalid += 1
            continue
        if not watch.get("candidate"):
            continue
        try:
            if watch.get("live_eligible") is not False:
                raise ValueError("Research-only marker required")
            observed = _utc(observation["observed_at_utc"])
            version, fingerprint = watch["strategy_version"], observation["config_fingerprint"]
            scope = observation["account_scope_id"]
            if not all(isinstance(v, str) and v for v in (scope, version, fingerprint)):
                raise ValueError("Account, version and configuration provenance required")
            key = (scope, fingerprint, version, _utc(watch["signal_time_utc"]).isoformat(), watch["direction"])
            if observed > as_of:
                continue
            if key not in unique or observed < unique[key][0]:
                unique[key] = (observed, watch)
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid += 1
    groups = {}
    for key, (observed, watch) in sorted(unique.items(), key=lambda item: item[1][0]):
        group_key = key[:3]
        group = groups.setdefault(group_key, {"rows": [], "busy_until": None})
        if group["busy_until"] is not None and observed.ceil("min") < group["busy_until"]:
            result = {"status": "OVERLAP_SKIPPED", "net_r": None}
        else:
            try:
                result = _simulate(watch, observed, frame, costs, horizon_minutes)
            except (KeyError, TypeError, ValueError, OverflowError):
                result = {"status": "INVALID_OBSERVATION", "net_r": None}
            if result.get("entry_time"):
                group["busy_until"] = _utc(result["exit_time"])
        group["rows"].append({**result, "observed_at_utc": observed.isoformat(),
                              "direction": watch.get("direction", "")})
    reports = []
    for (scope, fingerprint, version), group in groups.items():
        rows = group["rows"]
        start = _utc(rows[0]["observed_at_utc"]).floor("min")
        end = _utc(rows[-1]["observed_at_utc"]).ceil("min") + pd.Timedelta(minutes=horizon_minutes)
        boundaries = [start + (end - start) * i / folds for i in range(folds + 1)]
        samples, purged = [[] for _ in range(folds)], 0
        for row in rows:
            if row.get("net_r") is None:
                continue
            entry, exited = _utc(row["entry_time"]), _utc(row["exit_time"])
            bucket = next((i for i in range(folds) if boundaries[i] <= entry < boundaries[i+1]), None)
            if bucket is None or exited > boundaries[bucket+1]:
                purged += 1
            else:
                samples[bucket].append(row)
        fold_stats = [{"start_utc": boundaries[i].isoformat(), "end_utc": boundaries[i+1].isoformat(),
                       **_stats(sample)} for i, sample in enumerate(samples)]
        sufficient = all(item["trades"] >= min_trades_per_fold for item in fold_stats)
        positive = all((item["expectancy_r"] or 0) > 0 for item in fold_stats)
        unresolved = sum(row["status"] in {"DATA_GAP", "INCOMPLETE_DATA", "INVALID_OBSERVATION"} for row in rows)
        reports.append({"symbol": symbol.upper(), "config_fingerprint": fingerprint, "account_scope_id": scope,
                        "strategy_version": version, "observations": len(rows),
                        "status_counts": dict(Counter(row["status"] for row in rows)),
                        "overall": _stats(rows), "folds": fold_stats, "boundary_purged": purged,
                        "unresolved_observations": unresolved,
                        "by_direction": {side: _stats([row for row in rows if row["direction"] == side]) for side in ("BUY", "SELL")},
                        "by_year": {year: _stats([row for row in rows if row["observed_at_utc"][:4] == year])
                                    for year in sorted({row["observed_at_utc"][:4] for row in rows})},
                        "evidence": "INSUFFICIENT DATA" if not sufficient else (
                            "INCOMPLETE COVERAGE" if unresolved else
                            "POSITIVE SIMULATION ONLY" if positive else "NEGATIVE/INCONCLUSIVE"),
                        "results": rows})
    return {"schema_version": 1, "automatic_live_promotion": False,
            "method": "Recorded forward observations; chronological folds; no parameter fitting",
            "as_of_utc": as_of.isoformat(), "costs": asdict(costs),
            "horizon_minutes": horizon_minutes, "min_trades_per_fold": min_trades_per_fold,
            "invalid_observations": invalid, "groups": reports,
            "limitations": ["Hypothetical M1 BID fills, not broker executions or full live-strategy P/L.",
                            "Constant spread/slippage and commission-R assumptions; no margin, swaps or portfolio simulation.",
                            "Missing bars are unresolved; both-touched bars assume stop first.",
                            "Selective scan coverage is not a complete historical-market backtest.",
                            "Positive results require independent demo review; never authorize live orders."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, required=True, help="Saved /api/scans JSON (one account)")
    parser.add_argument("--bars", type=Path, required=True, help="M1 BID CSV with UTC time,open,high,low,close")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--spread-price", type=float, required=True)
    parser.add_argument("--slippage-price", type=float, required=True)
    parser.add_argument("--commission-r", type=float, required=True, help="Round-trip commission / initial price-risk value")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    args = parser.parse_args()
    try:
        raw_scans, raw_bars = args.scans.read_bytes(), args.bars.read_bytes()
        payload = json.loads(raw_scans.decode("utf-8-sig"))
        observations = payload["observations"] if isinstance(payload, dict) else payload
        costs = ValidationCosts(args.spread_price, args.slippage_price, args.commission_r)
        bars = pd.read_csv(io.BytesIO(raw_bars))
        now = datetime.now(timezone.utc)
        report = {"input_sha256": {"scans": hashlib.sha256(raw_scans).hexdigest(),
                                  "bars": hashlib.sha256(raw_bars).hexdigest()}}
        for label, multiplier in (("baseline", 1), ("double_cost_stress", 2)):
            scenario = ValidationCosts(*(value * multiplier for value in asdict(costs).values()))
            report[label] = validate_observations(observations, bars, args.symbol, scenario,
                                                  as_of=now, folds=args.folds,
                                                  horizon_minutes=args.horizon_minutes)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.exit(2, f"Validation input error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

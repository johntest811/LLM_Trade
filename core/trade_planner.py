"""Deterministic broker-valid entry planning and capital-fit diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, Optional

from mt5.safe_api import mt5

from app_config.settings import settings
from risk.budget import risk_capital, entry_risk_budget
from core.live_reversal import live_reversal_plan, live_reversal_risk_percent
from mt5.timebase import broker_tick_age_seconds
from risk.execution_costs import estimate_execution_risk
from risk.instruments import (
    pip_size,
    spread_metrics,
    validate_spread,
    validate_symbol_trade_mode,
)


@dataclass(frozen=True)
class TradePlan:
    valid: bool
    action: str
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    risk_distance: float = 0.0
    planned_rr: float = 0.0
    source: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeterministicTradePlanner:
    """Convert a direction into levels without trusting generated prices."""

    @staticmethod
    def _execution_adjusted_take_profit(
        symbol: str,
        action: str,
        info: Any,
        entry: float,
        stop_loss: float,
        take_profit: float,
        risk_distance: float,
        *,
        allow_extension: bool = True,
    ) -> tuple[Optional[float], float, bool, str]:
        """Make a bounded baseline-target adjustment for execution reserves."""
        lot = float(getattr(info, "volume_min", 0.0) or 0.0)
        if lot <= 0:
            return None, 0.0, False, "Broker minimum volume is unavailable"
        estimate = estimate_execution_risk(
            symbol, action, lot, entry, stop_loss, info
        )
        if estimate is None or estimate.total_risk_usd <= 0:
            return None, 0.0, False, "Broker could not calculate execution-adjusted plan risk"

        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        direction = 1.0 if action == "BUY" else -1.0
        desired_net_rr = settings.min_risk_reward_ratio + max(
            0.0, settings.plan_net_rr_buffer
        )
        def gross_reward(distance: float) -> Optional[float]:
            candidate = estimate.worst_entry + direction * distance
            if candidate <= 0 or not math.isfinite(candidate):
                return None
            result = mt5.order_calc_profit(
                order_type, symbol, lot, estimate.worst_entry, candidate
            )
            if result is None or not math.isfinite(float(result)):
                return None
            return max(0.0, float(result))

        initial_distance = abs(take_profit - estimate.worst_entry)
        initial_gross = gross_reward(initial_distance)
        if initial_gross is None:
            return None, 0.0, False, "Broker could not calculate execution-adjusted target reward"
        initial_net = max(0.0, initial_gross - estimate.configured_cost_usd)
        initial_rr = initial_net / estimate.total_risk_usd
        if initial_rr + 1e-9 < desired_net_rr:
            max_extension_r = max(
                0.0,
                float(
                    getattr(
                        settings,
                        "plan_max_cost_target_extension_r",
                        0.0,
                    )
                ),
            )
            if allow_extension and max_extension_r > 0:
                required_gross = (
                    desired_net_rr * estimate.total_risk_usd
                    + estimate.configured_cost_usd
                )
                maximum_distance = (
                    initial_distance + risk_distance * max_extension_r
                )
                maximum_gross = gross_reward(maximum_distance)
                if (
                    maximum_gross is not None
                    and maximum_gross + 1e-9 >= required_gross
                ):
                    low, high = initial_distance, maximum_distance
                    for _ in range(48):
                        middle = (low + high) / 2.0
                        middle_gross = gross_reward(middle)
                        if (
                            middle_gross is not None
                            and middle_gross >= required_gross
                        ):
                            high = middle
                        else:
                            low = middle
                    point = float(getattr(info, "point", 0.0) or 0.0)
                    digits = int(getattr(info, "digits", 5) or 5)
                    adjusted = estimate.worst_entry + direction * high
                    if point > 0:
                        if action == "BUY":
                            adjusted = math.ceil(
                                adjusted / point - 1e-10
                            ) * point
                        else:
                            adjusted = math.floor(
                                adjusted / point + 1e-10
                            ) * point
                    adjusted = round(adjusted, digits)
                    adjusted_distance = abs(
                        adjusted - estimate.worst_entry
                    )
                    adjusted_gross = gross_reward(adjusted_distance)
                    if adjusted_gross is not None:
                        adjusted_net = max(
                            0.0,
                            adjusted_gross - estimate.configured_cost_usd,
                        )
                        adjusted_rr = (
                            adjusted_net / estimate.total_risk_usd
                        )
                        if adjusted_rr + 1e-9 >= desired_net_rr:
                            return adjusted, adjusted_rr, True, ""
            return (
                None,
                initial_rr,
                False,
                (
                    f"Planned target provides only {initial_rr:.2f} "
                    f"net R:R; required {desired_net_rr:.2f}. Target will not "
                    "be extended beyond the configured execution-cost limit"
                ),
            )
        return take_profit, initial_rr, False, ""

    @staticmethod
    def _technical_target(
        action: str,
        entry: float,
        atr: float,
        structure: Dict[str, Any],
        minimum_distance: float = 0.0,
    ) -> tuple[Optional[float], str]:
        """Return the nearest usable broker-derived objective ahead of entry."""
        candidates: list[float] = []
        buffer_distance = atr * max(
            0.0, float(getattr(settings, "plan_target_buffer_atr", 0.10))
        )
        minimum_raw_distance = max(0.0, minimum_distance) + buffer_distance

        def add(value: Any) -> None:
            try:
                level = float(value)
            except (TypeError, ValueError, OverflowError):
                return
            if not math.isfinite(level) or level <= 0:
                return
            distance = (
                level - entry if action == "BUY" else entry - level
            )
            if distance + 1e-12 >= minimum_raw_distance:
                candidates.append(level)

        add(
            structure.get("resistance")
            if action == "BUY"
            else structure.get("support")
        )
        range_setup = structure.get("range_reversion", {}) or {}
        if (
            isinstance(range_setup, dict)
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper() == action
        ):
            add(range_setup.get("target_price"))
        zone_key = "supply_zones" if action == "BUY" else "demand_zones"
        for zone in structure.get(zone_key, []) or []:
            if not isinstance(zone, dict):
                continue
            add(zone.get("bottom") if action == "BUY" else zone.get("top"))
        liquidity = structure.get("liquidity_zones", {}) or {}
        liquidity_key = (
            "buy_side_liquidity_levels"
            if action == "BUY"
            else "sell_side_liquidity_levels"
        )
        for level in liquidity.get(liquidity_key, []) or []:
            add(level)

        if not candidates:
            return (
                None,
                "No opposing structure or liquidity objective is far enough "
                "ahead to serve as the planned target",
            )

        nearest = min(candidates) if action == "BUY" else max(candidates)
        buffered = (
            nearest - buffer_distance
            if action == "BUY"
            else nearest + buffer_distance
        )
        if (action == "BUY" and buffered <= entry) or (
            action == "SELL" and buffered >= entry
        ):
            buffered = nearest
        return buffered, ""

    @staticmethod
    def build(
        symbol: str,
        action: str,
        analysis: Dict[str, Any],
        *,
        info: Any = None,
        tick: Any = None,
    ) -> TradePlan:
        action = str(action).upper()
        if action not in {"BUY", "SELL"}:
            return TradePlan(False, action, reason="An entry plan requires BUY or SELL")
        info = info or mt5.symbol_info(symbol)
        tick = tick or mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return TradePlan(False, action, reason="Broker symbol metadata or quote is unavailable")
        trade_allowed, trade_reason = validate_symbol_trade_mode(info, action)
        if not trade_allowed:
            return TradePlan(False, action, reason=trade_reason)
        tick_time = float(getattr(tick, "time", time.time()) or 0.0)
        tick_age = broker_tick_age_seconds(tick_time, symbol=symbol)
        if not math.isfinite(tick_age) or tick_age > settings.max_tick_age_seconds:
            return TradePlan(False, action, reason="Broker quote is stale")

        indicators = (analysis or {}).get("indicators", {})
        atr = float(indicators.get("atr_14") or 0.0)
        if atr <= 0:
            atr_pips = float(indicators.get("atr_14_pips") or 0.0)
            atr = atr_pips * pip_size(info)
        if atr <= 0:
            return TradePlan(False, action, reason="ATR is unavailable")

        entry = float(tick.ask if action == "BUY" else tick.bid)
        if entry <= 0:
            return TradePlan(False, action, reason="Live entry quote is invalid")
        point = float(getattr(info, "point", 0.0) or 0.0)
        broker_points = max(
            float(getattr(info, "trade_stops_level", 0.0) or 0.0),
            float(getattr(info, "trade_freeze_level", 0.0) or 0.0),
        )
        broker_min = broker_points * point
        spread_abs = max(0.0, float(tick.ask) - float(tick.bid))
        risk_distance = max(
            atr * settings.plan_stop_atr,
            broker_min * 1.20,
            spread_abs * (100.0 / max(settings.max_spread_to_stop_pct, 1e-9)),
        )

        structure = (analysis or {}).get("market_structure", {})
        reversal_setup = live_reversal_plan(analysis, action, config=settings)
        reversal_matches = bool(reversal_setup)
        if reversal_matches:
            try:
                invalidation = float(reversal_setup["invalidation_price"])
                reversal_target = float(reversal_setup["target_price"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return TradePlan(False, action, reason="Validated live reversal levels are unavailable")
            if (not all(math.isfinite(value) and value > 0 for value in (invalidation, reversal_target))
                    or (action == "BUY" and not invalidation < entry < reversal_target)
                    or (action == "SELL" and not reversal_target < entry < invalidation)):
                return TradePlan(False, action, reason="Live quote has left the validated reversal entry zone")
            # This separate strategy uses its two-candle invalidation, not the
            # ordinary continuation's ATR stop. Never shrink broker/spread floors.
            risk_distance = max(abs(entry - invalidation), 0.5 * atr,
                                broker_min * 1.20,
                                spread_abs * (100.0 / max(settings.max_spread_to_stop_pct, 1e-9)))
        range_setup = structure.get("range_reversion", {}) or {}
        range_matches = bool(
            not reversal_matches
            and isinstance(range_setup, dict)
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper() == action
        )
        if range_matches and not reversal_matches:
            try:
                invalidation = float(range_setup.get("invalidation_price"))
                range_target = float(range_setup.get("target_price"))
            except (TypeError, ValueError, OverflowError):
                return TradePlan(False, action, reason="Validated range levels are unavailable")
            if not all(math.isfinite(value) and value > 0 for value in (invalidation, range_target)):
                return TradePlan(False, action, reason="Validated range levels are invalid")
            if action == "BUY" and not invalidation < entry < range_target:
                return TradePlan(False, action, reason="Live BUY quote has left the validated range entry zone")
            if action == "SELL" and not range_target < entry < invalidation:
                return TradePlan(False, action, reason="Live SELL quote has left the validated range entry zone")
            risk_distance = max(risk_distance, abs(entry - invalidation))
        support = float(structure.get("support") or 0.0)
        resistance = float(structure.get("resistance") or 0.0)
        source = f"{settings.plan_stop_atr:g} ATR"
        if reversal_matches:
            source = "validated local reversal invalidation (experimental)"
        if range_matches:
            source = "validated range invalidation"
        if not reversal_matches and action == "BUY" and 0 < support < entry:
            structural_distance = entry - (support - 0.20 * atr)
            if risk_distance < structural_distance <= 3.0 * atr:
                risk_distance = structural_distance
                source = "ATR + support invalidation"
        elif not reversal_matches and action == "SELL" and resistance > entry:
            structural_distance = (resistance + 0.20 * atr) - entry
            if risk_distance < structural_distance <= 3.0 * atr:
                risk_distance = structural_distance
                source = "ATR + resistance invalidation"

        rr = max(settings.min_risk_reward_ratio, settings.plan_target_rr)
        if action == "BUY":
            stop_loss = entry - risk_distance
            take_profit = entry + risk_distance * rr
        else:
            stop_loss = entry + risk_distance
            take_profit = entry - risk_distance * rr
        baseline_take_profit = take_profit
        if reversal_matches:
            take_profit = min(take_profit, reversal_target) if action == "BUY" else max(take_profit, reversal_target)

        technical_target, technical_error = DeterministicTradePlanner._technical_target(
            action,
            entry,
            atr,
            structure,
            # Discover the nearest real objective first. Its attainability is
            # validated below; filtering it out here would let a farther ATR
            # target be manufactured through known opposing structure.
            minimum_distance=0.0,
        )
        require_technical_target = bool(
            getattr(settings, "require_technical_target", False)
        )
        if technical_target is None and require_technical_target:
            return TradePlan(False, action, reason=technical_error)
        target_capped = reversal_matches
        # Optional rolling structure is advisory.  The entry gate already
        # rejects orders opened too close to opposing structure, and the final
        # risk manager independently validates net R:R.  Capping every BOS or
        # breakout plan to the nearest noisy rolling level duplicated those
        # protections and could collapse a broker-valid target onto the entry.
        # A validated range-reversion objective remains a hard cap because it
        # is intrinsic to that strategy, as does an explicitly required
        # technical target.
        constrain_to_technical_target = bool(
            technical_target is not None
            and (require_technical_target or range_matches or reversal_matches)
        )
        if constrain_to_technical_target:
            capped_take_profit = (
                min(take_profit, technical_target)
                if action == "BUY"
                else max(take_profit, technical_target)
            )
            target_capped = target_capped or not math.isclose(
                capped_take_profit,
                take_profit,
                rel_tol=0.0,
                abs_tol=max(point, 1e-12),
            )
            take_profit = capped_take_profit

        digits = int(getattr(info, "digits", 5) or 5)
        entry = round(entry, digits)
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)
        baseline_take_profit = round(baseline_take_profit, digits)
        if min(entry, stop_loss, take_profit) <= 0:
            return TradePlan(False, action, reason="Calculated price level is invalid")
        if action == "BUY" and not stop_loss < entry < take_profit:
            return TradePlan(False, action, reason="Calculated BUY levels are invalid")
        if action == "SELL" and not take_profit < entry < stop_loss:
            return TradePlan(False, action, reason="Calculated SELL levels are invalid")

        ok, reason, _ = validate_spread(
            symbol, info, tick, entry=entry, stop_loss=stop_loss
        )
        if not ok:
            return TradePlan(False, action, reason=reason)
        actual_risk = abs(entry - stop_loss)
        adjusted_tp, net_rr, target_adjusted, adjustment_error = (
            DeterministicTradePlanner._execution_adjusted_take_profit(
                symbol,
                action,
                info,
                entry,
                stop_loss,
                take_profit,
                actual_risk,
                allow_extension=not target_capped,
            )
        )
        if adjusted_tp is None:
            if target_capped:
                adjustment_error = (
                    f"Nearest technical objective cannot support the required "
                    f"net R:R ({adjustment_error}). Wait for a better entry "
                    "instead of targeting through opposing structure"
                )
            return TradePlan(
                False,
                action,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
                risk_distance=actual_risk,
                planned_rr=net_rr,
                source=source,
                reason=adjustment_error,
            )
        take_profit = adjusted_tp
        if action == "BUY" and not stop_loss < entry < take_profit:
            return TradePlan(False, action, reason="Execution-adjusted BUY target is invalid")
        if action == "SELL" and not take_profit < entry < stop_loss:
            return TradePlan(False, action, reason="Execution-adjusted SELL target is invalid")
        if target_adjusted:
            source += " + execution-adjusted target"
        elif target_capped:
            source += " + technical target cap"
        return TradePlan(
            True,
            action,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            risk_distance=actual_risk,
            planned_rr=net_rr,
            source=source,
        )

    @classmethod
    def assess_capital_fit(
        cls,
        symbol: str,
        analysis: Dict[str, Any],
        account: Dict[str, Any],
        *, daily_loss: float = 0.0,
    ) -> Dict[str, Any]:
        """Assess both directions at broker minimum volume without sending."""
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        capital = risk_capital(account)
        balance = float(account["balance"]) if capital > 0 else 0.0
        equity = float(account["equity"]) if capital > 0 else 0.0
        try:
            current_margin = float(account.get("margin", 0.0))
            free_margin = float(account["margin_free"])
        except (KeyError, TypeError, ValueError, OverflowError):
            current_margin = free_margin = math.nan
        risk_budget = entry_risk_budget(account, settings, daily_loss=daily_loss)
        result: Dict[str, Any] = {
            "symbol": symbol,
            "status": "DATA ERROR",
            "capital_fit": False,
            "reason": "Broker metadata is unavailable",
            "risk_budget_usd": round(risk_budget, 4),
            "risk_capital": capital,
            "risk_basis": "MIN_BALANCE_EQUITY",
            "min_volume": 0.0,
            "min_stop_risk_usd": 0.0,
            "min_price_stop_risk_usd": 0.0,
            "min_slippage_reserve_usd": 0.0,
            "min_execution_cost_usd": 0.0,
            "min_stop_risk_pct": 0.0,
            "min_margin_usd": 0.0,
            "projected_margin_pct": 0.0,
            "spread_value": 0.0,
            "spread_unit": "",
            "broker_open": False,
            "broker_trade_mode": None,
            "quote_age_seconds": None,
            "directions": {},
        }
        if info is None or tick is None or balance <= 0 or equity <= 0:
            return result
        if not all(math.isfinite(v) and v >= 0 for v in (current_margin, free_margin)):
            result["reason"] = "Valid finite account margin is required"
            return result

        raw_tick_time = getattr(tick, "time", None)
        tick_age = broker_tick_age_seconds(
            time.time() if raw_tick_time is None else float(raw_tick_time or 0.0),
            symbol=symbol,
        )
        trade_mode = int(getattr(info, "trade_mode", 4))
        result.update(
            broker_trade_mode=trade_mode,
            quote_age_seconds=(
                round(tick_age, 3) if math.isfinite(tick_age) else None
            ),
        )
        spread = spread_metrics(symbol, info, tick)
        result.update(
            min_volume=float(info.volume_min),
            spread_value=round(float(spread["value"]), 2),
            spread_unit=str(spread["unit"]),
            asset_class=str(spread["asset_class"]),
        )
        candidates = []
        diagnostics = []
        for action in ("BUY", "SELL"):
            direction_budget = risk_budget
            if live_reversal_plan(analysis, action, config=settings):
                direction_budget = entry_risk_budget(account, settings,
                    risk_percent=live_reversal_risk_percent(settings), daily_loss=daily_loss)
            plan = cls.build(symbol, action, analysis, info=info, tick=tick)
            direction: Dict[str, Any] = {"plan": plan.to_dict(), "capital_fit": False,
                                         "risk_budget_usd": round(direction_budget, 4)}
            if plan.entry > 0 and plan.stop_loss > 0:
                order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
                estimate = estimate_execution_risk(
                    symbol,
                    action,
                    float(info.volume_min),
                    plan.entry,
                    plan.stop_loss,
                    info,
                )
                risk = estimate.total_risk_usd if estimate is not None else 0.0
                margin = float(mt5.order_calc_margin(
                    order_type, symbol, float(info.volume_min), plan.entry
                ) or 0.0)
                projected_margin_pct = (current_margin + margin) / equity * 100.0
                fit = (
                    plan.valid
                    and risk > 0
                    and risk <= direction_budget + 1e-9
                    and margin > 0
                    and projected_margin_pct <= settings.max_margin_usage_pct + 1e-9
                    and free_margin >= margin * 1.05
                )
                direction.update(
                    capital_fit=fit,
                    risk_usd=round(risk, 4),
                    price_stop_risk_usd=(
                        round(estimate.stop_risk_usd, 4) if estimate is not None else 0.0
                    ),
                    slippage_reserve_usd=(
                        round(estimate.slippage_reserve_usd, 4) if estimate is not None else 0.0
                    ),
                    execution_cost_usd=(
                        round(estimate.configured_cost_usd, 4) if estimate is not None else 0.0
                    ),
                    risk_pct=round(risk / balance * 100.0, 3),
                    margin_usd=round(margin, 4),
                    projected_margin_pct=round(projected_margin_pct, 2),
                )
                diagnostics.append(direction)
                if plan.valid:
                    candidates.append(direction)
            result["directions"][action] = direction

        diagnostic_rows = [
            item for item in diagnostics if item.get("risk_usd", 0.0) > 0
        ]
        if diagnostic_rows:
            lowest_risk = min(
                diagnostic_rows, key=lambda item: item["risk_usd"]
            )
            lowest_margin = min(
                diagnostic_rows, key=lambda item: item["margin_usd"]
            )
            result.update(
                min_stop_risk_usd=lowest_risk["risk_usd"],
                min_price_stop_risk_usd=lowest_risk["price_stop_risk_usd"],
                min_slippage_reserve_usd=lowest_risk["slippage_reserve_usd"],
                min_execution_cost_usd=lowest_risk["execution_cost_usd"],
                min_stop_risk_pct=lowest_risk["risk_pct"],
                min_margin_usd=lowest_margin["margin_usd"],
                projected_margin_pct=lowest_margin["projected_margin_pct"],
            )
        valid = [item for item in candidates if item.get("risk_usd", 0.0) > 0]
        fit_any = any(item.get("capital_fit") for item in candidates)
        result["capital_fit"] = fit_any
        result["broker_open"] = bool(
            math.isfinite(tick_age)
            and tick_age <= settings.max_tick_age_seconds
            and any(
                validate_symbol_trade_mode(info, action)[0]
                for action in ("BUY", "SELL")
            )
        )
        if fit_any:
            result["status"] = "CAPITAL FIT"
            result["reason"] = "At least one direction fits risk and projected margin limits"
        elif valid and all(item["risk_usd"] > item["risk_budget_usd"] + 1e-9 for item in valid):
            result["status"] = "UNAFFORDABLE"
            result["reason"] = (
                f"Minimum-volume technical stop risks ${result['min_stop_risk_usd']:.2f}; "
                "exceeds its direction-specific risk budget"
            )
        elif valid:
            result["status"] = "MARGIN BLOCK"
            result["reason"] = "Broker minimum volume exceeds projected margin or free-margin limits"
        else:
            plan_reasons = list(dict.fromkeys(
                str(direction.get("plan", {}).get("reason", "")).strip()
                for direction in result["directions"].values()
                if str(direction.get("plan", {}).get("reason", "")).strip()
            ))
            if not result["broker_open"]:
                # A stale tick does not prove a closed trading session,
                # particularly for the broker's extended-hours instruments.
                result["status"] = (
                    "TRADING DISABLED" if trade_mode == 0 else
                    "CLOSE ONLY" if trade_mode == 3 else
                    "QUOTE STALE" if tick_age > settings.max_tick_age_seconds else
                    "QUOTE UNAVAILABLE"
                )
                result["reason"] = (
                    plan_reasons[0]
                    if plan_reasons
                    else "Broker is not publishing a fresh tradable quote"
                )
            else:
                result["status"] = "SETUP BLOCK"
                result["reason"] = (
                    "; ".join(plan_reasons[:2])
                    if plan_reasons
                    else "No broker-valid ATR plan passed quote and spread checks"
                )
        return result

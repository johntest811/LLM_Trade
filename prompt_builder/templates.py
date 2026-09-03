"""Compact, risk-neutral prompts for the local decision model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

class PromptTemplateBuilder:
    @staticmethod
    def get_compact_schema() -> str:
        return (
            '{"action":"BUY|SELL|HOLD|CLOSE","confidence":0.0,'
            '"ticket_to_close":null,'
            '"evidence_ids":[],'
            '"reasoning":"brief evidence","trade_management":"brief invalidation"}'
        )

    @classmethod
    def build_system_text(cls, base_instruction: str = "", fast_exit_review: bool = False) -> str:
        common = (
            "You are one component in a risk-controlled trading system. "
            "Use only the supplied market data. Never invent news, prices or indicators. "
            "Do not infer indicator history, slope, prior overbought/oversold states, or "
            "support/resistance that is not explicitly supplied. "
            "For BUY or SELL, copy supplied evidence IDs verbatim; never rename, "
            "abbreviate, or reorder their tokens. "
            "The deterministic risk engine, not you, controls position size. "
            "HOLD is the correct decision whenever evidence is mixed or the setup is marginal. "
            "Return only the requested JSON object."
        )
        if fast_exit_review:
            rules = (
                "Review the single open position. Choose CLOSE only for its exact ticket when "
                "a supplied completed M1 or M5 BOS, CHoCH, or breakout invalidates "
                "the original directional thesis. Treat the live quote as execution "
                "context, not as an invented candle or indicator. "
                "Do not close merely because a trade is temporarily negative, and do not promise profit. "
                "This is exit-only: never return BUY or SELL."
            )
        else:
            rules = (
                "Choose BUY or SELL only when multi-timeframe direction, momentum and structure agree. "
                "A supplied M5_RETEST ID is a deterministic, fresh pullback-resumption "
                "trigger; it is not permission to invent a retest. "
                "A supplied M5_RANGE ID is a deterministic low-ADX outer-band reversal; "
                "choose only its encoded direction and never invent range eligibility. "
                "Deterministic code calculates broker-valid entry, stop, target and volume after your "
                "directional decision. Never invent price levels. Trade management must describe only "
                "which supplied directional evidence would invalidate the setup. Otherwise choose HOLD."
            )
        return f"{common}\n{rules}\nSchema: {cls.get_compact_schema()}"

    @classmethod
    def build_user_prompt(
        cls,
        symbol: str,
        timeframe: str,
        macro_trend: str,
        market_state: Dict[str, Any],
        m15_state: Optional[Dict[str, Any]],
        pivot_points: Optional[Dict[str, Any]],
        price_action_data: Optional[Dict[str, Any]],
        account_info: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        fast_exit_review: bool = False,
    ) -> str:
        if not fast_exit_review:
            return f"{symbol} {timeframe} market state: {market_state}; macro trend={macro_trend}."
        position = open_positions[0] if open_positions else {}
        side = "BUY" if position.get("type") == 0 else "SELL"
        return (
            f"EXIT REVIEW {symbol}: ticket={position.get('ticket')} side={side} "
            f"open={position.get('price_open')} current={position.get('price_current')} "
            f"SL={position.get('sl')} TP={position.get('tp')} "
            f"PnL_USD={position.get('profit', 0.0)} PnL_pips={position.get('profit_pips', 0.0)}; "
            f"trend={macro_trend}; indicators={market_state}."
        )

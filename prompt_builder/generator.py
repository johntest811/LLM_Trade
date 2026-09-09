import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from app_config.settings import settings
from core.evidence import build_evidence_ids, permitted_entry_actions
from prompt_builder.templates import PromptTemplateBuilder

logger = logging.getLogger("TradingSystem.PromptGenerator")

class PromptGenerator:
    """
    Translates raw and structured trading states into highly compressed prompts.
    Entry prompts contain market evidence only: account balance, risk budget and
    prior P/L belong to deterministic controls and must not bias direction.
    """
    
    @staticmethod
    def compress_history(trade_history: List[Dict[str, Any]], symbol: str) -> str:
        """
        Compresses a list of past trades into a compact text summary.
        Saves hundreds of tokens by avoiding raw JSON lists.
        """
        if not trade_history:
            return "No recent trades recorded."

        symbol_trades = [t for t in trade_history if t.get("symbol") == symbol]
        if not symbol_trades:
            return f"No recent trades recorded for {symbol}."

        # Filter closures to calculate metrics
        closures = [
            t for t in symbol_trades
            if t.get("action") == "CLOSE" or "net_profit" in t
        ]
        wins = [t for t in closures if t.get("net_profit", t.get("profit", 0.0)) > 0.0]
        losses = [t for t in closures if t.get("net_profit", t.get("profit", 0.0)) <= 0.0]

        total_profit = sum(t.get("net_profit", t.get("profit", 0.0)) for t in closures)
        win_rate = (len(wins) / len(closures)) * 100.0 if closures else 0.0

        summary = (
            f"History: {len(closures)} trades closed | Win Rate: {win_rate:.1f}% ({len(wins)}W, {len(losses)}L) | "
            f"Cumulative Net Profit: ${total_profit:.2f}. "
        )

        # Highlight the last trade outcome
        if closures:
            last_trade = closures[0]
            ticket = last_trade.get("position_id", last_trade.get("ticket"))
            
            # Find matching entry to check direction
            direction = last_trade.get("direction", "UNKNOWN")
            for t in reversed(symbol_trades):
                if t.get("ticket") == ticket and t.get("action") in ["BUY", "SELL"]:
                    direction = t.get("action")
                    break

            profit_val = last_trade.get("net_profit", last_trade.get("profit", 0.0))
            status = "WIN" if profit_val > 0.0 else "LOSS"
            summary += f"Last trade ({direction} ticket {ticket}) ended in a {status} with ${profit_val:.2f} profit."

        return summary

    @staticmethod
    def compress_news(calendar_events: Optional[List[Dict[str, Any]]], symbol: str) -> str:
        """
        Filters and compresses economic calendar events.
        Only shows high-impact news matching currency pair within a 2-hour window.
        """
        if calendar_events is None:
            return "Unavailable (not evidence that the calendar is clear)"
        if not calendar_events:
            return "None (No high-impact news events scheduled)"

        # Determine currencies involved
        base_curr = symbol[:3].upper()
        quote_curr = symbol[3:].upper()

        now = datetime.now()
        event_strings = []

        for event in calendar_events:
            impact = event.get("impact", "LOW").upper()
            currency = event.get("currency", "").upper()
            
            # Filter for high/medium impact events affecting the traded pair
            if impact in ["HIGH", "MEDIUM"] and currency in [base_curr, quote_curr]:
                event_time_str = event.get("time", "")
                try:
                    event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                    # Check if event is within 2 hours
                    if abs((event_time - now).total_seconds()) <= 7200:
                        event_strings.append(
                            f"[{impact} IMPACT] {event_time.strftime('%H:%M')} - {currency} {event.get('event_name')}"
                        )
                except Exception:
                    continue

        if not event_strings:
            return "None (No high-impact currency news in the next 2 hours)"

        return " | ".join(event_strings)

    @classmethod
    def generate(
        cls,
        symbol: str,
        timeframe: str,
        analysis_data: Dict[str, Any],
        m15_analysis: Optional[Dict[str, Any]],
        h1_analysis: Optional[Dict[str, Any]],
        h4_analysis: Optional[Dict[str, Any]],
        account_info: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        trade_history: List[Dict[str, Any]],
        calendar_events: Optional[List[Dict[str, Any]]],
        fast_exit_review: bool = False,
        m1_analysis: Optional[Dict[str, Any]] = None,
        live_tick: Optional[Dict[str, Any]] = None,
        forex_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Builds the final system and user prompts, fully optimized for token usage and model reasoning.
        """
        # 1. Base System instructions
        base_instr = "Evaluate a trend/structure setup conservatively from supplied data only."
        system_prompt = PromptTemplateBuilder.build_system_text(base_instr, fast_exit_review=fast_exit_review)

        # 2. Dynamic Exit Prompt
        if fast_exit_review:
            # Re-use the templates logic for exit review as it is already highly optimized
            exit_analysis = m1_analysis or analysis_data
            user_prompt = PromptTemplateBuilder.build_user_prompt(
                symbol=symbol,
                timeframe=timeframe,
                macro_trend=exit_analysis.get("market_structure", {}).get(
                    "trend_state",
                    exit_analysis.get("market_structure", {}).get(
                        "trend", "NEUTRAL"
                    ),
                ),
                market_state=exit_analysis.get("indicators", {}),
                m15_state=None,
                pivot_points=None,
                price_action_data=None,
                account_info=account_info,
                open_positions=open_positions,
                fast_exit_review=True
            )
            exit_evidence_ids = build_evidence_ids(
                {
                    "M1": m1_analysis or {},
                    "M5": analysis_data,
                    "M15": m15_analysis or {},
                    "H1": h1_analysis or {},
                    "H4": h4_analysis or {},
                }
            )
            position = open_positions[0] if open_positions else {}
            position_side = "BUY" if position.get("type") == 0 else "SELL"
            opposing = "BEARISH" if position_side == "BUY" else "BULLISH"
            live_quote = (
                f" Live broker quote: bid={live_tick.get('bid')} "
                f"ask={live_tick.get('ask')} time={live_tick.get('time')}."
                if live_tick
                else ""
            )
            user_prompt += (
                live_quote
                + f" Allowed Evidence IDs: {list(exit_evidence_ids)}. "
                f"Choose CLOSE only with a supplied M1 or M5 {opposing} BOS, "
                "CHoCH, or breakout ID. M1 is the fast reversal lane; M5 is "
                "the stronger confirmation. Otherwise HOLD with evidence_ids=[]."
            )
            return system_prompt, user_prompt


        # 3. Compact Market Summary & Multi-Timeframe Trends
        curr_price = analysis_data.get("indicators", {}).get("current_price", 0.0)
        def trend_fields(analysis):
            structure = (analysis or {}).get("market_structure", {})
            regime = structure.get("regime_trend", structure.get("trend", "NEUTRAL"))
            fallback_state = "NEUTRAL" if regime == "NEUTRAL" else f"CONFIRMED_{regime}"
            state = structure.get("trend_state", fallback_state)
            direction = structure.get("trend_state_direction", regime)
            return regime, state, direction

        m5_trend, m5_state, m5_direction = trend_fields(analysis_data)
        m15_trend, m15_state, m15_direction = trend_fields(m15_analysis)
        h1_trend, h1_state, h1_direction = trend_fields(h1_analysis)
        h4_trend, h4_state, h4_direction = trend_fields(h4_analysis)

        m5_adx = analysis_data.get("indicators", {}).get("adx_14", 0.0) or 0.0
        m15_adx = m15_analysis.get("indicators", {}).get("adx_14", 0.0) if m15_analysis else 0.0
        h1_adx = h1_analysis.get("indicators", {}).get("adx_14", 0.0) if h1_analysis else 0.0
        h4_adx = h4_analysis.get("indicators", {}).get("adx_14", 0.0) if h4_analysis else 0.0

        # Calculate a simple trend/momentum bias to guide the model
        ind = analysis_data.get("indicators", {})
        rsi = ind.get("rsi_14", 50.0) or 50.0
        macd = ind.get("macd", {}).get("diff", 0.0) or 0.0
        ema9 = ind.get("ema_9") or curr_price
        ema21 = ind.get("ema_21") or curr_price

        # Count bullish vs bearish alignment
        bull_counts = sum([m5_direction == "BULLISH", m15_direction == "BULLISH", h1_direction == "BULLISH", h4_direction == "BULLISH"])
        bear_counts = sum([m5_direction == "BEARISH", m15_direction == "BEARISH", h1_direction == "BEARISH", h4_direction == "BEARISH"])

        bias_summary = "NEUTRAL"
        if bull_counts >= 3 and rsi > 48 and macd >= -0.01:
            bias_summary = f"STRONG BULLISH (Trend Alignment: {bull_counts}/4 Bullish, RSI: {rsi:.1f}, MACD positive)"
        elif bear_counts >= 3 and rsi < 52 and macd <= 0.01:
            bias_summary = f"STRONG BEARISH (Trend Alignment: {bear_counts}/4 Bearish, RSI: {rsi:.1f}, MACD negative)"
        elif ema9 > ema21 and rsi > 52:
            bias_summary = "WEAK BULLISH (EMA Cross Bullish, RSI > 52)"
        elif ema9 < ema21 and rsi < 48:
            bias_summary = "WEAK BEARISH (EMA Cross Bearish, RSI < 48)"

        market_summary = f"{symbol.upper()} @ {curr_price:.5f} | Regimes: H4={h4_trend} H1={h1_trend} M15={m15_trend} M5={m5_trend}"
        m5_adx_delta = analysis_data.get("indicators", {}).get("adx_delta")
        delta_label = (
            f"{float(m5_adx_delta):+.1f}"
            if m5_adx_delta is not None
            else "unavailable"
        )
        mtf_data = (
            f"ADX Trend Strength: H4={h4_adx:.1f} | H1={h1_adx:.1f} | "
            f"M15={m15_adx:.1f} | M5={m5_adx:.1f} (delta={delta_label})"
        )
        transition_data = (
            f"Trend states: H4={h4_state} | H1={h1_state} | "
            f"M15={m15_state} | M5={m5_state}"
        )

        # 4. Compact Indicators String
        indicators_str = (
            f"EMA9={ema9} | EMA21={ema21} | RSI={rsi:.1f} | ATR_Pips={ind.get('atr_14_pips') or 0.0:.2f} | "
            f"MACD_Diff={macd:.5f} | Stoch_K/D={ind.get('stochastic', {}).get('k')}/{ind.get('stochastic', {}).get('d')}"
        )

        # 5. Compact Market Structure String
        ms = analysis_data.get("market_structure", {})

        def compress_events(analysis):
            events = (analysis or {}).get("market_structure", {}).get("structure_events", [])
            compact = []
            for event in events[:2]:
                compact.append(
                    f"{str(event.get('type', 'EVENT')).upper()}-"
                    f"{str(event.get('direction', 'NEUTRAL')).upper()}@"
                    f"{event.get('level', 'n/a')}"
                )
            return ",".join(compact) if compact else "None"

        structure_events_str = (
            f"H4={compress_events(h4_analysis)} | H1={compress_events(h1_analysis)} | "
            f"M15={compress_events(m15_analysis)} | M5={compress_events(analysis_data)}"
        )
        retest = ms.get("retest_continuation")
        retest_str = (
            (
                f"{retest.get('direction')} PRICE_PULLBACK_RESUMPTION; "
                f"closed through {retest.get('break_level')} after "
                f"{retest.get('bars_since_pullback')} bar(s); "
                f"depth={retest.get('pullback_depth_atr')} ATR"
            )
            if isinstance(retest, dict) and retest.get("kind") == "PRICE_PULLBACK_RESUMPTION"
            else
            (
                f"{retest.get('direction')} after "
                f"{retest.get('previous_state')} -> {retest.get('current_state')}; "
                f"resumption={retest.get('resumption_atr')} ATR"
            )
            if isinstance(retest, dict)
            else "None"
        )
        range_setup = ms.get("range_reversion", {}) or {}
        range_str = (
            (
                f"eligible={bool(range_setup.get('eligible'))} "
                f"direction={range_setup.get('direction') or 'NONE'} "
                f"score={range_setup.get('score', 0)} "
                f"reason={range_setup.get('reason', '')}"
            )
            if isinstance(range_setup, dict)
            else "eligible=False"
        )
        evidence_ids = build_evidence_ids(
            {
                "M5": analysis_data,
                "M15": m15_analysis or {},
                "H1": h1_analysis or {},
                "H4": h4_analysis or {},
            }
        )
        
        m5_triggers = [
            eid for eid in evidence_ids
            if eid.startswith(("M5_BOS_", "M5_CHOCH_", "M5_BREAKOUT_", "M5_RETEST_", "M5_RANGE_", "M5_LOCAL_REVERSAL_"))
        ]
        if m5_triggers:
            m5_trigger_text = f"AVAILABLE ({m5_triggers})"
        else:
            m5_trigger_text = "NONE (No M5 BOS, CHoCH, breakout, retest, or range ID. YOU MUST CHOOSE HOLD with evidence_ids=[])"
        entry_actions = permitted_entry_actions(evidence_ids)
        entry_contract = ", ".join(entry_actions) if entry_actions else "HOLD ONLY"
        local_reversal_rule = ""
        if any(eid.startswith("M5_LOCAL_REVERSAL_") for eid in evidence_ids):
            local_reversal_rule = (
                "A supplied M5_LOCAL_REVERSAL ID is a separately qualified experimental reversal: "
                "M5/M15 agree, H1/H4 oppose, and momentum is rising. It is an explicit exception "
                "to the ordinary H4 continuation and CHoCH/BOS rules below, not macro alignment. "
                "Cite this exact ID only if you independently confirm its direction; otherwise HOLD. "
                "Final risk requires confidence >= 0.85; never inflate confidence to pass."
            )

        # Compress FVG array
        fvgs = ms.get("fair_value_gaps", [])
        fvg_strings = [f"[{f.get('type')}: {f.get('bottom'):.5f}-{f.get('top'):.5f}]" for f in fvgs[:1]]
        fvg_str = ", ".join(fvg_strings) if fvg_strings else "None"

        # Compress OB array
        obs = ms.get("order_blocks", [])
        ob_strings = [f"[{o.get('type')}: {o.get('low'):.5f}-{o.get('high'):.5f}]" for o in obs[:1]]
        ob_str = ", ".join(ob_strings) if ob_strings else "None"

        structure_str = (
            f"Support={ms.get('support')} | Resistance={ms.get('resistance')} | Breakout={ms.get('breakout_status')} | "
            f"Patterns={ms.get('candlestick_patterns')} | "
            f"FVGs={fvg_str} | OrderBlocks={ob_str}"
        )

        # 6. Compress Calendar
        news_str = cls.compress_news(calendar_events, symbol)
        forex_context = forex_context or {}
        fx_context_str = (
            f"source={forex_context.get('source', 'UNAVAILABLE')} | "
            f"sessions={forex_context.get('active_sessions_utc', [])} | "
            f"base={forex_context.get('base_currency', 'n/a')} "
            f"{forex_context.get('base_strength', 0.0)} | "
            f"quote={forex_context.get('quote_currency', 'n/a')} "
            f"{forex_context.get('quote_strength', 0.0)} | "
            f"pair_bias={forex_context.get('bias', 'NEUTRAL')} | "
            f"reliable={bool(forex_context.get('reliable', False))}"
        )

        # 7. Optional Visual Highlights Placeholder
        visual_str = ""
        if settings.screenshots_enabled:
            visual_str = (
                "\n[VISUAL CHART HIGHLIGHTS]\n"
                f"Chart patterns show momentum shifts toward {m5_trend}."
            )

        if settings.allow_strong_countertrend_entries:
            counter_h4_rule = (
                "A counter-H4 continuation is exceptional: choose it only when M5, "
                "M15 and H1 all align with the action, "
                f"M5 ADX >= {settings.countertrend_min_adx:.1f}, confidence >= "
                f"{settings.countertrend_min_confidence:.0%}, and momentum is not "
                "beyond the configured extreme-RSI guard; otherwise HOLD. "
                "State the H4 conflict accurately."
            )
        else:
            counter_h4_rule = (
                "Do not choose an ordinary continuation entry against H4. A trade "
                "against the prior H4 direction is eligible only as an explicit "
                "early reversal satisfying the M5/M15 CHoCH/BOS rule above; "
                "otherwise HOLD. State the H4 conflict accurately."
            )

        # 12. Combine everything into an ultra-compact reasoning-focused prompt
        user_prompt = f"""### MARKET STATE — {market_summary}
{mtf_data}
{transition_data}

[BIAS SUMMARY]
Current Technical Setup: {bias_summary}

[KEY TECHNICALS]
- Indicators: {indicators_str}
- Structure: {structure_str}
- CHoCH/BOS events: {structure_events_str}
- Verified M5 pullback retest: {retest_str}
- Deterministic M5 range reversal: {range_str}
- Allowed Evidence IDs: {list(evidence_ids)}
- Deterministic Entry Contract: {entry_contract}
- M5 Entry Triggers Status: {m5_trigger_text}
- news: {news_str}
- broker-derived FX context: {fx_context_str}
{visual_str}

### DECISION
Use the explicit trend states and CHoCH/BOS events before the slower regime label.
CHoCH is early reversal evidence; BOS confirms direction; a PULLBACK state is not a reversal.
The deterministic entry contract owns direction. Confirm one listed action or veto it with HOLD; never choose a direction outside that contract.
Choose a permitted BUY or SELL only when its M5 Entry Trigger ID is AVAILABLE and momentum and structure agree with ADX >= {settings.entry_min_adx:.0f}.
IF M5 Entry Triggers Status is NONE, OUTPUT HOLD WITH evidence_ids=[].
An M5_RETEST ID with M15/H1 alignment is a valid fresh entry trigger when momentum agrees. A trend label by itself is context, not an entry trigger.
An early reversal requires matching M5 and M15 direction plus CHoCH or a momentum-confirmed breakout.
{local_reversal_rule}
A supplied M5_RANGE ID is the sole low-ADX exception. Choose only the direction encoded by that ID, or HOLD.
{counter_h4_rule}
When broker-derived FX context is reliable, do not trade against its pair bias.
Never trade merely to meet a profit target. Keep reasoning under 2 sentences.
For BUY or SELL, copy evidence_ids verbatim, character-for-character, from the Allowed Evidence IDs list.
Descriptive text such as "BEARISH BREAKOUT" is not an evidence ID.
Use an empty evidence_ids list for HOLD.
Output JSON only. Do not propose prices or volume; deterministic code owns the order plan.
Account balance, prior P/L and risk budget are deliberately excluded from this directional decision."""
        if live_tick:
            user_prompt += (
                "\n[LIVE BROKER QUOTE — execution context only]\n"
                f"bid={live_tick.get('bid')} ask={live_tick.get('ask')} "
                f"time={live_tick.get('time')}. Indicators and structure above "
                "remain based on completed candles; do not invent intrabar indicators.\n"
            )
        return system_prompt, user_prompt

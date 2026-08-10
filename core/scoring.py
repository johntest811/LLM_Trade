"""
core/scoring.py — Decision Scoring & Confluence Engine

Calculates independent sub-scores for trend, momentum, liquidity, volatility,
structure, risk, and news. Combines them into an Overall Trade Quality Score
and evaluates confluence across 13 variables to protect capital on micro accounts.
"""
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from app_config.settings import settings

logger = logging.getLogger("TradingSystem.ScoringEngine")


class DecisionScoringEngine:
    """
    Computes qualitative scores and confluence checks for a proposed trade setup.
    Tuned for conservative capital preservation (aiming for scores 0-100).
    """

    @staticmethod
    def calculate_confluence_score(
        action: str,
        m5_analysis: Dict[str, Any],
        m15_analysis: Optional[Dict[str, Any]],
        h1_analysis: Optional[Dict[str, Any]],
        h4_analysis: Optional[Dict[str, Any]],
        session_info: str = "UNKNOWN",
        strategy_mode: str = "",
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Evaluates 13 confluence parameters to produce a score from 0% to 100%.
        """
        factors = {}
        action = action.upper()
        if action == "HOLD":
            return 0.0, {}

        ind_m5 = m5_analysis.get("indicators", {})
        struct_m5 = m5_analysis.get("market_structure", {})

        # Helper flags
        is_buy = action == "BUY"
        is_sell = action == "SELL"

        # 1. Trend Alignment (M5, M15, H1, H4 alignment)
        normalized_mode = str(strategy_mode).upper()
        reversal = normalized_mode == "CONFIRMED_REVERSAL"
        range_mode = normalized_mode == "RANGE_REVERSION"
        range_setup = struct_m5.get("range_reversion", {}) or {}
        range_matches = bool(
            range_mode
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper() == action
        )

        def trend(analysis: Optional[Dict[str, Any]]) -> str:
            structure = (analysis or {}).get("market_structure", {})
            # Keep scoring aligned with the deterministic structure gate.  The
            # state-machine direction is canonical for every strategy mode;
            # legacy ``trend`` is only a compatibility fallback.
            value = (
                structure.get("trend_state_direction")
                or structure.get("trend")
            )
            return str(value or "NEUTRAL").upper()

        t_m5 = trend(m5_analysis)
        t_m15 = trend(m15_analysis)
        t_h1 = trend(h1_analysis)
        t_h4 = trend(h4_analysis)

        if range_mode:
            factors["trend_align"] = range_matches
        elif is_buy:
            factors["trend_align"] = (
                t_m5 == "BULLISH"
                and (t_m15 == "BULLISH" if reversal else t_h1 == "BULLISH")
            )
        else:
            factors["trend_align"] = (
                t_m5 == "BEARISH"
                and (t_m15 == "BEARISH" if reversal else t_h1 == "BEARISH")
            )

        symbol = m5_analysis.get("symbol", "")
        # Resolve pip multiplier based on currency properties
        if any(c in symbol.upper() for c in ["ETH", "LTC", "BTC"]):
            pip_multi = 1.0
        elif "XRP" in symbol.upper():
            pip_multi = 100.0
        elif any(j in symbol.upper() for j in ["JPY", "XAU", "GOLD"]):
            pip_multi = 100.0
        else:
            pip_multi = 10000.0

        # 2. Support / Resistance Proximity
        close_price = ind_m5.get("current_price", 0.0)
        sup = struct_m5.get("support", 0.0)
        res = struct_m5.get("resistance", 0.0)
        if is_buy and sup > 0:
            # Near support is good for buys (within 2.5x ATR of support)
            atr = ind_m5.get("atr_14_pips", 10.0)
            dist_pips = abs(close_price - sup) * pip_multi
            factors["sup_res_proximity"] = dist_pips < (atr * 2.5)
        elif is_sell and res > 0:
            atr = ind_m5.get("atr_14_pips", 10.0)
            dist_pips = abs(res - close_price) * pip_multi
            factors["sup_res_proximity"] = dist_pips < (atr * 2.5)
        else:
            factors["sup_res_proximity"] = False

        # 3. Order Blocks (OB) Proximity
        obs = struct_m5.get("order_blocks", [])
        factors["ob_proximity"] = False
        for ob in obs:
            if is_buy and ob.get("type") == "BULLISH":
                if close_price >= ob.get("low", 0.0) and close_price <= ob.get("high", 0.0) * 1.002:
                    factors["ob_proximity"] = True
            elif is_sell and ob.get("type") == "BEARISH":
                if close_price <= ob.get("high", 0.0) and close_price >= ob.get("low", 0.0) * 0.998:
                    factors["ob_proximity"] = True

        # 4. Fair Value Gaps (FVG)
        fvgs = struct_m5.get("fair_value_gaps", [])
        factors["fvg_alignment"] = False
        for f in fvgs:
            if is_buy and f.get("type") == "BULLISH":
                if close_price >= f.get("bottom", 0.0) and close_price <= f.get("top", 0.0):
                    factors["fvg_alignment"] = True
            elif is_sell and f.get("type") == "BEARISH":
                if close_price >= f.get("bottom", 0.0) and close_price <= f.get("top", 0.0):
                    factors["fvg_alignment"] = True

        # 5. Liquidity Pools
        liq = struct_m5.get("liquidity_zones", {})
        factors["liquidity_proximity"] = False
        if is_buy:
            levels = liq.get("sell_side_liquidity_levels", [])
            for lvl in levels:
                if abs(close_price - lvl) * pip_multi < 12.0:
                    factors["liquidity_proximity"] = True
        else:
            levels = liq.get("buy_side_liquidity_levels", [])
            for lvl in levels:
                if abs(close_price - lvl) * pip_multi < 12.0:
                    factors["liquidity_proximity"] = True

        # 6. EMA Alignment (Relaxed to EMA 9 vs EMA 21 alignment for M5 entries)
        ema9 = ind_m5.get("ema_9") or ind_m5.get("ema_20")
        ema21 = ind_m5.get("ema_21") or ind_m5.get("ema_50")
        if range_mode:
            # An inward completed candle at the outer band is the timing
            # signal; an EMA continuation cross is intentionally irrelevant.
            factors["ema_alignment"] = range_matches
        elif ema9 and ema21:
            if is_buy:
                factors["ema_alignment"] = (ema9 > ema21)
            else:
                factors["ema_alignment"] = (ema9 < ema21)
        else:
            factors["ema_alignment"] = False

        # 7. RSI Alignment
        rsi = ind_m5.get("rsi_14")
        if range_mode:
            factors["rsi_alignment"] = bool(
                rsi is not None
                and (
                    rsi <= settings.range_buy_max_rsi
                    if is_buy
                    else rsi >= settings.range_sell_min_rsi
                )
            )
        elif rsi is not None:
            if is_buy:
                factors["rsi_alignment"] = (45.0 <= rsi <= 68.0)  # Momentum without being overbought
            else:
                factors["rsi_alignment"] = (32.0 <= rsi <= 55.0)  # Momentum without being oversold
        else:
            factors["rsi_alignment"] = False

        # 8. MACD Alignment
        macd = ind_m5.get("macd", {})
        macd_diff = macd.get("diff")
        if macd_diff is not None:
            factors["macd_alignment"] = (macd_diff > 0 if is_buy else macd_diff < 0)
        else:
            factors["macd_alignment"] = False

        # 9. ATR Status (volatility expansion)
        atr = ind_m5.get("atr_14_pips", 0.0)
        is_crypto = any(c in symbol.upper() for c in ["ETH", "LTC", "XRP", "BTC"])
        if is_crypto:
            price = ind_m5.get("current_price", 1.0)
            raw_atr = atr / 100.0 if "XRP" in symbol.upper() else atr
            factors["atr_expansion"] = (raw_atr > (price * 0.0004))
        else:
            factors["atr_expansion"] = (atr >= 1.0)  # minimum volatility threshold lowered to 1.0 pips

        # 10. ADX Strength (trending market)
        adx = ind_m5.get("adx_14", 0.0)
        factors["adx_trending"] = (
            range_matches if range_mode else adx >= 18.0
        )

        # 11. Candlestick Pattern
        patterns = [str(pattern).upper() for pattern in struct_m5.get("candlestick_patterns", [])]
        direction_word = "BULLISH" if is_buy else "BEARISH"
        factors["pattern_confluence"] = bool(
            range_matches
            or any(direction_word in pattern for pattern in patterns)
        )

        # 12. Session Check (Tokyo / London / NY high volume sessions).
        # ``active_sessions`` supplies values such as NEW_YORK and may supply
        # more than one session during an overlap.  Normalize those broker
        # context values instead of silently treating them as UNKNOWN.
        if isinstance(session_info, str) or session_info is None:
            raw_sessions = [session_info]
        else:
            try:
                raw_sessions = list(session_info)
            except TypeError:
                raw_sessions = [session_info]
        session_tokens = set()
        for raw_session in raw_sessions:
            words = re.findall(r"[A-Z]+", str(raw_session or "").upper())
            session_tokens.update(words)
            if words:
                session_tokens.add("".join(words))
        factors["session_confluence"] = bool(
            session_tokens.intersection(
                {"ASIA", "TOKYO", "LONDON", "NEWYORK", "US"}
            )
        )

        # 13. Timeframe Agreement (H1 agrees with entry M5)
        expected_trend = "BULLISH" if is_buy else "BEARISH"
        factors["tf_agreement"] = (
            range_matches
            if range_mode
            else (
                t_m5 == expected_trend
                and (t_m15 == expected_trend if reversal else t_h1 == expected_trend)
            )
        )

        # Compute confluence percent
        score = (sum(1 for val in factors.values() if val) / len(factors)) * 100.0
        logger.info(f"Confluence factors calculated for {action} on {symbol}: score={score:.1f}%, details: {factors}")
        return round(score, 1), factors

    @classmethod
    def calculate_trade_quality_score(
        cls,
        action: str,
        llm_decision: Dict[str, Any],
        m5_analysis: Dict[str, Any],
        m15_analysis: Optional[Dict[str, Any]],
        h1_analysis: Optional[Dict[str, Any]],
        h4_analysis: Optional[Dict[str, Any]],
        calendar_events: Optional[List[Dict[str, Any]]],
        session_info: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Computes 7 sub-scores and aggregates them into the Overall Trade Quality Score (0-100).
        """
        action = action.upper()
        if action == "HOLD":
            return {
                "overall_score": 0.0,
                "trend_score": 0.0,
                "momentum_score": 0.0,
                "liquidity_score": 0.0,
                "volatility_score": 0.0,
                "structure_score": 0.0,
                "risk_score": 0.0,
                "news_score": 0.0
            }

        ind_m5 = m5_analysis.get("indicators", {})
        struct_m5 = m5_analysis.get("market_structure", {})
        is_buy = action == "BUY"

        # 1. Trend Alignment Score (0-100)
        def trend(analysis: Optional[Dict[str, Any]]) -> str:
            structure = (analysis or {}).get("market_structure", {})
            value = (
                structure.get("trend_state_direction")
                or structure.get("trend")
            )
            return str(value or "NEUTRAL").upper()

        t_m5 = trend(m5_analysis)
        t_m15 = trend(m15_analysis)
        t_h1 = trend(h1_analysis)
        t_h4 = trend(h4_analysis)

        expected_trend = "BULLISH" if is_buy else "BEARISH"
        strategy_mode = str(
            (llm_decision.get("_strategy") or {}).get("mode", "")
        ).upper()
        range_setup = struct_m5.get("range_reversion", {}) or {}
        range_matches = bool(
            strategy_mode == "RANGE_REVERSION"
            and range_setup.get("eligible")
            and str(range_setup.get("direction", "")).upper() == action
        )
        if range_matches:
            trend_score = 75.0
        else:
            # A direction label with weak ADX is context, not full-strength
            # confirmation. Preserve 100 for aligned, adequately trending
            # timeframes while proportionally discounting weak regimes.
            strength_floor = max(
                1.0, float(getattr(settings, "entry_min_adx", 20.0))
            )

            def trend_points(
                analysis: Optional[Dict[str, Any]], direction_value: str
            ) -> float:
                if direction_value != expected_trend:
                    return 0.0
                try:
                    timeframe_adx = float(
                        (analysis or {}).get("indicators", {}).get(
                            "adx_14", 0.0
                        )
                        or 0.0
                    )
                except (TypeError, ValueError):
                    timeframe_adx = 0.0
                strength = max(0.0, min(1.0, timeframe_adx / strength_floor))
                return 25.0 * strength

            trend_score = sum(
                (
                    trend_points(m5_analysis, t_m5),
                    trend_points(m15_analysis, t_m15),
                    trend_points(h1_analysis, t_h1),
                    trend_points(h4_analysis, t_h4),
                )
            )

        # 2. Momentum Score (0-100)
        rsi = ind_m5.get("rsi_14", 50.0)
        adx = ind_m5.get("adx_14", 20.0)
        macd = ind_m5.get("macd", {}).get("diff", 0.0)

        rsi_pts = 0
        if range_matches:
            rsi_pts = 40
        elif is_buy:
            if 50.0 <= rsi <= 65.0: rsi_pts = 40
            elif 40.0 <= rsi < 50.0: rsi_pts = 20
        else:
            if 35.0 <= rsi <= 50.0: rsi_pts = 40
            elif 50.0 < rsi <= 60.0: rsi_pts = 20

        adx_pts = 30 if range_matches else (
            min(40, int(adx - 15) * 2) if adx > 15 else 0
        )
        macd_pts = (
            20
            if range_matches or (macd > 0 if is_buy else macd < 0)
            else 0
        )
        momentum_score = float(max(0, min(100, rsi_pts + adx_pts + macd_pts)))

        # 3. Liquidity Score (0-100)
        ob_pts = 0
        obs = struct_m5.get("order_blocks", [])
        close_p = ind_m5.get("current_price", 0.0)
        for ob in obs:
            if is_buy and ob.get("type") == "BULLISH":
                if close_p >= ob.get("low", 0.0) and close_p <= ob.get("high", 0.0) * 1.002:
                    ob_pts = 60
            elif not is_buy and ob.get("type") == "BEARISH":
                if close_p <= ob.get("high", 0.0) and close_p >= ob.get("low", 0.0) * 0.998:
                    ob_pts = 60

        liq_pts = 0
        liq = struct_m5.get("liquidity_zones", {})
        symbol = m5_analysis.get("symbol", "")
        is_crypto = any(c in symbol.upper() for c in ["ETH", "LTC", "XRP", "BTC"])

        # Resolve pip multiplier based on currency properties
        if is_crypto:
            if "XRP" in symbol.upper():
                pip_multi = 100.0
            else:
                pip_multi = 1.0
        elif any(j in symbol.upper() for j in ["JPY", "XAU", "GOLD"]):
            pip_multi = 100.0
        else:
            pip_multi = 10000.0

        if is_buy:
            for lvl in liq.get("sell_side_liquidity_levels", []):
                if abs(close_p - lvl) * pip_multi < 12.0:
                    liq_pts = 40
        else:
            for lvl in liq.get("buy_side_liquidity_levels", []):
                if abs(close_p - lvl) * pip_multi < 12.0:
                    liq_pts = 40
        liquidity_score = float(ob_pts + liq_pts)

        # 4. Volatility Score (0-100)
        atr_pips = ind_m5.get("atr_14_pips", 0.0)
        bb_width = ind_m5.get("bollinger_bands", {}).get("width_pct", 0.0)
        # Optimal volatility: not too low (ranging squeeze) and not excessively high (panic spreads)
        if is_crypto:
            price = ind_m5.get("current_price", 1.0)
            raw_atr = atr_pips / 100.0 if "XRP" in symbol.upper() else atr_pips
            pct_atr = (raw_atr / price) * 100.0
            atr_pts = 50 if (0.04 <= pct_atr <= 0.6) else (20 if pct_atr > 0.6 else 10)
        else:
            atr_pts = 50 if (5.0 <= atr_pips <= 25.0) else (20 if atr_pips > 25.0 else 10)
        bb_pts = 50 if (bb_width > 0.05) else 15
        volatility_score = float(atr_pts + bb_pts)

        # 5. Market Structure Score (0-100)
        bos_choch = struct_m5.get("structure_events", [])
        breakout = str(struct_m5.get("breakout_status", "NONE")).upper()
        patterns = [str(pattern).upper() for pattern in struct_m5.get("candlestick_patterns", [])]
        expected_direction = "BULLISH" if is_buy else "BEARISH"

        struct_pts = 80 if range_matches else 20
        for ev in bos_choch:
            if str(ev.get("direction", "")).upper() != expected_direction:
                continue
            if ev.get("type") == "CHOCH":
                struct_pts = max(struct_pts, 60)
            elif ev.get("type") == "BOS":
                struct_pts = max(struct_pts, 40)

        breakout_pts = 30 if expected_direction in breakout and "BREAKOUT" in breakout else 0
        pattern_pts = 10 if any(expected_direction in pattern for pattern in patterns) else 0
        structure_score = float(min(100, struct_pts + breakout_pts + pattern_pts))

        # 6. Risk Score (0-100)
        entry = float(llm_decision.get("entry", 0.0) or 0.0)
        sl = float(llm_decision.get("stop_loss", 0.0) or 0.0)
        tp = float(llm_decision.get("take_profit", 0.0) or 0.0)

        risk_score = 0.0
        if entry > 0 and sl > 0 and tp > 0:
            risk_dist = abs(entry - sl)
            reward_dist = abs(tp - entry)
            if risk_dist > 0:
                rr = reward_dist / risk_dist
                if rr >= 2.0: risk_score = 100.0
                elif rr + 1e-9 >= settings.min_risk_reward_ratio: risk_score = 75.0
                elif rr >= 1.0: risk_score = 40.0
                else: risk_score = 10.0

        # 7. News Risk Score (0-100)
        # Missing calendar data is unknown, not the same as a clear calendar.
        news_score = 50.0 if calendar_events is None else 100.0
        # Check calendar proximity
        now = datetime.now()
        base_curr = m5_analysis.get("symbol", "")[:3].upper()
        quote_curr = m5_analysis.get("symbol", "")[3:].upper()
        for ev in calendar_events or []:
            impact = ev.get("impact", "LOW").upper()
            currency = ev.get("currency", "").upper()
            if impact in ["HIGH", "MEDIUM"] and currency in [base_curr, quote_curr]:
                ev_time_str = ev.get("time", "")
                try:
                    ev_time = datetime.strptime(ev_time_str, "%Y-%m-%d %H:%M:%S")
                    diff_mins = abs((ev_time - now).total_seconds()) / 60.0
                    if diff_mins < 30.0:
                        news_score = 10.0 # Extreme proximity
                    elif diff_mins < 60.0:
                        news_score = 40.0
                    elif diff_mins < 120.0:
                        news_score = 70.0
                except Exception:
                    pass

        # Weighted calculation of the Overall Trade Quality Score
        weights = {
            "trend": 0.25,
            "momentum": 0.15,
            "liquidity": 0.15,
            "volatility": 0.10,
            "structure": 0.15,
            "risk": 0.15,
            "news": 0.05
        }

        overall = (
            trend_score * weights["trend"] +
            momentum_score * weights["momentum"] +
            liquidity_score * weights["liquidity"] +
            volatility_score * weights["volatility"] +
            structure_score * weights["structure"] +
            risk_score * weights["risk"] +
            news_score * weights["news"]
        )

        return {
            "overall_score": round(overall, 1),
            "trend_score": round(trend_score, 1),
            "momentum_score": round(momentum_score, 1),
            "liquidity_score": round(liquidity_score, 1),
            "volatility_score": round(volatility_score, 1),
            "structure_score": round(structure_score, 1),
            "risk_score": round(risk_score, 1),
            "news_score": round(news_score, 1)
        }

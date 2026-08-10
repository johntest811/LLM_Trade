import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from app_config.settings import settings
from trading_indicators.calculations import TechnicalIndicators
from market_structure.swing_detector import SwingDetector
from market_structure.breakout import BreakoutDetector
from market_structure.candlestick import CandlestickPatternDetector
from market_structure.smc import SMCAnalyzer
from core.trend_state import classify_trend_state

logger = logging.getLogger("TradingSystem.AnalysisEngine")

class MarketAnalysisEngine:
    """
    Computes technical indicators and Smart Money Concepts (SMC) structural levels.
    Uses timestamp-based caching to avoid recalculations if the latest candle hasn't closed.
    """
    def __init__(self) -> None:
        # Structured caching: {(symbol, timeframe): (last_candle_timestamp, analysis_dict)}
        self._cache: Dict[Tuple[str, str], Tuple[datetime, Dict[str, Any]]] = {}

    def analyze(self, symbol: str, timeframe: str, df_candles: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Runs the full analysis pipeline on the provided historical DataFrame.
        Returns a clean structured dictionary ready for direct LLM ingestion.
        """
        if df_candles.empty or len(df_candles) < 25:
            logger.warning(f"Insufficient candle count ({len(df_candles)}) to run analysis for {symbol}")
            return None

        # Resolve pip multiplier based on currency pair properties
        if any(c in symbol.upper() for c in ["ETH", "LTC", "BTC"]):
            pip_multiplier = 1.0
        elif "XRP" in symbol.upper():
            pip_multiplier = 100.0
        elif any(j in symbol.upper() for j in ["JPY", "XAU", "GOLD"]):
            pip_multiplier = 100.0
        else:
            pip_multiplier = 10000.0

        # Extract the latest candle close timestamp to check cache status
        latest_candle = df_candles.iloc[-1]
        latest_time = latest_candle['time']
        
        cache_key = (symbol.upper(), timeframe.upper())
        cached_record = self._cache.get(cache_key)
        
        # 1. Skip calculations if latest candle is already cached
        if cached_record and cached_record[0] == latest_time:
            logger.debug(f"Cache hit: Skip recalculating unchanged candles for {symbol} ({timeframe})")
            return cached_record[1]

        logger.info(f"Running full market and SMC analysis for {symbol} ({timeframe})...")

        # 2. Run Technical Indicators Calculations
        df_indicators = TechnicalIndicators.calculate_all(df_candles)
        last_row = df_indicators.iloc[-1]
        atr_value = (
            float(last_row.get("atr_14"))
            if pd.notna(last_row.get("atr_14"))
            else 0.0
        )

        # 3. Detect Swing Levels & Trends
        swing_highs, swing_lows = SMCAnalyzer.detect_swings(df_indicators, window=5)
        
        support, resistance = SwingDetector.calculate_levels(df_indicators, window=20)
        latest_close = float(last_row['close'])
        breakout_previous_close = (
            float(df_indicators.iloc[-2]["close"])
            if len(df_indicators) >= 2
            else None
        )
        breakout = BreakoutDetector.detect(
            latest_close,
            support,
            resistance,
            previous_close=breakout_previous_close,
            atr=atr_value,
            min_displacement_atr=settings.breakout_min_displacement_atr,
        )
        patterns = CandlestickPatternDetector.detect_patterns(df_indicators)

        # Basic EMA trend classification
        current_trend = "NEUTRAL"
        ema50 = last_row.get("ema_50")
        ema200 = last_row.get("ema_200")
        close_price = last_row.get("close")
        if pd.notna(ema50) and pd.notna(ema200) and pd.notna(close_price):
            if ema50 > ema200 and close_price > ema200:
                current_trend = "BULLISH"
            elif ema50 < ema200 and close_price < ema200:
                current_trend = "BEARISH"

        # 4. Run SMC Calculations
        structure_events = SMCAnalyzer.detect_bos_choch(
            df_indicators,
            swing_highs,
            swing_lows,
            current_trend,
            min_displacement_atr=settings.breakout_min_displacement_atr,
        )
        trend_state = classify_trend_state(
            slow_trend=current_trend,
            indicators={
                "ema_9": last_row.get("ema_9"),
                "ema_21": last_row.get("ema_21"),
                "rsi_14": last_row.get("rsi_14"),
                "adx_14": last_row.get("adx"),
                "macd": {"diff": last_row.get("macd_diff")},
            },
            structure_events=structure_events,
            breakout_status=breakout,
        )
        fvgs = SMCAnalyzer.detect_fvgs(df_indicators)
        order_blocks = SMCAnalyzer.detect_order_blocks(df_indicators, swing_highs, swing_lows)
        liquidity = SMCAnalyzer.detect_liquidity_zones(swing_highs, swing_lows, pip_multiplier=pip_multiplier)

        # 5. Extract Supply & Demand Zones from active Order Blocks and key Swings
        supply_zones: List[Dict[str, float]] = []
        demand_zones: List[Dict[str, float]] = []
        for ob in order_blocks:
            if ob["type"] == "BULLISH":
                demand_zones.append({"top": ob["high"], "bottom": ob["low"]})
            else:
                supply_zones.append({"top": ob["high"], "bottom": ob["low"]})

        # Append recent significant swings to S&D zones if OBs are empty
        if not demand_zones and swing_lows:
            recent_low = swing_lows[-1]["price"]
            demand_zones.append({"top": recent_low + (5.0 / pip_multiplier), "bottom": recent_low})
        if not supply_zones and swing_highs:
            recent_high = swing_highs[-1]["price"]
            supply_zones.append({"top": recent_high, "bottom": recent_high - (5.0 / pip_multiplier)})

        # 6. Build the final structured analysis payload
        current_adx = (
            float(last_row.get("adx")) if pd.notna(last_row.get("adx")) else None
        )
        previous_adx = (
            float(df_indicators.iloc[-2].get("adx"))
            if len(df_indicators) >= 2
            and pd.notna(df_indicators.iloc[-2].get("adx"))
            else None
        )
        candle_range = float(last_row["high"] - last_row["low"])
        previous_close = (
            float(df_indicators.iloc[-2]["close"])
            if len(df_indicators) >= 2 and pd.notna(df_indicators.iloc[-2].get("close"))
            else float(last_row["open"])
        )
        opening_gap = abs(float(last_row["open"]) - previous_close)
        candle_range_atr = candle_range / atr_value if atr_value > 0 else 0.0
        opening_gap_atr = opening_gap / atr_value if atr_value > 0 else 0.0
        candle_body_atr_signed = (
            (float(last_row["close"]) - float(last_row["open"])) / atr_value
            if atr_value > 0
            else 0.0
        )
        analysis = {
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "timestamp": latest_time.strftime("%Y-%m-%d %H:%M:%S"),
            "indicators": {
                "current_price": float(last_row['close']),
                "ema_9": float(last_row.get('ema_9')) if pd.notna(last_row.get('ema_9')) else None,
                "ema_21": float(last_row.get('ema_21')) if pd.notna(last_row.get('ema_21')) else None,
                "ema_20": float(last_row.get('ema_20')) if pd.notna(last_row.get('ema_20')) else None,
                "ema_50": float(last_row.get('ema_50')) if pd.notna(last_row.get('ema_50')) else None,
                "ema_100": float(last_row.get('ema_100')) if pd.notna(last_row.get('ema_100')) else None,
                "ema_200": float(last_row.get('ema_200')) if pd.notna(last_row.get('ema_200')) else None,
                "rsi_14": float(last_row.get('rsi_14')) if pd.notna(last_row.get('rsi_14')) else None,
                "atr_14": float(last_row.get('atr_14')) if pd.notna(last_row.get('atr_14')) else None,
                "atr_14_pips": float(last_row.get('atr_14') * pip_multiplier) if pd.notna(last_row.get('atr_14')) else None,
                "candle_range_atr": round(candle_range_atr, 4),
                "candle_body_atr_signed": round(candle_body_atr_signed, 4),
                "opening_gap_atr": round(opening_gap_atr, 4),
                "candle_return_atr": round(
                    (latest_close - previous_close) / atr_value
                    if atr_value > 0
                    else 0.0,
                    4,
                ),
                "adx_14": current_adx,
                "adx_previous": previous_adx,
                "adx_delta": (
                    current_adx - previous_adx
                    if current_adx is not None and previous_adx is not None
                    else None
                ),
                "macd": {
                    "line": float(last_row.get('macd_line')) if pd.notna(last_row.get('macd_line')) else None,
                    "signal": float(last_row.get('macd_signal')) if pd.notna(last_row.get('macd_signal')) else None,
                    "diff": float(last_row.get('macd_diff')) if pd.notna(last_row.get('macd_diff')) else None
                },
                "bollinger_bands": {
                    "upper": float(last_row.get('bb_upper')) if pd.notna(last_row.get('bb_upper')) else None,
                    "middle": float(last_row.get('bb_middle')) if pd.notna(last_row.get('bb_middle')) else None,
                    "lower": float(last_row.get('bb_lower')) if pd.notna(last_row.get('bb_lower')) else None,
                    "width_pct": float(last_row.get('bb_width_pct')) if pd.notna(last_row.get('bb_width_pct')) else None
                },
                "stochastic": {
                    "k": float(last_row.get('stoch_k')) if pd.notna(last_row.get('stoch_k')) else None,
                    "d": float(last_row.get('stoch_d')) if pd.notna(last_row.get('stoch_d')) else None
                },
                "vwap": float(last_row.get('vwap')) if pd.notna(last_row.get('vwap')) else None
            },
            "market_structure": {
                "trend": current_trend,
                "trend_state": trend_state["state"],
                "trend_state_direction": trend_state["direction"],
                "regime_trend": trend_state["regime"],
                "fast_trend": trend_state["fast_direction"],
                "trend_state_evidence": trend_state["evidence"],
                "support": round(support, 5),
                "resistance": round(resistance, 5),
                "breakout_status": breakout,
                "candlestick_patterns": patterns,
                "swings": {
                    "last_high": swing_highs[-1] if swing_highs else {},
                    "last_low": swing_lows[-1] if swing_lows else {}
                },
                "structure_events": structure_events,
                "order_blocks": order_blocks[:3],
                "fair_value_gaps": fvgs[:3],
                "liquidity_zones": liquidity,
                "supply_zones": supply_zones[:2],
                "demand_zones": demand_zones[:2]
            }
        }

        # Save to cache
        self._cache[cache_key] = (latest_time, analysis)
        return analysis

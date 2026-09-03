import pandas as pd
from typing import List, Dict, Any, Tuple

class SMCAnalyzer:
    """
    Calculates advanced Smart Money Concepts (SMC) and market structure levels:
    - Swing Highs & Lows (HH, HL, LH, LL)
    - Break of Structure (BOS) & Change of Character (CHoCH)
    - Order Blocks (OB)
    - Fair Value Gaps (FVG)
    - Liquidity Zones (Double Tops/Bottoms)
    - Supply & Demand Zones
    """

    @staticmethod
    def detect_swings(df: pd.DataFrame, window: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Detects swing highs and swing lows using a rolling window fractal peak/trough search.
        """
        highs: List[Dict[str, Any]] = []
        lows: List[Dict[str, Any]] = []
        
        if len(df) < (window * 2 + 1):
            return highs, lows

        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        time_arr = df['time'].to_numpy()

        for i in range(window, len(df) - window):
            # Check if high is local maximum
            is_high = True
            for j in range(1, window + 1):
                if high_arr[i] < high_arr[i - j] or high_arr[i] < high_arr[i + j]:
                    is_high = False
                    break
            
            # Check if low is local minimum
            is_low = True
            for j in range(1, window + 1):
                if low_arr[i] > low_arr[i - j] or low_arr[i] > low_arr[i + j]:
                    is_low = False
                    break

            time_str = pd.to_datetime(time_arr[i]).strftime("%Y-%m-%d %H:%M:%S")

            if is_high:
                highs.append({
                    "index": i,
                    "price": float(high_arr[i]),
                    "time": time_str,
                    "label": "SH" # Placeholder for classification
                })
            if is_low:
                lows.append({
                    "index": i,
                    "price": float(low_arr[i]),
                    "time": time_str,
                    "label": "SL" # Placeholder for classification
                })

        # Classify swing highs (HH vs LH)
        for idx in range(len(highs)):
            if idx == 0:
                continue
            prev_price = highs[idx - 1]["price"]
            curr_price = highs[idx]["price"]
            label = "HH" if curr_price > prev_price else "LH"
            highs[idx] = {**highs[idx], "label": label}

        # Classify swing lows (HL vs LL)
        for idx in range(len(lows)):
            if idx == 0:
                continue
            prev_price = lows[idx - 1]["price"]
            curr_price = lows[idx]["price"]
            label = "HL" if curr_price > prev_price else "LL"
            lows[idx] = {**lows[idx], "label": label}

        return highs, lows

    @staticmethod
    def detect_bos_choch(
        df: pd.DataFrame, 
        swing_highs: List[Dict[str, Any]], 
        swing_lows: List[Dict[str, Any]], 
        current_trend: str,
        min_displacement_atr: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Detects Break of Structure (BOS) and Change of Character (CHoCH) structural breakout events.
        """
        events: List[Dict[str, Any]] = []
        if len(df) < 2 or not swing_highs or not swing_lows:
            return events

        latest_close = float(df.iloc[-1]['close'])
        previous_close = float(df.iloc[-2]['close'])
        latest_time = pd.to_datetime(df.iloc[-1]['time']).strftime("%Y-%m-%d %H:%M:%S")

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]
        high_level = float(last_high["price"])
        low_level = float(last_low["price"])
        atr = float(df.iloc[-1].get("atr_14", 0.0) or 0.0)
        required_displacement = max(
            0.0, atr * float(min_displacement_atr or 0.0)
        )

        # Structure events are transitions, not persistent states. Requiring the
        # prior completed candle to remain on the unbroken side of the level
        # prevents the same BOS/CHoCH from being emitted on every later candle.
        crossed_above_high = (
            previous_close <= high_level < latest_close
            and latest_close - high_level + 1e-12 >= required_displacement
        )
        crossed_below_low = (
            previous_close >= low_level > latest_close
            and low_level - latest_close + 1e-12 >= required_displacement
        )

        # Bullish market conditions
        if current_trend == "BULLISH":
            # BOS: Close price breaks above the previous swing high (trend continuation)
            if crossed_above_high:
                events.append({
                    "type": "BOS",
                    "direction": "BULLISH",
                    "level": high_level,
                    "time": latest_time,
                    "description": f"Price broke above previous swing high {high_level:.5f} (Trend continuation)"
                })
            # CHoCH: Close price breaks below the previous swing low (early trend reversal)
            elif crossed_below_low:
                events.append({
                    "type": "CHOCH",
                    "direction": "BEARISH",
                    "level": low_level,
                    "time": latest_time,
                    "description": f"Price broke below previous swing low {low_level:.5f} (Potential reversal)"
                })

        # Bearish market conditions
        elif current_trend == "BEARISH":
            # BOS: Close price breaks below the previous swing low (trend continuation)
            if crossed_below_low:
                events.append({
                    "type": "BOS",
                    "direction": "BEARISH",
                    "level": low_level,
                    "time": latest_time,
                    "description": f"Price broke below previous swing low {low_level:.5f} (Trend continuation)"
                })
            # CHoCH: Close price breaks above the previous swing high (early trend reversal)
            elif crossed_above_high:
                events.append({
                    "type": "CHOCH",
                    "direction": "BULLISH",
                    "level": high_level,
                    "time": latest_time,
                    "description": f"Price broke above previous swing high {high_level:.5f} (Potential reversal)"
                })

        return events

    @staticmethod
    def detect_fvgs(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifies Fair Value Gaps (FVG) within a 3-candle rolling range.
        """
        fvgs: List[Dict[str, Any]] = []
        if len(df) < 3:
            return fvgs

        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        close_arr = df['close'].to_numpy()
        open_arr = df['open'].to_numpy()
        time_arr = df['time'].to_numpy()

        # Scan the last 15 completed candles
        start_idx = max(2, len(df) - 15)
        for i in range(start_idx, len(df)):
            # 1. Bullish FVG: Low of candle[i] is greater than High of candle[i-2]
            if low_arr[i] > high_arr[i-2]:
                # Make sure the middle candle was strongly bullish
                if close_arr[i-1] > open_arr[i-1]:
                    time_str = pd.to_datetime(time_arr[i-1]).strftime("%Y-%m-%d %H:%M:%S")
                    fvgs.append({
                        "type": "BULLISH",
                        "top": float(low_arr[i]),
                        "bottom": float(high_arr[i-2]),
                        "gap_size": float(low_arr[i] - high_arr[i-2]),
                        "time": time_str
                    })
            
            # 2. Bearish FVG: High of candle[i] is lower than Low of candle[i-2]
            elif high_arr[i] < low_arr[i-2]:
                # Make sure the middle candle was strongly bearish
                if close_arr[i-1] < open_arr[i-1]:
                    time_str = pd.to_datetime(time_arr[i-1]).strftime("%Y-%m-%d %H:%M:%S")
                    fvgs.append({
                        "type": "BEARISH",
                        "top": float(low_arr[i-2]),
                        "bottom": float(high_arr[i]),
                        "gap_size": float(low_arr[i-2] - high_arr[i]),
                        "time": time_str
                    })

        return fvgs

    @staticmethod
    def detect_order_blocks(
        df: pd.DataFrame, 
        swing_highs: List[Dict[str, Any]], 
        swing_lows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detects active Bullish and Bearish Order Blocks (OB).
        Bullish OB: last down candle before a structure break above a swing high.
        Bearish OB: last up candle before a structure break below a swing low.
        """
        obs: List[Dict[str, Any]] = []
        if len(df) < 5 or not swing_highs or not swing_lows:
            return obs

        close_arr = df['close'].to_numpy()
        open_arr = df['open'].to_numpy()
        high_arr = df['high'].to_numpy()
        low_arr = df['low'].to_numpy()
        time_arr = df['time'].to_numpy()

        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]
        
        # Look back over recent candles to find opposing candles
        lookback = min(15, len(df) - 1)
        for i in range(len(df) - lookback, len(df)):
            close_price = close_arr[i]
            time_str = pd.to_datetime(time_arr[i]).strftime("%Y-%m-%d %H:%M:%S")
            
            # 1. Bullish OB: Price breaks above resistance, check if candle i was the last down candle
            if close_price > last_sh["price"]:
                # Find the last bearish candle prior to this breakout index
                for j in range(i - 1, max(0, i - 10), -1):
                    if close_arr[j] < open_arr[j]:
                        obs.append({
                            "type": "BULLISH",
                            "high": float(high_arr[j]),
                            "low": float(low_arr[j]),
                            "price": float(close_arr[j]),
                            "time": pd.to_datetime(time_arr[j]).strftime("%Y-%m-%d %H:%M:%S"),
                            "description": "Bullish Order Block (Demand zone marker)"
                        })
                        break
                break

            # 2. Bearish OB: Price breaks below support, check if candle i was the last up candle
            if close_price < last_sl["price"]:
                # Find the last bullish candle prior to this breakout index
                for j in range(i - 1, max(0, i - 10), -1):
                    if close_arr[j] > open_arr[j]:
                        obs.append({
                            "type": "BEARISH",
                            "high": float(high_arr[j]),
                            "low": float(low_arr[j]),
                            "price": float(close_arr[j]),
                            "time": pd.to_datetime(time_arr[j]).strftime("%Y-%m-%d %H:%M:%S"),
                            "description": "Bearish Order Block (Supply zone marker)"
                        })
                        break
                break

        return obs

    @staticmethod
    def detect_liquidity_zones(
        swing_highs: List[Dict[str, Any]], 
        swing_lows: List[Dict[str, Any]],
        pip_threshold: float = 5.0,
        pip_multiplier: float = 10000.0
    ) -> Dict[str, List[float]]:
        """
        Identifies liquidity pool zones (Buy Stop Liquidity above double tops, 
        and Sell Stop Liquidity below double bottoms).
        """
        buy_liquidity: List[float] = []
        sell_liquidity: List[float] = []

        # Find double tops / Equal Highs (Buy stops cluster above)
        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                p1 = swing_highs[i]["price"]
                p2 = swing_highs[j]["price"]
                diff_pips = abs(p1 - p2) * pip_multiplier
                if diff_pips <= pip_threshold:
                    buy_liquidity.append(round(max(p1, p2), 5))

        # Find double bottoms / Equal Lows (Sell stops cluster below)
        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                p1 = swing_lows[i]["price"]
                p2 = swing_lows[j]["price"]
                diff_pips = abs(p1 - p2) * pip_multiplier
                if diff_pips <= pip_threshold:
                    sell_liquidity.append(round(min(p1, p2), 5))

        return {
            "buy_side_liquidity_levels": sorted(set(buy_liquidity)),
            "sell_side_liquidity_levels": sorted(
                set(sell_liquidity), reverse=True
            ),
        }

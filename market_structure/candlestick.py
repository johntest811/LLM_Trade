import pandas as pd
from typing import List

class CandlestickPatternDetector:
    """
    Identifies common candlestick reversal and continuation patterns from historical candle data.
    """
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> List[str]:
        """
        Scans the latest completed candle for Hammer, Shooting Star, Doji, and Engulfing patterns.
        """
        if df.empty or len(df) < 2:
            return []

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        patterns: List[str] = []

        # Calculate values
        body_last = abs(last_row['close'] - last_row['open'])
        body_prev = abs(prev_row['close'] - prev_row['open'])
        range_last = last_row['high'] - last_row['low']

        is_bullish_last = last_row['close'] > last_row['open']
        is_bearish_last = last_row['close'] < last_row['open']
        is_bullish_prev = prev_row['close'] > prev_row['open']
        is_bearish_prev = prev_row['close'] < prev_row['open']

        # 1. Doji Detection (very small body relative to total range)
        if range_last > 0 and (body_last / range_last) < 0.1:
            patterns.append("Doji (Indecision)")

        # 2. Bullish Engulfing
        if is_bearish_prev and is_bullish_last and last_row['close'] >= prev_row['open'] and last_row['open'] <= prev_row['close']:
            patterns.append("Bullish Engulfing (Bullish Reversal)")

        # 3. Bearish Engulfing
        if is_bullish_prev and is_bearish_last and last_row['close'] <= prev_row['open'] and last_row['open'] >= prev_row['close']:
            patterns.append("Bearish Engulfing (Bearish Reversal)")

        # 4. Hammer (Long lower shadow, small body near the candle high)
        lower_shadow = min(last_row['open'], last_row['close']) - last_row['low']
        upper_shadow = last_row['high'] - max(last_row['open'], last_row['close'])
        if body_last > 0 and (lower_shadow / body_last) > 2 and (upper_shadow / body_last) < 0.5:
            patterns.append("Hammer (Bullish Reversal)")

        # 5. Shooting Star (Long upper shadow, small body near the candle low)
        if body_last > 0 and (upper_shadow / body_last) > 2 and (lower_shadow / body_last) < 0.5:
            patterns.append("Shooting Star (Bearish Reversal)")

        return patterns

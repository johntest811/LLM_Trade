import pandas as pd
from typing import Tuple

class SwingDetector:
    """
    Detects swing highs, swing lows, and calculates support and resistance lines.
    """
    @staticmethod
    def calculate_levels(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """
        Returns the (Support, Resistance) levels calculated over a recent candle lookback window.
        Support = lowest swing low in lookback.
        Resistance = highest swing high in lookback.
        """
        if df.empty or len(df) <= window:
            return 0.0, 0.0

        # We look back excluding the active forming candle (the last row)
        recent_highs = df['high'].iloc[-window-1:-1]
        recent_lows = df['low'].iloc[-window-1:-1]

        support = float(recent_lows.min())
        resistance = float(recent_highs.max())

        return support, resistance

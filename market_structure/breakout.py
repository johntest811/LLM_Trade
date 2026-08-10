class BreakoutDetector:
    """
    Identifies bullish and bearish price breakouts above swing support and resistance thresholds.
    """
    @staticmethod
    def detect(
        close_price: float,
        support: float,
        resistance: float,
        previous_close: float | None = None,
        atr: float = 0.0,
        min_displacement_atr: float = 0.0,
    ) -> str:
        """
        Returns a string only when the latest completed close crosses a level.

        A breakout is an event, not a persistent state. Once price is already
        beyond the level, later candles must not reuse it as fresh evidence.
        ``previous_close=None`` retains compatibility for direct legacy callers.
        Returns: "BULLISH BREAKOUT", "BEARISH BREAKOUT", or "None".
        """
        if resistance == 0.0 or support == 0.0:
            return "None"

        required_displacement = max(
            0.0, float(atr or 0.0) * float(min_displacement_atr or 0.0)
        )
        crossed_above = (
            close_price > resistance
            and close_price - resistance + 1e-12 >= required_displacement
            and (previous_close is None or previous_close <= resistance)
        )
        crossed_below = (
            close_price < support
            and support - close_price + 1e-12 >= required_displacement
            and (previous_close is None or previous_close >= support)
        )
        if crossed_above:
            return f"BULLISH BREAKOUT (Price {close_price:.5f} broke above resistance {resistance:.5f})"
        if crossed_below:
            return f"BEARISH BREAKOUT (Price {close_price:.5f} broke below support {support:.5f})"

        return "None"

import logging
import pandas as pd
import ta

logger = logging.getLogger("TradingSystem.Indicators")

class TechnicalIndicators:
    """
    Handles calculations of technical indicators on historical candles.
    """
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates EMA, RSI, MACD, Bollinger Bands, and ADX on the input DataFrame.
        """
        if df.empty or len(df) < 25:
            logger.warning("DataFrame is too small to calculate indicators. Need at least 25 bars.")
            return df

        # Make copy to avoid setting-with-copy warnings
        df_calc = df.copy()

        try:
            # 1. EMAs
            df_calc['ema_9'] = ta.trend.ema_indicator(close=df_calc['close'], window=9)
            df_calc['ema_20'] = ta.trend.ema_indicator(close=df_calc['close'], window=20)
            df_calc['ema_21'] = ta.trend.ema_indicator(close=df_calc['close'], window=21)
            df_calc['ema_50'] = ta.trend.ema_indicator(close=df_calc['close'], window=50)
            df_calc['ema_100'] = ta.trend.ema_indicator(close=df_calc['close'], window=100)
            df_calc['ema_200'] = ta.trend.ema_indicator(close=df_calc['close'], window=200)

            # VWAP Calculation (using typical price and tick volume)
            typical_price = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3.0
            volume = df_calc['tick_volume'].replace(0, 1)
            df_calc['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()

            # 2. RSI
            df_calc['rsi_14'] = ta.momentum.rsi(close=df_calc['close'], window=14)

            # 3. MACD
            macd = ta.trend.MACD(close=df_calc['close'], window_slow=26, window_fast=12, window_sign=9)
            df_calc['macd_line'] = macd.macd()
            df_calc['macd_signal'] = macd.macd_signal()
            df_calc['macd_diff'] = macd.macd_diff()

            # 4. Bollinger Bands
            bb = ta.volatility.BollingerBands(close=df_calc['close'], window=20, window_dev=2)
            df_calc['bb_upper'] = bb.bollinger_hband()
            df_calc['bb_lower'] = bb.bollinger_lband()
            df_calc['bb_middle'] = bb.bollinger_mavg()
            # Calculate band width percentage
            df_calc['bb_width_pct'] = ((df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle']) * 100.0

            # 5. ADX
            # Ensure correct high, low, close columns are passed
            adx = ta.trend.ADXIndicator(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], window=14)
            df_calc['adx'] = adx.adx()

            # 6. Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], window=14, smooth_window=3
            )
            df_calc['stoch_k'] = stoch.stoch()
            df_calc['stoch_d'] = stoch.stoch_signal()

            # 7. ATR (Average True Range)
            atr = ta.volatility.AverageTrueRange(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], window=14)
            df_calc['atr_14'] = atr.average_true_range()

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise

        return df_calc

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from mt5.safe_api import mt5

from app_config.settings import settings
from risk.instruments import pip_size, instrument_asset_class
from mt5.timebase import (
    OFFSET_MATCH_TOLERANCE_SECONDS,
    broker_tick_age_seconds,
    infer_positive_server_offset_seconds,
    normalized_broker_epoch,
)

logger = logging.getLogger("TradingSystem.MT5Data")

_MAX_TICK_REBUILD_SECONDS = 12 * 60 * 60


_TIMEFRAMES = {
    "M1": (mt5.TIMEFRAME_M1, 60),
    "M5": (mt5.TIMEFRAME_M5, 5 * 60),
    "M15": (mt5.TIMEFRAME_M15, 15 * 60),
    "M30": (mt5.TIMEFRAME_M30, 30 * 60),
    "H1": (mt5.TIMEFRAME_H1, 60 * 60),
    "H4": (mt5.TIMEFRAME_H4, 4 * 60 * 60),
    "D1": (mt5.TIMEFRAME_D1, 24 * 60 * 60),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MT5DataReader:
    """
    Retrieves real-time and historical market data from MetaTrader 5.
    """
    def __init__(self, connection_manager) -> None:
        self._conn = connection_manager
        self._last_stale_bar: dict[tuple[str, str], str] = {}
        self._server_offsets: dict[str, int] = {}
        self._tick_rebuild_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._last_tick_rebuild_bar: dict[tuple[str, str], str] = {}
        self._frozen_history_symbols: set[str] = set()

    def _server_offset_seconds(self, symbol: str, tick, now_epoch: float) -> int:
        """Detect and remember Pepperstone's positive broker-server UTC offset."""
        key = symbol.upper()
        raw_tick_time = float(getattr(tick, "time", 0.0) or 0.0)
        detected = infer_positive_server_offset_seconds(
            raw_tick_time,
            now_epoch=now_epoch,
        )
        if detected:
            previous = self._server_offsets.get(key)
            self._server_offsets[key] = detected
            if previous != detected:
                logger.info(
                    "Detected %s broker clock offset of UTC+%d for MT5 history normalization.",
                    symbol,
                    detected // 3600,
                )
            return detected

        # A current UTC timestamp explicitly resets an older cached DST offset.
        if raw_tick_time > 0 and abs(raw_tick_time - now_epoch) <= OFFSET_MATCH_TOLERANCE_SECONDS:
            self._server_offsets[key] = 0
            return 0
        return self._server_offsets.get(key, 0)

    def _rebuild_stale_history_from_ticks(
        self,
        *,
        symbol: str,
        timeframe: str,
        timeframe_seconds: int,
        count: int,
        broker_frame: pd.DataFrame,
        raw_current_bar_open: int,
        server_offset_seconds: int,
        now_epoch: float,
        tick,
    ) -> Optional[pd.DataFrame]:
        """Repair a frozen MT5 candle cache using immutable broker tick history."""
        if broker_frame is None or broker_frame.empty or tick is None:
            return None
        if broker_tick_age_seconds(
            float(getattr(tick, "time", 0.0) or 0.0),
            now_epoch=now_epoch,
            offset_seconds=server_offset_seconds,
            symbol=symbol,
        ) > settings.max_tick_age_seconds:
            return None

        expected_latest_raw = raw_current_bar_open - timeframe_seconds
        expected_latest_utc = pd.to_datetime(
            expected_latest_raw - server_offset_seconds,
            unit="s",
            utc=True,
        )
        cache_key = (symbol.upper(), timeframe)
        cached = self._tick_rebuild_cache.get(cache_key)
        if cached is not None and not cached.empty:
            if cached.iloc[-1]["time"] == expected_latest_utc:
                return cached.copy(deep=True)

        # Rebuild from the opening of the last broker bar because a frozen bar
        # can exist but contain only its first few ticks.
        latest_broker_utc = int(broker_frame.iloc[-1]["time"].timestamp())
        rebuild_start_raw = latest_broker_utc + server_offset_seconds
        gap_seconds = raw_current_bar_open - rebuild_start_raw
        if gap_seconds <= 0 or gap_seconds > _MAX_TICK_REBUILD_SECONDS:
            return None

        start = datetime.fromtimestamp(rebuild_start_raw, tz=timezone.utc)
        end = datetime.fromtimestamp(now_epoch + server_offset_seconds, tz=timezone.utc)
        ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return None

        tick_frame = pd.DataFrame(ticks)
        if "time" not in tick_frame.columns:
            return None
        raw_times = pd.to_numeric(tick_frame["time"], errors="coerce")
        bid = pd.to_numeric(tick_frame.get("bid"), errors="coerce")
        if bid is None:
            return None
        price = bid.where(bid > 0)
        if "last" in tick_frame.columns:
            last = pd.to_numeric(tick_frame["last"], errors="coerce")
            price = price.fillna(last.where(last > 0))
        tick_frame["_raw_time"] = raw_times
        tick_frame["_price"] = price
        tick_frame = tick_frame.loc[
            tick_frame["_raw_time"].notna()
            & tick_frame["_price"].notna()
            & (tick_frame["_raw_time"] >= rebuild_start_raw)
            & (tick_frame["_raw_time"] < raw_current_bar_open)
        ].copy()
        if tick_frame.empty:
            return None

        tick_frame["_bucket_raw"] = (
            tick_frame["_raw_time"].astype("int64") // timeframe_seconds
        ) * timeframe_seconds
        if "time_msc" in tick_frame.columns:
            tick_frame = tick_frame.sort_values("time_msc")
        else:
            tick_frame = tick_frame.sort_values("_raw_time")

        info = mt5.symbol_info(symbol)
        point = float(getattr(info, "point", 0.0) or 0.0)
        if point > 0 and "ask" in tick_frame.columns and "bid" in tick_frame.columns:
            ask = pd.to_numeric(tick_frame["ask"], errors="coerce")
            spread_points = ((ask - bid) / point).clip(lower=0)
            tick_frame["_spread_points"] = spread_points
        else:
            tick_frame["_spread_points"] = 0.0
        real_volume_column = "volume_real" if "volume_real" in tick_frame.columns else "volume"
        if real_volume_column in tick_frame.columns:
            tick_frame["_real_volume"] = pd.to_numeric(
                tick_frame[real_volume_column], errors="coerce"
            ).fillna(0.0)
        else:
            tick_frame["_real_volume"] = 0.0

        grouped = tick_frame.groupby("_bucket_raw", sort=True)
        rebuilt = pd.DataFrame({
            "time_raw": grouped["_bucket_raw"].first(),
            "open": grouped["_price"].first(),
            "high": grouped["_price"].max(),
            "low": grouped["_price"].min(),
            "close": grouped["_price"].last(),
            "tick_volume": grouped.size(),
            "spread": grouped["_spread_points"].median().round(),
            "real_volume": grouped["_real_volume"].sum(),
        }).reset_index(drop=True)
        if rebuilt.empty or int(rebuilt.iloc[-1]["time_raw"]) != expected_latest_raw:
            return None
        rebuilt["time"] = pd.to_datetime(
            rebuilt.pop("time_raw") - server_offset_seconds,
            unit="s",
            utc=True,
        )

        rebuild_first = rebuilt.iloc[0]["time"]
        base = broker_frame.loc[broker_frame["time"] < rebuild_first].copy()
        merged = pd.concat([base, rebuilt], ignore_index=True, sort=False)
        merged = (
            merged.drop_duplicates(subset="time", keep="last")
            .sort_values("time")
            .tail(count)
            .reset_index(drop=True)
        )
        latest_open = int(merged.iloc[-1]["time"].timestamp())
        age_after_close = max(
            0.0,
            now_epoch - (latest_open + timeframe_seconds),
        )
        merged.attrs.update(
            source="tick_rebuild",
            timeframe=timeframe,
            latest_bar_utc=merged.iloc[-1]["time"].isoformat(),
            age_after_close_seconds=age_after_close,
            is_stale=latest_open < (
                raw_current_bar_open - server_offset_seconds - timeframe_seconds
            ),
            server_offset_seconds=server_offset_seconds,
            rebuilt_from_ticks=True,
            pip_size=pip_size(info),
            asset_class=instrument_asset_class(symbol, info),
        )
        if merged.attrs["is_stale"]:
            return None
        self._tick_rebuild_cache[cache_key] = merged.copy(deep=True)
        latest_bar = str(merged.attrs["latest_bar_utc"])
        if self._last_tick_rebuild_bar.get(cache_key) != latest_bar:
            logger.warning(
                "Rebuilt frozen %s candle history for %s from broker ticks through %s.",
                timeframe,
                symbol,
                latest_bar,
            )
            self._last_tick_rebuild_bar[cache_key] = latest_bar
        return merged

    async def get_ohlcv(self, symbol: str, timeframe_str: str, count: int = 100) -> Optional[pd.DataFrame]:
        """
        Asynchronously fetches the latest completed historical candles for a symbol.
        """
        if not await self._conn.is_connected():
            logger.warning("MT5 connection is offline. Cannot retrieve OHLCV data.")
            return None

        timeframe = timeframe_str.upper()
        tf, timeframe_seconds = _TIMEFRAMES.get(timeframe, _TIMEFRAMES["M5"])

        def _fetch() -> Optional[pd.DataFrame]:
            # Make sure symbol is added to Market Watch
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to add symbol '{symbol}' to Market Watch: {mt5.last_error()}")
                return None
                
            # Pepperstone's terminal can expose server-wall-clock epochs (GMT+3
            # or GMT+2) even though the Python contract calls them UTC. Detect
            # that offset from the live tick, query at broker-now, then normalize
            # every returned timestamp to real UTC.
            now_utc = _utc_now()
            now_epoch = now_utc.timestamp()
            tick = mt5.symbol_info_tick(symbol)
            instrument = mt5.symbol_info(symbol)
            broker_pip = pip_size(instrument) if instrument is not None else 0.0
            server_offset_seconds = self._server_offset_seconds(symbol, tick, now_epoch)
            broker_now_epoch = now_epoch + server_offset_seconds
            broker_now = datetime.fromtimestamp(broker_now_epoch, tz=timezone.utc)
            raw_current_bar_open = (
                int(broker_now_epoch) // timeframe_seconds * timeframe_seconds
            )

            def _prepare(rates, source: str) -> Optional[pd.DataFrame]:
                if rates is None or len(rates) == 0:
                    return None
                frame = pd.DataFrame(rates)
                if "time" not in frame.columns:
                    logger.warning(
                        "MT5 returned malformed candle data for %s on %s (source=%s).",
                        symbol,
                        timeframe,
                        source,
                    )
                    return None
                # Keep only bars whose full timeframe has closed. This protects
                # the LLM from both the changing bar and future-stamped cache rows.
                numeric_time = pd.to_numeric(frame["time"], errors="coerce")
                frame = frame.loc[numeric_time < raw_current_bar_open].copy()
                if frame.empty:
                    return None
                normalized_time = (
                    pd.to_numeric(frame["time"], errors="coerce")
                    - server_offset_seconds
                )
                frame["time"] = pd.to_datetime(normalized_time, unit="s", utc=True)
                frame = (
                    frame.drop_duplicates(subset="time")
                    .sort_values("time")
                    .tail(count)
                    .reset_index(drop=True)
                )
                latest_open = int(frame.iloc[-1]["time"].timestamp())
                expected_latest_open = (
                    raw_current_bar_open
                    - server_offset_seconds
                    - timeframe_seconds
                )
                age_after_close = max(
                    0.0,
                    now_epoch - (latest_open + timeframe_seconds),
                )
                frame.attrs.update(
                    pip_size=broker_pip,
                    asset_class=instrument_asset_class(symbol, instrument),
                    source=source,
                    timeframe=timeframe,
                    latest_bar_utc=frame.iloc[-1]["time"].isoformat(),
                    age_after_close_seconds=age_after_close,
                    is_stale=latest_open < expected_latest_open,
                    server_offset_seconds=server_offset_seconds,
                )
                return frame

            broker_rates = mt5.copy_rates_from(symbol, tf, broker_now, count + 2)
            df = _prepare(broker_rates, "broker_clock")

            # Pepperstone's datetime-indexed cache can remain populated but
            # frozen while ticks continue. A populated stale result therefore
            # needs the same positional recovery as an empty result.
            if df is None or bool(df.attrs.get("is_stale", False)):
                positional_rates = mt5.copy_rates_from_pos(symbol, tf, 0, count + 2)
                positional = _prepare(positional_rates, "position_fallback")
                if positional is not None:
                    positional_time = positional.iloc[-1]["time"]
                    utc_time = df.iloc[-1]["time"] if df is not None else None
                    if utc_time is None or positional_time > utc_time:
                        logger.info(
                            "Using fresher positional %s candles for %s (%s > %s).",
                            timeframe,
                            symbol,
                            positional_time,
                            utc_time if utc_time is not None else "no UTC data",
                        )
                        df = positional

            broker_history_stale = bool(df is not None and df.attrs.get("is_stale", False))
            symbol_key = symbol.upper()
            if timeframe == "M5":
                if broker_history_stale:
                    self._frozen_history_symbols.add(symbol_key)
                else:
                    self._frozen_history_symbols.discard(symbol_key)

            if df is not None and (
                broker_history_stale or symbol_key in self._frozen_history_symbols
            ):
                rebuilt = self._rebuild_stale_history_from_ticks(
                    symbol=symbol,
                    timeframe=timeframe,
                    timeframe_seconds=timeframe_seconds,
                    count=count,
                    broker_frame=df,
                    raw_current_bar_open=raw_current_bar_open,
                    server_offset_seconds=server_offset_seconds,
                    now_epoch=now_epoch,
                    tick=tick,
                )
                if rebuilt is not None:
                    df = rebuilt

            if df is None:
                logger.error(
                    "Failed to copy completed rates for %s on %s: %s",
                    symbol,
                    timeframe,
                    mt5.last_error(),
                )
                return None

            if df.attrs["is_stale"]:
                stale_key = (symbol.upper(), timeframe)
                latest_bar = str(df.attrs["latest_bar_utc"])
                if self._last_stale_bar.get(stale_key) != latest_bar:
                    logger.warning(
                        "Stale %s candles for %s: latest completed bar=%s, age after close=%.0fs.",
                        timeframe,
                        symbol,
                        latest_bar,
                        float(df.attrs["age_after_close_seconds"]),
                    )
                    self._last_stale_bar[stale_key] = latest_bar
            else:
                self._last_stale_bar.pop((symbol.upper(), timeframe), None)
                if df.attrs.get("source") != "tick_rebuild":
                    self._tick_rebuild_cache.pop((symbol.upper(), timeframe), None)
            return df

        return await asyncio.to_thread(_fetch)

    async def get_live_tick(
        self,
        symbol: str,
        assume_connected: bool = False,
    ) -> Optional[dict]:
        """
        Queries the current ask/bid price tick information for a symbol.

        High-frequency callers may set ``assume_connected`` only after they
        have completed one connection heartbeat for the surrounding batch.
        This avoids repeating a blocking terminal-status call for every symbol
        while preserving the safe default for entry and protection callers.
        """
        if not assume_connected and not await self._conn.is_connected():
            logger.warning("MT5 connection is offline. Cannot retrieve tick data.")
            return None

        def _fetch_tick() -> Optional[dict]:
            if not mt5.symbol_select(symbol, True):
                return None
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            now_epoch = _utc_now().timestamp()
            server_offset_seconds = self._server_offset_seconds(symbol, tick, now_epoch)
            age_seconds = broker_tick_age_seconds(
                float(tick.time),
                now_epoch=now_epoch,
                offset_seconds=server_offset_seconds,
                symbol=symbol,
            )
            if age_seconds > settings.max_tick_age_seconds:
                logger.debug(
                    "Ignoring stale tick for %s (age %.1fs > %.1fs).",
                    symbol,
                    age_seconds,
                    settings.max_tick_age_seconds,
                )
                return None
            return {
                "time": normalized_broker_epoch(tick.time, server_offset_seconds),
                "time_msc": (
                    float(getattr(tick, "time_msc", tick.time * 1000))
                    - server_offset_seconds * 1000
                ),
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume
            }

        return await asyncio.to_thread(_fetch_tick)

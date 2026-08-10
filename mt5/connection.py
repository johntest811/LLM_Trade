import asyncio
import logging
from mt5.safe_api import mt5, mt5_session
from typing import Dict, Any, Optional
from app_config.settings import settings

logger = logging.getLogger("TradingSystem.MT5Connection")

_TRADE_MODE_NAMES = {
    getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0): "DEMO",
    getattr(mt5, "ACCOUNT_TRADE_MODE_CONTEST", 1): "CONTEST",
    getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2): "LIVE",
}

class MT5ConnectionManager:
    """
    Manages the MetaTrader 5 terminal connection state, initialization, and heartbeat checks.
    """
    def __init__(self) -> None:
        self._connected: bool = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """
        Asynchronously initializes the MetaTrader 5 terminal connection.
        Runs blocking MT5 API initialization inside a thread pool.
        """
        async with self._lock:
            def _connect() -> bool:
                with mt5_session():
                    return self._connect_exclusive()

            success = await asyncio.to_thread(_connect)
            self._connected = success
            return success

    def _connect_exclusive(self) -> bool:
        """Connect while the process-wide native MT5 session gate is held."""
        # `_connected` is only a cache. Always verify the terminal before
        # taking the fast path; otherwise a dropped terminal can never
        # reinitialize because the stale flag remains True.
        if self._connected:
            terminal = mt5.terminal_info()
            if terminal is not None and bool(getattr(terminal, "connected", False)):
                return True
            logger.warning("Cached MT5 connection is stale; reinitializing terminal session.")
            mt5.shutdown()

        # 1. Initialize the terminal path if configured
        init_args: Dict[str, Any] = {}
        if settings.mt5_path:
            init_args["path"] = settings.mt5_path

        logger.info("Initializing MetaTrader 5 connection...")
        if not mt5.initialize(**init_args):
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # 2. Login to specific account if credentials are configured
        if settings.mt5_account:
            logger.info(f"Logging in to MT5 account {settings.mt5_account} on server {settings.mt5_server}...")
            login_ok = mt5.login(
                login=settings.mt5_account,
                password=settings.mt5_password,
                server=settings.mt5_server
            )
            if not login_ok:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            logger.info("MT5 logged in successfully.")
        else:
            logger.info("No credentials provided. MT5 will use the currently logged-in account in the terminal.")

        # 3. Verify account connection details
        account_info = mt5.account_info()
        if account_info is None:
            logger.error(f"Failed to query account info after connection: {mt5.last_error()}")
            mt5.shutdown()
            return False

        mode_name = _TRADE_MODE_NAMES.get(account_info.trade_mode, "UNKNOWN")
        if settings.expected_broker and settings.expected_broker.lower() not in (
            f"{account_info.company} {account_info.server}".lower()
        ):
            logger.error(
                "Broker mismatch: connected company/server does not match EXPECTED_BROKER=%s.",
                settings.expected_broker,
            )
            mt5.shutdown()
            return False

        logger.info(
            "Connected to %s account ending %s on '%s' (%s).",
            mode_name,
            str(account_info.login)[-4:],
            account_info.server,
            account_info.company,
        )
        return True

    async def shutdown(self) -> None:
        """
        Shuts down the MetaTrader 5 terminal connection safely.
        """
        async with self._lock:
            if not self._connected:
                return
                
            logger.info("Shutting down MetaTrader 5 terminal connection...")
            await asyncio.to_thread(mt5.shutdown)
            self._connected = False

    async def is_connected(self) -> bool:
        """
        Verifies if the connection to MT5 and the broker server is active.
        """
        if not self._connected:
            return False

        def _check() -> bool:
            # Query terminal status to verify live broker data stream
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return False
            # Check if connected to trading server
            return bool(terminal_info.connected)

        connected = await asyncio.to_thread(_check)
        if not connected:
            # Permit initialize() to perform a real reconnect on the next call.
            self._connected = False
        return connected

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Queries the current account properties (balance, equity, free margin).
        """
        if not await self.is_connected():
            logger.warning("MT5 connection is offline. Attempting quick reconnection...")
            if not await self.initialize():
                return None

        def _get_info() -> Optional[Dict[str, Any]]:
            info = mt5.account_info()
            if info is None:
                return None
            terminal = mt5.terminal_info()
            return {
                "login": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "profit": info.profit,
                "margin": info.margin,
                "margin_free": info.margin_free,
                "margin_level": info.margin_level,
                "leverage": info.leverage,
                "currency": info.currency,
                "company": info.company,
                "server": info.server,
                "trade_mode": info.trade_mode,
                "trade_mode_name": _TRADE_MODE_NAMES.get(info.trade_mode, "UNKNOWN"),
                "account_trade_allowed": bool(getattr(info, "trade_allowed", False)),
                "expert_trading_allowed": bool(getattr(info, "trade_expert", False)),
                "terminal_connected": bool(getattr(terminal, "connected", False)),
                "terminal_trade_allowed": bool(getattr(terminal, "trade_allowed", False)),
                "tradeapi_disabled": bool(getattr(terminal, "tradeapi_disabled", True)),
            }

        return await asyncio.to_thread(_get_info)

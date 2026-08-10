import os
import sqlite3
import json
import logging
import asyncio
from contextlib import closing
from typing import List, Dict, Any, Optional

from app_config.paths import DEFAULT_DB_PATH

logger = logging.getLogger("TradingSystem.Storage")

class TradingDatabase:
    """
    Manages persistent SQLite storage for trade records and candle caches.
    Runs database writes in a worker thread pool.
    """
    def __init__(self, db_path: str = str(DEFAULT_DB_PATH)) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
            self._init_schema(conn)

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        """Create or migrate the schema using an already-owned connection."""
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        cursor = conn.cursor()
        
        # 1. Table for trade transaction records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                ticket INTEGER PRIMARY KEY,
                time TEXT,
                symbol TEXT,
                action TEXT,
                lot_size REAL,
                price REAL,
                sl REAL,
                tp REAL,
                profit REAL,
                reasoning TEXT,
                status TEXT
            )
        """)

        # 2. Table for persistent key-value caching (e.g. last candle cache)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_value_cache (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # Broker-confirmed, fully reconstructed positions. This table is the
        # source of truth for performance and daily-loss controls.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS closed_positions (
                account_login INTEGER NOT NULL DEFAULT 0,
                position_id INTEGER NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                volume REAL NOT NULL,
                open_price REAL NOT NULL,
                close_price REAL NOT NULL,
                gross_profit REAL NOT NULL,
                commission REAL NOT NULL,
                swap REAL NOT NULL,
                fee REAL NOT NULL,
                net_profit REAL NOT NULL,
                magic INTEGER NOT NULL,
                close_reason TEXT,
                profit_pips REAL NOT NULL DEFAULT 0,
                initial_risk_pips REAL NOT NULL DEFAULT 0,
                initial_risk_usd REAL NOT NULL DEFAULT 0,
                mfe_pips REAL NOT NULL DEFAULT 0,
                mae_pips REAL NOT NULL DEFAULT 0,
                mfe_usd REAL NOT NULL DEFAULT 0,
                mae_usd REAL NOT NULL DEFAULT 0,
                rr_achieved REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (account_login, position_id)
            )
        """)

        # Immutable per-position risk baselines.  Protective SL changes must
        # never redefine one R after a restart, so the first accepted row wins
        # for an account identity + MT5 position ticket.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_risk_baselines (
                account_scope TEXT NOT NULL,
                ticket INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                initial_sl REAL NOT NULL,
                initial_risk_pips REAL NOT NULL,
                initial_risk_usd REAL NOT NULL,
                initial_volume REAL NOT NULL,
                mfe_pips REAL NOT NULL DEFAULT 0,
                mae_pips REAL NOT NULL DEFAULT 0,
                mfe_usd REAL NOT NULL DEFAULT 0,
                mae_usd REAL NOT NULL DEFAULT 0,
                excursion_updated_at TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (account_scope, ticket)
            )
        """)

        # Migrate the original single-account schema without losing history.
        # Legacy rows remain under account 0; all new reconciliations are scoped
        # to the active MT5 login so demo/live histories cannot contaminate one
        # another or collide on position IDs.
        columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(closed_positions)").fetchall()
        }
        if "account_login" not in columns:
            cursor.execute("ALTER TABLE closed_positions RENAME TO closed_positions_legacy")
            cursor.execute("""
                CREATE TABLE closed_positions (
                    account_login INTEGER NOT NULL DEFAULT 0,
                    position_id INTEGER NOT NULL,
                    open_time TEXT NOT NULL,
                    close_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    gross_profit REAL NOT NULL,
                    commission REAL NOT NULL,
                    swap REAL NOT NULL,
                    fee REAL NOT NULL,
                    net_profit REAL NOT NULL,
                    magic INTEGER NOT NULL,
                    close_reason TEXT,
                    profit_pips REAL NOT NULL DEFAULT 0,
                    initial_risk_pips REAL NOT NULL DEFAULT 0,
                    initial_risk_usd REAL NOT NULL DEFAULT 0,
                    mfe_pips REAL NOT NULL DEFAULT 0,
                    mae_pips REAL NOT NULL DEFAULT 0,
                    mfe_usd REAL NOT NULL DEFAULT 0,
                    mae_usd REAL NOT NULL DEFAULT 0,
                    rr_achieved REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (account_login, position_id)
                )
            """)
            cursor.execute("""
                INSERT INTO closed_positions (
                    account_login, position_id, open_time, close_time, symbol,
                    direction, volume, open_price, close_price, gross_profit,
                    commission, swap, fee, net_profit, magic, close_reason
                )
                SELECT 0, position_id, open_time, close_time, symbol, direction,
                       volume, open_price, close_price, gross_profit, commission,
                       swap, fee, net_profit, magic, close_reason
                FROM closed_positions_legacy
            """)
            cursor.execute("DROP TABLE closed_positions_legacy")

        closed_metric_columns = {
            "profit_pips": "REAL NOT NULL DEFAULT 0",
            "initial_risk_pips": "REAL NOT NULL DEFAULT 0",
            "initial_risk_usd": "REAL NOT NULL DEFAULT 0",
            "mfe_pips": "REAL NOT NULL DEFAULT 0",
            "mae_pips": "REAL NOT NULL DEFAULT 0",
            "mfe_usd": "REAL NOT NULL DEFAULT 0",
            "mae_usd": "REAL NOT NULL DEFAULT 0",
            "rr_achieved": "REAL NOT NULL DEFAULT 0",
        }
        closed_columns = {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(closed_positions)"
            ).fetchall()
        }
        for name, declaration in closed_metric_columns.items():
            if name not in closed_columns:
                cursor.execute(
                    f"ALTER TABLE closed_positions ADD COLUMN {name} {declaration}"
                )

        baseline_metric_columns = {
            "mfe_pips": "REAL NOT NULL DEFAULT 0",
            "mae_pips": "REAL NOT NULL DEFAULT 0",
            "mfe_usd": "REAL NOT NULL DEFAULT 0",
            "mae_usd": "REAL NOT NULL DEFAULT 0",
            "excursion_updated_at": "TEXT",
        }
        baseline_columns = {
            row[1]
            for row in cursor.execute(
                "PRAGMA table_info(position_risk_baselines)"
            ).fetchall()
        }
        for name, declaration in baseline_metric_columns.items():
            if name not in baseline_columns:
                cursor.execute(
                    "ALTER TABLE position_risk_baselines "
                    f"ADD COLUMN {name} {declaration}"
                )

        conn.commit()

    async def log_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Asynchronously writes a completed or active trade record into the database.
        """
        def _write() -> bool:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trade_history 
                    (ticket, time, symbol, action, lot_size, price, sl, tp, profit, reasoning, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get("ticket"),
                    trade_data.get("time"),
                    trade_data.get("symbol"),
                    trade_data.get("action"),
                    trade_data.get("lot_size"),
                    trade_data.get("price"),
                    trade_data.get("sl"),
                    trade_data.get("tp"),
                    trade_data.get("profit", 0.0),
                    trade_data.get("reasoning", ""),
                    trade_data.get("status", "OPEN")
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to log trade to SQLite: {e}")
                return False
            finally:
                if conn is not None:
                    conn.close()

        return await asyncio.to_thread(_write)

    async def get_trade_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves historical trades from the database.
        """
        def _read() -> List[Dict[str, Any]]:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if symbol:
                    cursor.execute(
                        "SELECT * FROM trade_history WHERE symbol = ? ORDER BY time DESC LIMIT ?", 
                        (symbol, limit)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM trade_history ORDER BY time DESC LIMIT ?", 
                        (limit,)
                    )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to query trade history: {e}")
                return []
            finally:
                if conn is not None:
                    conn.close()

        return await asyncio.to_thread(_read)

    async def get_trade_record(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Return the local execution record for one ticket, if available."""
        def _read() -> Optional[Dict[str, Any]]:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    "SELECT * FROM trade_history WHERE ticket=?",
                    (int(ticket),),
                ).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()

        return await asyncio.to_thread(_read)

    async def set_position_risk_baseline(
        self, account_scope: str, baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Persist an immutable initial-risk row and return the canonical row.

        ``INSERT OR IGNORE`` is deliberate: later trailing/break-even updates
        cannot overwrite the original risk even if two observations race.
        """
        def _write() -> Dict[str, Any]:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO position_risk_baselines (
                        account_scope, ticket, symbol, direction, entry_price,
                        initial_sl, initial_risk_pips, initial_risk_usd,
                        initial_volume, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                    """,
                    (
                        str(account_scope),
                        int(baseline["ticket"]),
                        str(baseline["symbol"]),
                        str(baseline["direction"]),
                        float(baseline["entry_price"]),
                        float(baseline["initial_sl"]),
                        float(baseline["initial_risk_pips"]),
                        float(baseline.get("initial_risk_usd", 0.0)),
                        float(baseline.get("initial_volume", 0.0)),
                        baseline.get("created_at"),
                    ),
                )
                conn.commit()
                row = conn.execute(
                    """
                    SELECT * FROM position_risk_baselines
                    WHERE account_scope=? AND ticket=?
                    """,
                    (str(account_scope), int(baseline["ticket"])),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Risk baseline insert did not produce a row")
                return dict(row)
            finally:
                conn.close()

        return await asyncio.to_thread(_write)

    async def get_position_risk_baseline(
        self, account_scope: str, ticket: int
    ) -> Optional[Dict[str, Any]]:
        def _read() -> Optional[Dict[str, Any]]:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT * FROM position_risk_baselines
                    WHERE account_scope=? AND ticket=?
                    """,
                    (str(account_scope), int(ticket)),
                ).fetchone()
                return dict(row) if row is not None else None
            finally:
                conn.close()

        return await asyncio.to_thread(_read)

    async def get_position_risk_baselines(self, account_scope: str) -> List[Dict[str, Any]]:
        """Load immutable risk rows for one exact login/server/mode identity."""
        def _read() -> List[Dict[str, Any]]:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM position_risk_baselines
                    WHERE account_scope=? ORDER BY created_at, ticket
                    """,
                    (str(account_scope),),
                ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

        return await asyncio.to_thread(_read)

    async def update_position_excursion(
        self,
        account_scope: str,
        ticket: int,
        *,
        profit_pips: float,
        profit_usd: float,
        observed_at: str,
    ) -> bool:
        """Persist MFE/MAE only when a monitored position sets a new extreme."""
        def _write() -> bool:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                cursor = conn.execute(
                    """
                    UPDATE position_risk_baselines
                    SET mfe_pips=MAX(mfe_pips, ?, 0),
                        mae_pips=MIN(mae_pips, ?, 0),
                        mfe_usd=MAX(mfe_usd, ?, 0),
                        mae_usd=MIN(mae_usd, ?, 0),
                        excursion_updated_at=?
                    WHERE account_scope=? AND ticket=?
                    """,
                    (
                        float(profit_pips),
                        float(profit_pips),
                        float(profit_usd),
                        float(profit_usd),
                        str(observed_at),
                        str(account_scope),
                        int(ticket),
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        return await asyncio.to_thread(_write)

    async def upsert_closed_positions(
        self, positions: List[Dict[str, Any]], account_login: int
    ) -> int:
        """Persist broker-reconciled closed positions idempotently."""
        if not positions:
            return 0

        def _write() -> int:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                conn.executemany("""
                    INSERT INTO closed_positions (
                        account_login, position_id, open_time, close_time, symbol, direction,
                        volume, open_price, close_price, gross_profit, commission,
                        swap, fee, net_profit, magic, close_reason, profit_pips,
                        initial_risk_pips, initial_risk_usd, mfe_pips, mae_pips,
                        mfe_usd, mae_usd, rr_achieved
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_login, position_id) DO UPDATE SET
                        close_time=excluded.close_time,
                        close_price=excluded.close_price,
                        gross_profit=excluded.gross_profit,
                        commission=excluded.commission,
                        swap=excluded.swap,
                        fee=excluded.fee,
                        net_profit=excluded.net_profit,
                        close_reason=excluded.close_reason,
                        profit_pips=excluded.profit_pips,
                        initial_risk_pips=excluded.initial_risk_pips,
                        initial_risk_usd=excluded.initial_risk_usd,
                        mfe_pips=excluded.mfe_pips,
                        mae_pips=excluded.mae_pips,
                        mfe_usd=excluded.mfe_usd,
                        mae_usd=excluded.mae_usd,
                        rr_achieved=excluded.rr_achieved
                """, [
                    (
                        int(account_login), p["position_id"], p["open_time"], p["close_time"], p["symbol"],
                        p["direction"], p["volume"], p["open_price"], p["close_price"],
                        p["gross_profit"], p["commission"], p["swap"], p["fee"],
                        p["net_profit"], p["magic"], p.get("close_reason", ""),
                        p.get("profit_pips", 0.0),
                        p.get("initial_risk_pips", 0.0),
                        p.get("initial_risk_usd", 0.0),
                        p.get("mfe_pips", 0.0),
                        p.get("mae_pips", 0.0),
                        p.get("mfe_usd", 0.0),
                        p.get("mae_usd", 0.0),
                        p.get("rr_achieved", 0.0),
                    )
                    for p in positions
                ])
                conn.commit()
                return len(positions)
            finally:
                conn.close()

        return await asyncio.to_thread(_write)

    async def get_closed_positions(
        self,
        symbol: Optional[str] = None,
        limit: int = 200,
        account_login: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        def _read() -> List[Dict[str, Any]]:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            try:
                if symbol and account_login is not None:
                    rows = conn.execute(
                        "SELECT * FROM closed_positions WHERE symbol=? AND account_login=? ORDER BY close_time DESC LIMIT ?",
                        (symbol, int(account_login), limit),
                    ).fetchall()
                elif symbol:
                    rows = conn.execute(
                        "SELECT * FROM closed_positions WHERE symbol=? ORDER BY close_time DESC LIMIT ?",
                        (symbol, limit),
                    ).fetchall()
                elif account_login is not None:
                    rows = conn.execute(
                        "SELECT * FROM closed_positions WHERE account_login=? ORDER BY close_time DESC LIMIT ?",
                        (int(account_login), limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM closed_positions ORDER BY close_time DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

        return await asyncio.to_thread(_read)

    async def set_cache(self, key: str, data: Any) -> bool:
        """
        Asynchronously stores serialized JSON data in the key-value cache.
        """
        def _write_cache() -> bool:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO key_value_cache (key, value) VALUES (?, ?)",
                    (key, json.dumps(data))
                )
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to set database cache for key '{key}': {e}")
                return False
            finally:
                if conn is not None:
                    conn.close()

        return await asyncio.to_thread(_write_cache)

    async def get_cache(self, key: str) -> Optional[Any]:
        """
        Retrieves and deserializes JSON data from the key-value cache.
        """
        def _read_cache() -> Optional[Any]:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM key_value_cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            except Exception as e:
                logger.error(f"Failed to read database cache for key '{key}': {e}")
                return None
            finally:
                if conn is not None:
                    conn.close()

        return await asyncio.to_thread(_read_cache)

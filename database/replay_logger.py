"""
database/replay_logger.py — Trade Replay Logger

Manages the storage of trade replay data, including full inputs (prompts, indicators, structures)
and outputs (LLM decisions, scoring, outcomes) to allow exact strategy auditing and replay analysis.
"""
import json
import sqlite3
import logging
import re
from contextlib import closing
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

from app_config.paths import DEFAULT_DB_PATH
from app_config.settings import settings

logger = logging.getLogger("TradingSystem.ReplayLogger")


class TradeReplayLogger:
    """
    Saves comprehensive snapshots of all evaluation cycles (including prompts,
    signals, models choices, scores, and final trade outcomes).
    """

    def __init__(self, db_path: str = str(DEFAULT_DB_PATH)) -> None:
        self.db_path = str(db_path)
        self._init_table()

    def _init_table(self) -> None:
        """Creates the trade_replay audit table if it does not exist."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_observation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at_utc TEXT NOT NULL,
                    account_scope TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    candle_time TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_observation_account_symbol
                ON scan_observation(account_scope, symbol, id DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_observation_time
                ON scan_observation(observed_at_utc)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_replay (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    prompt_text TEXT,
                    llm_json TEXT,
                    indicators TEXT,
                    market_structure TEXT,
                    entry REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    risk_percent REAL,
                    quality_score REAL,
                    confluence_score REAL,
                    confidence REAL,
                    reasoning TEXT,
                    status TEXT,
                    ticket INTEGER,
                    close_time TEXT,
                    close_reason TEXT,
                    mfe_pips REAL,
                    mae_pips REAL,
                    mfe_usd REAL,
                    mae_usd REAL,
                    rr_achieved REAL
                )
            """)
            replay_columns = {
                row[1]
                for row in cursor.execute(
                    "PRAGMA table_info(trade_replay)"
                ).fetchall()
            }
            for name, declaration in {
                "close_reason": "TEXT",
                "mfe_pips": "REAL",
                "mae_pips": "REAL",
                "mfe_usd": "REAL",
                "mae_usd": "REAL",
                "rr_achieved": "REAL",
                "config_fingerprint": "TEXT",
            }.items():
                if name not in replay_columns:
                    cursor.execute(
                        f"ALTER TABLE trade_replay ADD COLUMN {name} {declaration}"
                    )
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_trace (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL UNIQUE,
                    time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    candle_time TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    latency_seconds REAL NOT NULL,
                    success INTEGER NOT NULL,
                    request_id TEXT,
                    prompt_sha256 TEXT,
                    prompt_tokens_estimate INTEGER,
                    action TEXT,
                    confidence REAL,
                    validation_status TEXT,
                    error TEXT,
                    response_json TEXT
                )
            """)
            decision_columns = {
                row[1]
                for row in cursor.execute(
                    "PRAGMA table_info(decision_trace)"
                ).fetchall()
            }
            if "config_fingerprint" not in decision_columns:
                cursor.execute(
                    "ALTER TABLE decision_trace "
                    "ADD COLUMN config_fingerprint TEXT"
                )
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_decision_trace_symbol_time
                ON decision_trace(symbol, time DESC)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shadow_outcome (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    candle_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    risk_distance REAL NOT NULL,
                    rejection_stage TEXT NOT NULL,
                    rejection_reason TEXT,
                    horizon_minutes REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    resolved_at_utc TEXT,
                    exit_price REAL,
                    outcome_r REAL,
                    mfe_r REAL,
                    mae_r REAL,
                    UNIQUE(symbol, action, candle_time, rejection_stage)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_shadow_outcome_status_time
                ON shadow_outcome(status, created_at_utc)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exit_counterfactual (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_login INTEGER NOT NULL,
                    ticket INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    exit_time_utc TEXT NOT NULL,
                    entry REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    actual_exit_price REAL NOT NULL,
                    realized_r REAL NOT NULL,
                    horizon_minutes REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    resolved_at_utc TEXT,
                    counterfactual_exit_price REAL,
                    outcome_r REAL,
                    delta_vs_realized_r REAL,
                    post_exit_mfe_r REAL,
                    post_exit_mae_r REAL,
                    UNIQUE(account_login, ticket, horizon_minutes)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exit_counterfactual_status_time
                ON exit_counterfactual(status, exit_time_utc)
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Error initializing trade_replay table: {e}")
        finally:
            if conn is not None:
                conn.close()

    def log_scan_observations(
        self, account_scope: str, lane: str, observations: List[Dict[str, Any]]
    ) -> bool:
        """Batch account-scoped scan reasons, including candidates never sent to an LLM."""
        if not observations:
            return True
        now = datetime.now(timezone.utc)
        try:
            values = [(
                now.isoformat(), account_scope, str(row["symbol"]).upper(),
                str(row.get("opportunity_bar") or "UNAVAILABLE"), lane,
                settings.config_fingerprint,
                json.dumps(row, allow_nan=False, default=str),
            ) for row in observations]
            with closing(sqlite3.connect(self.db_path, timeout=1.0)) as conn:
                conn.executemany("""
                    INSERT INTO scan_observation (
                        observed_at_utc, account_scope, symbol, candle_time,
                        lane, config_fingerprint, snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, values)
                # Bound only the new diagnostic history. Never prune trades,
                # broker records, model decisions, or shadow outcomes here.
                cutoff = (now - timedelta(days=settings.scan_audit_retention_days)).isoformat()
                conn.execute("DELETE FROM scan_observation WHERE observed_at_utc < ?", (cutoff,))
                conn.execute("""
                    DELETE FROM scan_observation WHERE id <= (
                        SELECT id FROM scan_observation ORDER BY id DESC LIMIT 1 OFFSET ?
                    )
                """, (settings.scan_audit_max_rows,))
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("Scan-audit write failed: %s", exc)
            return False

    def get_scan_observations(
        self, account_scope: str, symbol: str = "", limit: int = 50,
        before_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read diagnostic history for one exact account, newest first."""
        limit = max(1, min(200, int(limit)))
        clause = " AND symbol = ?" if symbol else ""
        params = [account_scope, symbol.upper()] if symbol else [account_scope]
        if before_id is not None:
            if int(before_id) <= 0:
                raise ValueError("before_id must be positive")
            clause += " AND id < ?"
            params.append(int(before_id))
        params.append(limit)
        from pathlib import Path
        uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, observed_at_utc, symbol, candle_time, lane, "
                "config_fingerprint, snapshot_json FROM scan_observation "
                "WHERE account_scope = ?" + clause + " ORDER BY id DESC LIMIT ?", params,
            ).fetchall()
        result = []
        import hashlib
        scope_id = hashlib.sha256(account_scope.encode("utf-8")).hexdigest()[:16]
        for row in rows:
            item = dict(row)
            item["account_scope_id"] = scope_id
            item["snapshot"] = json.loads(item.pop("snapshot_json"))
            result.append(item)
        return result

    def log_shadow_candidate(
        self,
        *,
        symbol: str,
        action: str,
        candle_time: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        rejection_stage: str,
        rejection_reason: str,
        horizon_minutes: float,
        created_at_utc: Optional[str] = None,
    ) -> Optional[int]:
        """Record one non-executing rejected signal for later broker-bar replay."""
        try:
            risk_distance = abs(float(entry) - float(stop_loss))
            if risk_distance <= 0:
                return None
            created = created_at_utc or datetime.now(timezone.utc).isoformat()
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.execute("PRAGMA busy_timeout=10000")
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO shadow_outcome (
                        created_at_utc, candle_time, symbol, action, entry,
                        stop_loss, take_profit, risk_distance, rejection_stage,
                        rejection_reason, horizon_minutes, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (
                        created,
                        str(candle_time),
                        str(symbol).upper(),
                        str(action).upper(),
                        float(entry),
                        float(stop_loss),
                        float(take_profit),
                        risk_distance,
                        str(rejection_stage).upper(),
                        str(rejection_reason)[:2000],
                        float(horizon_minutes),
                    ),
                )
                conn.commit()
                if cursor.rowcount:
                    return int(cursor.lastrowid)
                row = conn.execute(
                    """
                    SELECT id FROM shadow_outcome
                    WHERE symbol = ? AND action = ? AND candle_time = ?
                      AND rejection_stage = ?
                    """,
                    (
                        str(symbol).upper(),
                        str(action).upper(),
                        str(candle_time),
                        str(rejection_stage).upper(),
                    ),
                ).fetchone()
                return int(row[0]) if row else None
        except Exception as exc:
            logger.error("Error logging shadow candidate: %s", exc)
            return None

    def get_pending_shadow_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=10000")
                rows = conn.execute(
                    """
                    SELECT * FROM shadow_outcome
                    WHERE status = 'PENDING'
                    ORDER BY created_at_utc ASC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Error reading pending shadow candidates: %s", exc)
            return []

    def resolve_shadow_candidate(
        self,
        candidate_id: int,
        *,
        status: str,
        resolved_at_utc: str,
        exit_price: float,
        outcome_r: Optional[float],
        mfe_r: float,
        mae_r: float,
    ) -> bool:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.execute("PRAGMA busy_timeout=10000")
                cursor = conn.execute(
                    """
                    UPDATE shadow_outcome
                    SET status = ?, resolved_at_utc = ?, exit_price = ?,
                        outcome_r = ?, mfe_r = ?, mae_r = ?
                    WHERE id = ? AND status = 'PENDING'
                    """,
                    (
                        str(status).upper(),
                        str(resolved_at_utc),
                        float(exit_price),
                        None if outcome_r is None else float(outcome_r),
                        float(mfe_r),
                        float(mae_r),
                        int(candidate_id),
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1
        except Exception as exc:
            logger.error("Error resolving shadow candidate: %s", exc)
            return False

    def log_exit_candidates(
        self,
        *,
        account_login: int,
        trades: List[Dict[str, Any]],
        horizons_minutes: List[int],
    ) -> int:
        """Queue diagnostic original-SL/TP hold comparisons in one transaction."""
        generic_or_broker = {
            "", "CLIENT", "MOBILE", "WEB", "EXPERT", "OTHER", "UNKNOWN",
            "SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "STOP_OUT",
        }
        eligible = [
            trade for trade in trades
            if str(trade.get("close_reason", "")).strip().upper()
            not in generic_or_broker
        ]
        horizons = sorted({max(5, int(value)) for value in horizons_minutes})
        if int(account_login) <= 0 or not eligible or not horizons:
            return 0
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=10000")
                tickets = sorted({int(row.get("position_id", 0) or 0) for row in eligible})
                placeholders = ",".join("?" for _ in tickets)
                replay_rows = conn.execute(
                    f"""
                    SELECT id, ticket, stop_loss, take_profit
                    FROM trade_replay
                    WHERE ticket IN ({placeholders})
                    ORDER BY id DESC
                    """,
                    tickets,
                ).fetchall()
                plans: Dict[int, sqlite3.Row] = {}
                for replay in replay_rows:
                    plans.setdefault(int(replay["ticket"] or 0), replay)

                inserts = []
                now = datetime.now(timezone.utc)
                max_horizon = max(horizons)
                for trade in eligible:
                    ticket = int(trade.get("position_id", 0) or 0)
                    plan = plans.get(ticket)
                    if plan is None:
                        continue
                    try:
                        exit_time = datetime.fromisoformat(
                            str(trade.get("close_time", "")).replace("Z", "+00:00")
                        )
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.replace(tzinfo=timezone.utc)
                        else:
                            exit_time = exit_time.astimezone(timezone.utc)
                        # The M1 reader keeps a bounded recent window. Do not
                        # create permanently pending rows for old history.
                        if now - exit_time > timedelta(minutes=max_horizon + 120):
                            continue
                        entry = float(trade.get("open_price", 0.0) or 0.0)
                        stop = float(plan["stop_loss"] or 0.0)
                        target = float(plan["take_profit"] or 0.0)
                        actual_exit = float(trade.get("close_price", 0.0) or 0.0)
                        realized_r = float(trade.get("rr_achieved", 0.0) or 0.0)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if min(entry, stop, target, actual_exit) <= 0 or entry == stop:
                        continue
                    for horizon in horizons:
                        inserts.append((
                            int(account_login), ticket,
                            str(trade.get("symbol", "")).upper(),
                            str(trade.get("direction", "")).upper(),
                            str(trade.get("close_reason", "")).upper(),
                            exit_time.isoformat(), entry, stop, target,
                            actual_exit, realized_r, float(horizon),
                        ))
                if not inserts:
                    return 0
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO exit_counterfactual (
                        account_login, ticket, symbol, action, exit_reason,
                        exit_time_utc, entry, stop_loss, take_profit,
                        actual_exit_price, realized_r, horizon_minutes, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    inserts,
                )
                conn.commit()
                return conn.total_changes - before
        except Exception as exc:
            logger.error("Error logging exit counterfactuals: %s", exc)
            return 0

    def get_pending_exit_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=10000")
                rows = conn.execute(
                    """
                    SELECT * FROM exit_counterfactual
                    WHERE status='PENDING'
                    ORDER BY exit_time_utc ASC, horizon_minutes ASC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Error reading exit counterfactuals: %s", exc)
            return []

    def resolve_exit_candidate(
        self,
        candidate_id: int,
        *,
        status: str,
        resolved_at_utc: str,
        exit_price: float,
        outcome_r: Optional[float],
        delta_vs_realized_r: Optional[float],
        post_exit_mfe_r: float,
        post_exit_mae_r: float,
    ) -> bool:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.execute("PRAGMA busy_timeout=10000")
                cursor = conn.execute(
                    """
                    UPDATE exit_counterfactual
                    SET status=?, resolved_at_utc=?,
                        counterfactual_exit_price=?, outcome_r=?,
                        delta_vs_realized_r=?, post_exit_mfe_r=?,
                        post_exit_mae_r=?
                    WHERE id=? AND status='PENDING'
                    """,
                    (
                        str(status).upper(), str(resolved_at_utc),
                        float(exit_price),
                        None if outcome_r is None else float(outcome_r),
                        None if delta_vs_realized_r is None else float(delta_vs_realized_r),
                        float(post_exit_mfe_r), float(post_exit_mae_r),
                        int(candidate_id),
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1
        except Exception as exc:
            logger.error("Error resolving exit counterfactual: %s", exc)
            return False

    def exit_counterfactual_summary(self) -> Dict[str, Any]:
        """Summarize whether original-bracket holds beat actual strategy exits."""
        empty = {
            "exit_pending": 0,
            "exit_resolved": 0,
            "exit_held_better": 0,
            "exit_actual_better": 0,
            "exit_average_delta_r": 0.0,
            "exit_horizon_breakdown": [],
        }
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=10000")
                row = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) pending,
                        SUM(CASE WHEN status<>'PENDING' THEN 1 ELSE 0 END) resolved,
                        SUM(CASE WHEN delta_vs_realized_r > 0.05 THEN 1 ELSE 0 END) held_better,
                        SUM(CASE WHEN delta_vs_realized_r < -0.05 THEN 1 ELSE 0 END) actual_better,
                        AVG(delta_vs_realized_r) average_delta_r
                    FROM exit_counterfactual
                    """
                ).fetchone()
                horizons = conn.execute(
                    """
                    SELECT horizon_minutes,
                           COUNT(*) resolved,
                           AVG(delta_vs_realized_r) average_delta_r
                    FROM exit_counterfactual
                    WHERE status<>'PENDING'
                      AND delta_vs_realized_r IS NOT NULL
                    GROUP BY horizon_minutes
                    ORDER BY horizon_minutes
                    """
                ).fetchall()
            return {
                "exit_pending": int(row["pending"] or 0),
                "exit_resolved": int(row["resolved"] or 0),
                "exit_held_better": int(row["held_better"] or 0),
                "exit_actual_better": int(row["actual_better"] or 0),
                "exit_average_delta_r": round(
                    float(row["average_delta_r"] or 0.0), 3
                ),
                "exit_horizon_breakdown": [
                    {
                        "minutes": int(item["horizon_minutes"]),
                        "resolved": int(item["resolved"] or 0),
                        "average_delta_r": round(
                            float(item["average_delta_r"] or 0.0), 3
                        ),
                    }
                    for item in horizons
                ],
            }
        except Exception as exc:
            logger.error("Error summarizing exit counterfactuals: %s", exc)
            return empty

    def shadow_summary(self) -> Dict[str, Any]:
        """Return compact rejected-signal and direction-funnel diagnostics.

        The rolling directional view is intentionally diagnostic. It exposes
        whether BUY and SELL candidates are generated and approved at similar
        rates without allowing recent outcomes to mutate live risk settings.
        """
        evidence_window_hours = 48
        empty = {
            "pending": 0,
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "ambiguous": 0,
            "win_rate_pct": 0.0,
            "expectancy_r": 0.0,
            "gate_breakdown": [],
            "evidence_window_hours": evidence_window_hours,
            "direction_breakdown": [],
        }
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10.0)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout=10000")
                row = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS pending,
                        SUM(CASE WHEN status <> 'PENDING' THEN 1 ELSE 0 END) AS resolved,
                        SUM(CASE WHEN outcome_r > 0 THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN outcome_r < 0 THEN 1 ELSE 0 END) AS losses,
                        SUM(CASE WHEN status = 'AMBIGUOUS' THEN 1 ELSE 0 END) AS ambiguous,
                        AVG(outcome_r) AS expectancy_r
                    FROM shadow_outcome
                    """
                ).fetchone()
                cutoff = (
                    datetime.now(timezone.utc)
                    - timedelta(hours=evidence_window_hours)
                ).isoformat()
                recent_rows = conn.execute(
                    """
                    SELECT action, rejection_stage, rejection_reason, outcome_r
                    FROM shadow_outcome
                    WHERE created_at_utc >= ?
                      AND status <> 'PENDING'
                      AND outcome_r IS NOT NULL
                    """,
                    (cutoff,),
                ).fetchall()
                local_cutoff = (
                    datetime.now() - timedelta(hours=evidence_window_hours)
                ).strftime("%Y-%m-%d %H:%M:%S")
                replay_rows = conn.execute(
                    """
                    SELECT action, status, pnl
                    FROM trade_replay
                    WHERE time >= ? AND action IN ('BUY', 'SELL')
                    """,
                    (local_cutoff,),
                ).fetchall()
            values = dict(row or {})
            wins = int(values.get("wins") or 0)
            losses = int(values.get("losses") or 0)
            scored = wins + losses
            gates: Dict[str, Dict[str, Any]] = {}
            for recent in recent_rows:
                stage = str(recent["rejection_stage"] or "").strip()
                reason = str(recent["rejection_reason"] or "").strip()
                match = re.search(r"REJECTED \[([^]]+)\]", reason)
                gate = (
                    match.group(1)
                    if match
                    else (
                        "Confidence"
                        if stage == "BELOW CONFIDENCE"
                        else stage.replace(" REJECTED", "").title()
                    )
                )
                outcome = float(recent["outcome_r"])
                item = gates.setdefault(
                    gate,
                    {"gate": gate, "resolved": 0, "wins": 0, "losses": 0, "total_r": 0.0},
                )
                item["resolved"] += 1
                item["wins"] += int(outcome > 0)
                item["losses"] += int(outcome < 0)
                item["total_r"] += outcome
            gate_breakdown = []
            for item in sorted(
                gates.values(),
                key=lambda value: (-value["resolved"], value["gate"]),
            )[:5]:
                resolved = int(item["resolved"])
                gate_breakdown.append(
                    {
                        "gate": item["gate"],
                        "resolved": resolved,
                        "wins": int(item["wins"]),
                        "losses": int(item["losses"]),
                        "expectancy_r": round(item["total_r"] / resolved, 3),
                    }
                )
            direction_breakdown = []
            for action in ("BUY", "SELL"):
                action_replays = [
                    item
                    for item in replay_rows
                    if str(item["action"] or "").upper() == action
                ]
                rejected = sum(
                    str(item["status"] or "").upper().startswith("REJECTED:")
                    for item in action_replays
                )
                executed_rows = [
                    item
                    for item in action_replays
                    if str(item["status"] or "").upper() == "CLOSED"
                    or str(item["status"] or "").upper().startswith("OPEN")
                ]
                closed_rows = [
                    item
                    for item in action_replays
                    if str(item["status"] or "").upper() == "CLOSED"
                ]
                shadow_rows = [
                    item
                    for item in recent_rows
                    if str(item["action"] or "").upper() == action
                ]
                shadow_outcomes = [
                    float(item["outcome_r"]) for item in shadow_rows
                ]
                rejection_gates: Dict[str, int] = {}
                for item in shadow_rows:
                    stage = str(item["rejection_stage"] or "").strip()
                    reason = str(item["rejection_reason"] or "").strip()
                    match = re.search(r"REJECTED \[([^]]+)\]", reason)
                    gate = (
                        match.group(1)
                        if match
                        else (
                            "Confidence"
                            if stage == "BELOW CONFIDENCE"
                            else stage.replace(" REJECTED", "").title()
                        )
                    )
                    rejection_gates[gate] = rejection_gates.get(gate, 0) + 1
                top_rejection_gates = [
                    {"gate": gate, "count": count}
                    for gate, count in sorted(
                        rejection_gates.items(),
                        key=lambda value: (-value[1], value[0]),
                    )[:3]
                ]
                candidates = len(action_replays)
                executed = len(executed_rows)
                direction_breakdown.append(
                    {
                        "action": action,
                        "candidates": candidates,
                        "rejected": rejected,
                        "executed": executed,
                        "approval_rate_pct": round(
                            100.0 * executed / candidates, 1
                        ) if candidates else 0.0,
                        "closed_wins": sum(
                            float(item["pnl"] or 0.0) > 0.0
                            for item in closed_rows
                        ),
                        "closed_losses": sum(
                            float(item["pnl"] or 0.0) < 0.0
                            for item in closed_rows
                        ),
                        "closed_net_usd": round(
                            sum(
                                float(item["pnl"] or 0.0)
                                for item in closed_rows
                            ),
                            2,
                        ),
                        "shadow_resolved": len(shadow_outcomes),
                        "shadow_positive": sum(
                            outcome > 0.0 for outcome in shadow_outcomes
                        ),
                        "shadow_expectancy_r": round(
                            sum(shadow_outcomes) / len(shadow_outcomes), 3
                        ) if shadow_outcomes else 0.0,
                        "top_rejection_gates": top_rejection_gates,
                    }
                )
            return {
                "pending": int(values.get("pending") or 0),
                "resolved": int(values.get("resolved") or 0),
                "wins": wins,
                "losses": losses,
                "ambiguous": int(values.get("ambiguous") or 0),
                "win_rate_pct": round(100.0 * wins / scored, 1) if scored else 0.0,
                "expectancy_r": round(float(values.get("expectancy_r") or 0.0), 3),
                "gate_breakdown": gate_breakdown,
                "evidence_window_hours": evidence_window_hours,
                "direction_breakdown": direction_breakdown,
            }
        except Exception as exc:
            logger.error("Error summarizing shadow outcomes: %s", exc)
            return empty

    def log_decision_trace(
        self,
        *,
        symbol: str,
        candle_time: str,
        telemetry: Dict[str, Any],
        decision: Optional[Dict[str, Any]],
        validation_status: str,
    ) -> Optional[int]:
        """Persist one model call, including HOLD and failed decisions, for evals."""
        trace_id = str(telemetry.get("trace_id", "")).strip()
        if not trace_id:
            logger.error("Decision trace was not logged because trace_id is missing")
            return None
        response = dict(decision or {})
        # Telemetry is stored in dedicated columns; avoid duplicating it in the
        # response JSON so later strategy evaluation has one canonical source.
        response.pop("_agent", None)
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA busy_timeout=10000")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO decision_trace (
                    trace_id, time, symbol, candle_time, provider, model,
                    latency_seconds, success, request_id, prompt_sha256,
                    prompt_tokens_estimate, action, confidence,
                    validation_status, error, config_fingerprint, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id,
                str(telemetry.get("started_at_utc") or datetime.now().isoformat()),
                symbol.upper(),
                candle_time,
                str(telemetry.get("provider", "unknown")),
                str(telemetry.get("model", "unknown")),
                float(telemetry.get("latency_seconds", 0.0) or 0.0),
                1 if telemetry.get("success") else 0,
                str(telemetry.get("request_id", "")),
                str(telemetry.get("prompt_sha256", "")),
                int(telemetry.get("prompt_tokens_estimate", 0) or 0),
                str(response.get("action", "")),
                float(response.get("confidence", 0.0) or 0.0),
                validation_status,
                str(telemetry.get("error", ""))[:1000],
                settings.config_fingerprint,
                json.dumps(response),
            ))
            conn.commit()
            row_id = cursor.lastrowid
            return row_id
        except Exception as exc:
            logger.error("Error logging decision trace: %s", exc)
            return None
        finally:
            if conn is not None:
                conn.close()

    def log_replay_attempt(
        self,
        symbol: str,
        action: str,
        prompt_text: str,
        llm_json: Dict[str, Any],
        indicators: Dict[str, Any],
        market_structure: Dict[str, Any],
        quality_score: float,
        confluence_score: float,
        status: str,
        ticket: Optional[int] = None
    ) -> Optional[int]:
        """
        Logs a full trade attempt snapshot. Returns the database record ID.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            entry = float(llm_json.get("entry", 0.0) or 0.0)
            sl = float(llm_json.get("stop_loss", 0.0) or 0.0)
            tp = float(llm_json.get("take_profit", 0.0) or 0.0)
            risk_pct = float(llm_json.get("risk_percentage", 1.0) or 1.0)
            confidence = float(llm_json.get("confidence", 0.0) or 0.0)
            reasoning = str(llm_json.get("reasoning", ""))
            
            cursor.execute("""
                INSERT INTO trade_replay (
                    time, symbol, action, prompt_text, llm_json, indicators,
                    market_structure, entry, stop_loss, take_profit, pnl,
                    risk_percent, quality_score, confluence_score, confidence,
                    reasoning, status, ticket, close_time, config_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol,
                action,
                prompt_text,
                json.dumps(llm_json),
                json.dumps(indicators),
                json.dumps(market_structure),
                entry,
                sl,
                tp,
                None,  # pnl is Null until closed
                risk_pct,
                quality_score,
                confluence_score,
                confidence,
                reasoning,
                status,
                ticket,
                None,
                settings.config_fingerprint,
            ))
            conn.commit()
            row_id = cursor.lastrowid
            return row_id
        except Exception as e:
            logger.error(f"Error logging trade replay attempt: {e}")
            return None
        finally:
            if conn is not None:
                conn.close()

    def update_replay_outcome(
        self,
        ticket: int,
        pnl: float,
        *,
        close_time: Optional[str] = None,
        close_reason: str = "",
        mfe_pips: float = 0.0,
        mae_pips: float = 0.0,
        mfe_usd: float = 0.0,
        mae_usd: float = 0.0,
        rr_achieved: float = 0.0,
    ) -> bool:
        """
        Update final broker-confirmed outcome and excursion telemetry.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            existing = cursor.execute(
                "SELECT close_reason FROM trade_replay WHERE ticket = ? "
                "ORDER BY id DESC LIMIT 1",
                (ticket,),
            ).fetchone()
            existing_reason = str(existing[0] or "").strip() if existing else ""
            candidate_reason = str(close_reason or "").strip()
            # MT5 commonly reconciles application closes as generic EXPERT
            # exits. Preserve the strategy cause recorded at submission time,
            # while still allowing broker TP/SL reasons to fill an empty row.
            generic_reasons = {"", "EXPERT", "CLIENT", "MOBILE", "WEB", "UNKNOWN"}
            if (
                existing_reason
                and existing_reason.upper() not in generic_reasons
                and candidate_reason.upper() in generic_reasons
            ):
                final_close_reason = existing_reason
            else:
                final_close_reason = candidate_reason or existing_reason
            cursor.execute("""
                UPDATE trade_replay
                SET pnl = ?, status = 'CLOSED', close_time = ?,
                    close_reason = ?, mfe_pips = ?, mae_pips = ?,
                    mfe_usd = ?, mae_usd = ?, rr_achieved = ?
                WHERE ticket = ?
            """, (
                pnl,
                close_time or datetime.now().isoformat(),
                final_close_reason,
                mfe_pips,
                mae_pips,
                mfe_usd,
                mae_usd,
                rr_achieved,
                ticket,
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating trade replay outcome: {e}")
            return False
        finally:
            if conn is not None:
                conn.close()

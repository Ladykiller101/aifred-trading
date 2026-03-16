"""Structured trade logging to SQLite with query interface."""

import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.types import TradeResult, TradeStatus

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persists every trade to SQLite with full metadata for analysis."""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables on first run."""
        try:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    asset TEXT NOT NULL,
                    asset_class TEXT,
                    direction TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    fill_price REAL,
                    fill_size REAL,
                    slippage REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    pnl REAL,
                    status TEXT NOT NULL,
                    exchange TEXT,
                    signal_source TEXT,
                    confidence REAL,
                    model_version TEXT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)
            """)
            conn.commit()
            conn.close()
            logger.info("Trade logger initialized (db=%s)", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize trade logger DB: %s", e)

    def log_trade(self, result: TradeResult) -> int:
        """Log a completed trade result. Returns the row ID."""
        proposal = result.proposal
        signal = proposal.signal

        side = "buy"
        if proposal.direction.value in ("SELL", "STRONG_SELL"):
            side = "sell"

        try:
            conn = self._get_conn()
            cursor = conn.execute(
                """INSERT INTO trades
                   (order_id, asset, asset_class, direction, side, order_type,
                    size, entry_price, stop_loss, take_profit,
                    fill_price, fill_size, slippage, fees,
                    status, exchange, signal_source, confidence,
                    entry_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.order_id,
                    proposal.asset,
                    proposal.asset_class.value if proposal.asset_class else None,
                    proposal.direction.value,
                    side,
                    proposal.order_type.value if proposal.order_type else None,
                    proposal.position_size,
                    proposal.entry_price,
                    proposal.stop_loss,
                    proposal.take_profit,
                    result.fill_price,
                    result.fill_size,
                    result.slippage,
                    result.fees,
                    result.status.value,
                    result.exchange,
                    signal.source if signal else None,
                    signal.confidence if signal else None,
                    result.timestamp.isoformat(),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid
            conn.close()
            logger.info("Trade logged: id=%d asset=%s status=%s",
                        row_id, proposal.asset, result.status.value)
            return row_id
        except Exception as e:
            logger.error("Failed to log trade: %s", e)
            return -1

    def log_exit(self, order_id: str, exit_price: float, pnl: float,
                 exit_time: Optional[datetime] = None) -> None:
        """Update a trade record with exit information."""
        exit_t = (exit_time or datetime.utcnow()).isoformat()
        try:
            conn = self._get_conn()
            conn.execute(
                """UPDATE trades SET exit_price = ?, pnl = ?, exit_time = ?,
                   status = 'filled' WHERE order_id = ?""",
                (exit_price, pnl, exit_t, order_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error("Failed to log exit for %s: %s", order_id, e)

    # --- Query interface ---

    def get_trades(self, start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   asset: Optional[str] = None,
                   strategy: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Query trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if strategy:
            query += " AND signal_source = ?"
            params.append(strategy)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        try:
            conn = self._get_conn()
            rows = conn.execute(query, params).fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error("Failed to query trades: %s", e)
            return []

    def get_trade_count(self) -> int:
        try:
            conn = self._get_conn()
            count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def get_winning_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM trades WHERE pnl > 0 ORDER BY pnl DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def get_losing_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM trades WHERE pnl < 0 ORDER BY pnl ASC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def get_pnl_summary(self) -> Dict[str, float]:
        """Get aggregate P&L statistics."""
        try:
            conn = self._get_conn()
            row = conn.execute("""
                SELECT
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    COALESCE(SUM(fees), 0) as total_fees,
                    COALESCE(AVG(slippage), 0) as avg_slippage
                FROM trades WHERE pnl IS NOT NULL
            """).fetchone()
            conn.close()
            total = row["total_trades"] or 1
            wins = row["winning_trades"] or 0
            return {
                "total_pnl": row["total_pnl"],
                "avg_pnl": row["avg_pnl"],
                "total_trades": row["total_trades"],
                "winning_trades": wins,
                "losing_trades": row["losing_trades"] or 0,
                "win_rate": wins / total * 100 if total > 0 else 0,
                "total_fees": row["total_fees"],
                "avg_slippage": row["avg_slippage"],
            }
        except Exception as e:
            logger.error("Failed to get P&L summary: %s", e)
            return {}

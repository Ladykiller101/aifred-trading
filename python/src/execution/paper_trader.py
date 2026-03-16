"""Paper trading: simulated execution engine for testing without real money."""

import logging
import random
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.types import AssetClass, OrderType, Position, TradeResult, TradeStatus

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulated exchange connector for paper trading.

    Implements the same interface as ExchangeConnector so it can be
    swapped in transparently. Tracks virtual balances, positions, and P&L.
    """

    def __init__(self, initial_balance: float = 100_000.0,
                 slippage_pct: float = 0.05,
                 fee_pct: float = 0.1,
                 db_path: str = "data/paper_trades.db"):
        self.name = "paper"
        self.slippage_pct = slippage_pct / 100.0
        self.fee_pct = fee_pct / 100.0
        self._balances: Dict[str, float] = {"USD": initial_balance, "USDT": initial_balance}
        self._positions: Dict[str, Position] = {}
        self._orders: List[Dict[str, Any]] = []
        self._prices: Dict[str, float] = {}  # symbol -> simulated current price
        self._trade_history: List[Dict[str, Any]] = []
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    requested_price REAL,
                    fill_price REAL NOT NULL,
                    slippage REAL NOT NULL,
                    fees REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_balances (
                    currency TEXT PRIMARY KEY,
                    amount REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Could not initialize paper trade DB: %s", e)

    def set_price(self, symbol: str, price: float) -> None:
        """Set the simulated current price for a symbol."""
        self._prices[symbol] = price

    def _get_price(self, symbol: str) -> float:
        price = self._prices.get(symbol)
        if price is None:
            raise ValueError(f"No simulated price set for {symbol}. Call set_price() first.")
        return price

    def _simulate_fill_price(self, symbol: str, side: str,
                             order_type: str) -> float:
        """Calculate a realistic fill price with slippage."""
        base_price = self._get_price(symbol)
        if order_type == "market":
            # Random slippage within configured range
            slip = random.uniform(0, self.slippage_pct)
            if side == "buy":
                return base_price * (1 + slip)
            else:
                return base_price * (1 - slip)
        # Limit orders fill at the requested price (no slippage)
        return base_price

    def _calculate_fees(self, amount: float, price: float) -> float:
        return amount * price * self.fee_pct

    def get_balance(self) -> Dict[str, Any]:
        return {
            "total": self._balances.copy(),
            "free": self._balances.copy(),
            "used": {k: 0.0 for k in self._balances},
        }

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        price = self._get_price(symbol)
        spread = price * 0.0005  # 0.05% simulated spread
        return {
            "symbol": symbol,
            "bid": price - spread / 2,
            "ask": price + spread / 2,
            "last": price,
            "timestamp": int(time.time() * 1000),
        }

    def place_order(self, symbol: str, side: str, order_type: str,
                    amount: float, price: Optional[float] = None,
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simulate placing an order with realistic fills."""
        order_id = str(uuid.uuid4())[:12]

        fill_price = self._simulate_fill_price(symbol, side, order_type)
        fees = self._calculate_fees(amount, fill_price)
        slippage = 0.0
        if price and price > 0:
            slippage = abs(fill_price - price) / price * 100

        # Update virtual balances
        quote = symbol.split("/")[1] if "/" in symbol else "USD"
        cost = amount * fill_price + fees

        if side == "buy":
            if self._balances.get(quote, 0) < cost:
                logger.warning("Paper trade: insufficient balance for %s %s (need %.2f, have %.2f)",
                               side, symbol, cost, self._balances.get(quote, 0))
                return {
                    "id": order_id, "status": "rejected",
                    "info": "Insufficient paper balance",
                }
            self._balances[quote] = self._balances.get(quote, 0) - cost
            base = symbol.split("/")[0] if "/" in symbol else symbol
            self._balances[base] = self._balances.get(base, 0) + amount
        else:
            base = symbol.split("/")[0] if "/" in symbol else symbol
            if self._balances.get(base, 0) < amount:
                logger.warning("Paper trade: insufficient %s balance", base)
                return {
                    "id": order_id, "status": "rejected",
                    "info": "Insufficient paper balance",
                }
            self._balances[base] = self._balances.get(base, 0) - amount
            self._balances[quote] = self._balances.get(quote, 0) + (amount * fill_price - fees)

        order_record = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price,
            "filled": amount,
            "average": fill_price,
            "status": "closed",
            "fee": {"cost": fees, "currency": quote},
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().isoformat(),
        }
        self._orders.append(order_record)

        trade_record = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "amount": amount,
            "requested_price": price or fill_price,
            "fill_price": fill_price,
            "slippage": slippage,
            "fees": fees,
            "status": "filled",
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._trade_history.append(trade_record)
        self._save_trade(trade_record)

        logger.info("Paper trade executed: %s %s %.6f @ %.4f (slip=%.4f%%, fee=%.4f)",
                     side, symbol, amount, fill_price, slippage, fees)
        return order_record

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return {"id": order_id, "status": "cancelled"}

    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        for order in self._orders:
            if order["id"] == order_id:
                return order
        return {"id": order_id, "status": "unknown"}

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        # Paper trades fill instantly, so there are no open orders
        return []

    def ping(self) -> float:
        return 0.1  # Simulated latency

    # --- Position tracking ---

    def open_position(self, asset: str, asset_class: AssetClass,
                      side: str, entry_price: float, size: float,
                      stop_loss: float, take_profit: float,
                      order_id: str = "", strategy: str = "") -> Position:
        pos = Position(
            asset=asset, asset_class=asset_class, side=side,
            entry_price=entry_price, current_price=entry_price,
            size=size, stop_loss=stop_loss, take_profit=take_profit,
            order_id=order_id, strategy=strategy,
        )
        self._positions[asset] = pos
        return pos

    def close_position(self, asset: str, exit_price: float) -> Optional[float]:
        """Close a position and return realized P&L."""
        pos = self._positions.pop(asset, None)
        if pos is None:
            return None
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size
        return pnl

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        cash = self._balances.get("USD", 0) + self._balances.get("USDT", 0)
        positions_value = sum(
            p.current_price * p.size for p in self._positions.values()
        )
        return cash + positions_value

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._trade_history[-limit:]

    # --- Persistence ---

    def _save_trade(self, trade: Dict[str, Any]) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT OR REPLACE INTO paper_trades
                   (id, symbol, side, order_type, amount, requested_price,
                    fill_price, slippage, fees, status, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade["id"], trade["symbol"], trade["side"], trade["order_type"],
                 trade["amount"], trade["requested_price"], trade["fill_price"],
                 trade["slippage"], trade["fees"], trade["status"], trade["timestamp"]),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Could not save paper trade: %s", e)

    def save_balances(self) -> None:
        try:
            conn = sqlite3.connect(self.db_path)
            now = datetime.utcnow().isoformat()
            for currency, amount in self._balances.items():
                conn.execute(
                    """INSERT OR REPLACE INTO paper_balances (currency, amount, updated_at)
                       VALUES (?, ?, ?)""",
                    (currency, amount, now),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Could not save paper balances: %s", e)

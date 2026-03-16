"""Order lifecycle management: creation, modification, tracking, and cancellation."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.types import OrderType, TradeStatus

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ManagedOrder:
    """Tracks the full lifecycle of an order."""

    def __init__(self, symbol: str, side: str, order_type: OrderType,
                 amount: float, price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 params: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())[:12]
        self.exchange_order_id: Optional[str] = None
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.amount = amount
        self.price = price
        self.stop_price = stop_price
        self.params = params or {}
        self.state = OrderState.PENDING
        self.filled_amount = 0.0
        self.average_fill_price = 0.0
        self.fees = 0.0
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.retries = 0
        self.max_retries = 3
        self.error: Optional[str] = None
        self.exchange: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "amount": self.amount,
            "price": self.price,
            "state": self.state.value,
            "filled_amount": self.filled_amount,
            "average_fill_price": self.average_fill_price,
            "fees": self.fees,
            "exchange": self.exchange,
            "created_at": self.created_at.isoformat(),
        }


class OrderManager:
    """Manages order creation, modification, cancellation, and state tracking."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._orders: Dict[str, ManagedOrder] = {}
        self._exchange_id_map: Dict[str, str] = {}  # exchange_order_id -> internal_id

    def create_order(self, symbol: str, side: str, order_type: OrderType,
                     amount: float, price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     params: Optional[Dict[str, Any]] = None) -> ManagedOrder:
        """Create a new managed order (does not submit it yet)."""
        order = ManagedOrder(symbol, side, order_type, amount, price, stop_price, params)
        order.max_retries = self.max_retries
        self._orders[order.id] = order
        logger.info("Created order %s: %s %s %s %.6f @ %s",
                     order.id, side, order_type.value, symbol, amount, price)
        return order

    def submit_order(self, order: ManagedOrder, connector) -> Dict[str, Any]:
        """Submit an order to an exchange via the connector.

        Handles retry logic for transient failures.
        """
        ccxt_type = self._map_order_type(order.order_type)
        params = dict(order.params)
        if order.stop_price and order.order_type == OrderType.STOP_LIMIT:
            params["stopPrice"] = order.stop_price

        last_error = None
        for attempt in range(order.max_retries):
            try:
                result = connector.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=ccxt_type,
                    amount=order.amount,
                    price=order.price,
                    params=params,
                )
                order.exchange_order_id = result.get("id")
                order.exchange = connector.name
                order.state = OrderState.SUBMITTED
                order.updated_at = datetime.utcnow()
                if order.exchange_order_id:
                    self._exchange_id_map[order.exchange_order_id] = order.id
                logger.info("Order %s submitted (exchange_id=%s, attempt=%d)",
                            order.id, order.exchange_order_id, attempt + 1)
                return result
            except Exception as e:
                last_error = e
                order.retries = attempt + 1
                logger.warning("Order %s submit attempt %d failed: %s",
                               order.id, attempt + 1, e)
                if attempt < order.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))  # linear backoff

        order.state = OrderState.FAILED
        order.error = str(last_error)
        order.updated_at = datetime.utcnow()
        logger.error("Order %s failed after %d retries: %s",
                      order.id, order.max_retries, last_error)
        raise last_error  # type: ignore[misc]

    def update_order_status(self, order: ManagedOrder, connector) -> ManagedOrder:
        """Poll exchange for the latest order status."""
        if not order.exchange_order_id:
            return order
        try:
            status = connector.get_order_status(order.exchange_order_id, order.symbol)
            exchange_status = status.get("status", "").lower()
            order.filled_amount = float(status.get("filled", 0))
            order.average_fill_price = float(status.get("average", 0) or 0)
            fee_info = status.get("fee") or {}
            order.fees = float(fee_info.get("cost", 0) or 0)

            if exchange_status == "closed":
                order.state = OrderState.FILLED
            elif exchange_status == "canceled" or exchange_status == "cancelled":
                order.state = OrderState.CANCELLED
            elif exchange_status == "open" and order.filled_amount > 0:
                order.state = OrderState.PARTIAL
            elif exchange_status == "rejected":
                order.state = OrderState.FAILED

            order.updated_at = datetime.utcnow()
        except Exception as e:
            logger.warning("Failed to update status for order %s: %s", order.id, e)
        return order

    def cancel_order(self, order: ManagedOrder, connector) -> bool:
        """Cancel an open order."""
        if order.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.FAILED):
            logger.info("Order %s already in terminal state %s", order.id, order.state.value)
            return False
        if not order.exchange_order_id:
            order.state = OrderState.CANCELLED
            order.updated_at = datetime.utcnow()
            return True
        try:
            connector.cancel_order(order.exchange_order_id, order.symbol)
            order.state = OrderState.CANCELLED
            order.updated_at = datetime.utcnow()
            logger.info("Order %s cancelled", order.id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order.id, e)
            return False

    def create_oco_orders(self, symbol: str, side: str, amount: float,
                          take_profit_price: float,
                          stop_loss_price: float) -> tuple[ManagedOrder, ManagedOrder]:
        """Create an OCO (One-Cancels-Other) pair: take-profit limit + stop-loss."""
        tp_order = self.create_order(
            symbol=symbol, side=side, order_type=OrderType.LIMIT,
            amount=amount, price=take_profit_price,
        )
        sl_order = self.create_order(
            symbol=symbol, side=side, order_type=OrderType.STOP_LIMIT,
            amount=amount, price=stop_loss_price, stop_price=stop_loss_price,
        )
        tp_order.params["oco_pair"] = sl_order.id
        sl_order.params["oco_pair"] = tp_order.id
        return tp_order, sl_order

    def create_twap_slices(self, symbol: str, side: str, total_amount: float,
                           price: Optional[float], num_slices: int = 5,
                           interval_seconds: int = 60) -> List[ManagedOrder]:
        """Split a large order into TWAP slices."""
        slice_amount = total_amount / num_slices
        slices = []
        for i in range(num_slices):
            order = self.create_order(
                symbol=symbol, side=side, order_type=OrderType.LIMIT,
                amount=slice_amount, price=price,
                params={"twap_slice": i, "twap_total": num_slices,
                        "twap_interval": interval_seconds},
            )
            slices.append(order)
        logger.info("Created %d TWAP slices for %s %s %.6f",
                     num_slices, side, symbol, total_amount)
        return slices

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(order_id)

    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[ManagedOrder]:
        internal_id = self._exchange_id_map.get(exchange_order_id)
        if internal_id:
            return self._orders.get(internal_id)
        return None

    def get_open_orders(self) -> List[ManagedOrder]:
        return [o for o in self._orders.values()
                if o.state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL)]

    def get_all_orders(self) -> List[ManagedOrder]:
        return list(self._orders.values())

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LIMIT: "stop",
            OrderType.TRAILING_STOP: "trailing_stop",
        }
        return mapping.get(order_type, "limit")

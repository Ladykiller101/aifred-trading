"""Unified exchange API wrapper using ccxt."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import ccxt
import ccxt.async_support as ccxt_async

from src.utils.types import AssetClass

logger = logging.getLogger(__name__)


class ExchangeConnector:
    """Unified interface for interacting with multiple exchanges."""

    # Mapping from exchange name to ccxt class name
    EXCHANGE_MAP = {
        "binance": "binance",
        "coinbase": "coinbasepro",
        "kraken": "kraken",
        "bybit": "bybit",
        "alpaca": "alpaca",
        "oanda": "oanda",
    }

    def __init__(self, name: str, api_key: str = "", secret: str = "",
                 sandbox: bool = True, extra_params: Optional[Dict[str, Any]] = None):
        self.name = name.lower()
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.extra_params = extra_params or {}
        self._sync_exchange: Optional[ccxt.Exchange] = None
        self._async_exchange: Optional[ccxt_async.Exchange] = None
        self._connected = False
        self._last_request_time = 0.0
        self._consecutive_failures = 0
        self._max_failures = 5

    def _build_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if self.api_key:
            config["apiKey"] = self.api_key
        if self.secret:
            config["secret"] = self.secret
        config.update(self.extra_params)
        return config

    def connect(self) -> None:
        """Establish synchronous connection to the exchange."""
        ccxt_name = self.EXCHANGE_MAP.get(self.name, self.name)
        exchange_class = getattr(ccxt, ccxt_name, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {self.name}")
        self._sync_exchange = exchange_class(self._build_config())
        if self.sandbox:
            self._sync_exchange.set_sandbox_mode(True)
        self._connected = True
        logger.info("Connected to %s (sandbox=%s)", self.name, self.sandbox)

    async def connect_async(self) -> None:
        """Establish async connection to the exchange."""
        ccxt_name = self.EXCHANGE_MAP.get(self.name, self.name)
        exchange_class = getattr(ccxt_async, ccxt_name, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {self.name}")
        self._async_exchange = exchange_class(self._build_config())
        if self.sandbox:
            self._async_exchange.set_sandbox_mode(True)
        self._connected = True
        logger.info("Async connected to %s (sandbox=%s)", self.name, self.sandbox)

    def _ensure_connected(self) -> ccxt.Exchange:
        if self._sync_exchange is None:
            self.connect()
        assert self._sync_exchange is not None
        return self._sync_exchange

    def _ensure_async(self) -> ccxt_async.Exchange:
        if self._async_exchange is None:
            raise RuntimeError("Call connect_async() first")
        return self._async_exchange

    def _handle_failure(self, error: Exception) -> None:
        self._consecutive_failures += 1
        logger.warning("Exchange %s failure #%d: %s",
                       self.name, self._consecutive_failures, error)
        if self._consecutive_failures >= self._max_failures:
            logger.error("Exchange %s hit max consecutive failures, reconnecting", self.name)
            self._connected = False
            self._consecutive_failures = 0
            try:
                self.connect()
            except Exception as reconn_err:
                logger.error("Reconnect failed: %s", reconn_err)

    def _handle_success(self) -> None:
        self._consecutive_failures = 0

    # --- Public API ---

    def get_balance(self) -> Dict[str, Any]:
        """Fetch account balance."""
        exchange = self._ensure_connected()
        try:
            balance = exchange.fetch_balance()
            self._handle_success()
            return balance
        except Exception as e:
            self._handle_failure(e)
            raise

    async def get_balance_async(self) -> Dict[str, Any]:
        exchange = self._ensure_async()
        try:
            balance = await exchange.fetch_balance()
            self._handle_success()
            return balance
        except Exception as e:
            self._handle_failure(e)
            raise

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker for a symbol."""
        exchange = self._ensure_connected()
        try:
            ticker = exchange.fetch_ticker(symbol)
            self._handle_success()
            return ticker
        except Exception as e:
            self._handle_failure(e)
            raise

    async def get_ticker_async(self, symbol: str) -> Dict[str, Any]:
        exchange = self._ensure_async()
        try:
            ticker = await exchange.fetch_ticker(symbol)
            self._handle_success()
            return ticker
        except Exception as e:
            self._handle_failure(e)
            raise

    def place_order(self, symbol: str, side: str, order_type: str,
                    amount: float, price: Optional[float] = None,
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Place an order on the exchange.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT")
            side: "buy" or "sell"
            order_type: "market", "limit", "stop_limit", etc.
            amount: Order size in base currency
            price: Limit price (required for limit orders)
            params: Extra exchange-specific params
        """
        exchange = self._ensure_connected()
        try:
            order = exchange.create_order(symbol, order_type, side, amount, price, params or {})
            self._handle_success()
            logger.info("Order placed on %s: %s %s %s %.6f @ %s",
                        self.name, side, order_type, symbol, amount, price)
            return order
        except Exception as e:
            self._handle_failure(e)
            raise

    async def place_order_async(self, symbol: str, side: str, order_type: str,
                                amount: float, price: Optional[float] = None,
                                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        exchange = self._ensure_async()
        try:
            order = await exchange.create_order(symbol, order_type, side, amount, price, params or {})
            self._handle_success()
            logger.info("Async order placed on %s: %s %s %s %.6f @ %s",
                        self.name, side, order_type, symbol, amount, price)
            return order
        except Exception as e:
            self._handle_failure(e)
            raise

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an open order."""
        exchange = self._ensure_connected()
        try:
            result = exchange.cancel_order(order_id, symbol)
            self._handle_success()
            logger.info("Order %s cancelled on %s", order_id, self.name)
            return result
        except Exception as e:
            self._handle_failure(e)
            raise

    async def cancel_order_async(self, order_id: str, symbol: str) -> Dict[str, Any]:
        exchange = self._ensure_async()
        try:
            result = await exchange.cancel_order(order_id, symbol)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure(e)
            raise

    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get the status of an order."""
        exchange = self._ensure_connected()
        try:
            order = exchange.fetch_order(order_id, symbol)
            self._handle_success()
            return order
        except Exception as e:
            self._handle_failure(e)
            raise

    async def get_order_status_async(self, order_id: str, symbol: str) -> Dict[str, Any]:
        exchange = self._ensure_async()
        try:
            order = await exchange.fetch_order(order_id, symbol)
            self._handle_success()
            return order
        except Exception as e:
            self._handle_failure(e)
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all open orders, optionally filtered by symbol."""
        exchange = self._ensure_connected()
        try:
            orders = exchange.fetch_open_orders(symbol)
            self._handle_success()
            return orders
        except Exception as e:
            self._handle_failure(e)
            raise

    def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                  limit: int = 100) -> List[List]:
        """Fetch OHLCV candle data."""
        exchange = self._ensure_connected()
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            self._handle_success()
            return data
        except Exception as e:
            self._handle_failure(e)
            raise

    def ping(self) -> float:
        """Check connectivity and return latency in ms."""
        exchange = self._ensure_connected()
        start = time.time()
        try:
            exchange.fetch_time()
            latency = (time.time() - start) * 1000
            self._handle_success()
            return latency
        except Exception as e:
            self._handle_failure(e)
            return -1.0

    async def close_async(self) -> None:
        """Close async exchange connection."""
        if self._async_exchange:
            await self._async_exchange.close()
            self._async_exchange = None
            self._connected = False

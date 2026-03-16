"""Unified broker adapter interface and implementations.

Provides a common abstraction over multiple brokers / exchanges so the
rest of the system can interact with any supported platform through the
same API.

Concrete adapters:
    - CCXTAdapter        — any CCXT-supported exchange (Binance, Coinbase, Kraken, Bybit)
    - AlpacaAdapter      — US stocks via Alpaca Trade API
    - OANDAAdapter       — forex via OANDA v20 API
    - InteractiveBrokersAdapter — stub (IB TWS / Gateway)
    - MetaTraderAdapter  — stub (MT4 / MT5)
    - BloombergAdapter   — stub (Bloomberg Terminal)

Usage::

    adapter = BrokerRegistry.create_adapter("binance")
    adapter.connect({"api_key": "...", "api_secret": "..."})
    print(adapter.get_balance())
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.utils.types import AssetClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Abstract base for all broker connections."""

    name: str = ""
    display_name: str = ""
    supported_assets: List[AssetClass] = []
    required_credentials: List[str] = []

    @abstractmethod
    def connect(self, credentials: dict) -> bool:
        """Establish a connection to the broker.

        Args:
            credentials: Dict of credential keys required by this adapter.

        Returns:
            True if connected successfully.
        """
        ...

    @abstractmethod
    def test_connection(self) -> dict:
        """Test the connection and return diagnostics.

        Returns:
            Dict with keys: connected (bool), latency_ms (float),
            account_id (str), balance (dict).
        """
        ...

    @abstractmethod
    def get_balance(self) -> dict:
        """Fetch account balance.

        Returns:
            Dict mapping currency -> available amount.
        """
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict:
        """Place an order.

        Args:
            symbol: Trading pair / ticker.
            side: "buy" or "sell".
            order_type: "market", "limit", etc.
            amount: Order size.
            price: Limit price (None for market orders).

        Returns:
            Order receipt dict (id, status, filled, etc.).
        """
        ...

    @abstractmethod
    def get_positions(self) -> list:
        """Return list of currently open positions."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release resources."""
        ...


# ---------------------------------------------------------------------------
# CCXTAdapter — Binance, Coinbase, Kraken, Bybit
# ---------------------------------------------------------------------------

class CCXTAdapter(BrokerAdapter):
    """Wraps the ExchangeConnector for any CCXT-supported exchange."""

    def __init__(self, exchange_id: str = "binance"):
        self._exchange_id = exchange_id.lower()
        self.name = self._exchange_id
        self.display_name = self._exchange_id.capitalize()
        self.supported_assets = [AssetClass.CRYPTO]
        self.required_credentials = self._creds_for_exchange(self._exchange_id)
        self._exchange = None  # ccxt.Exchange instance
        self._connected = False

    @staticmethod
    def _creds_for_exchange(exchange_id: str) -> List[str]:
        creds_map = {
            "binance": ["api_key", "api_secret"],
            "coinbase": ["api_key", "api_secret", "passphrase"],
            "coinbasepro": ["api_key", "api_secret", "passphrase"],
            "kraken": ["api_key", "api_secret"],
            "bybit": ["api_key", "api_secret"],
        }
        return creds_map.get(exchange_id, ["api_key", "api_secret"])

    def connect(self, credentials: dict) -> bool:
        import ccxt

        exchange_map = {
            "binance": "binance",
            "coinbase": "coinbasepro",
            "kraken": "kraken",
            "bybit": "bybit",
        }
        ccxt_id = exchange_map.get(self._exchange_id, self._exchange_id)
        exchange_class = getattr(ccxt, ccxt_id, None)
        if exchange_class is None:
            logger.error("Unsupported CCXT exchange: %s", ccxt_id)
            return False

        config: Dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if credentials.get("api_key"):
            config["apiKey"] = credentials["api_key"]
        if credentials.get("api_secret"):
            config["secret"] = credentials["api_secret"]
        if credentials.get("passphrase"):
            config["password"] = credentials["passphrase"]
        if credentials.get("sandbox", True):
            config["sandbox"] = True

        try:
            self._exchange = exchange_class(config)
            if credentials.get("sandbox", True):
                self._exchange.set_sandbox_mode(True)
            self._connected = True
            logger.info("CCXTAdapter connected to %s", self._exchange_id)
            return True
        except Exception as e:
            logger.error("Failed to connect to %s: %s", self._exchange_id, e)
            return False

    def test_connection(self) -> dict:
        result = {
            "connected": False,
            "latency_ms": -1.0,
            "account_id": "",
            "balance": {},
        }
        if self._exchange is None:
            return result

        try:
            start = time.time()
            self._exchange.fetch_time()
            result["latency_ms"] = (time.time() - start) * 1000
            result["connected"] = True
        except Exception as e:
            logger.warning("Connection test failed for %s: %s", self._exchange_id, e)
            return result

        try:
            balance = self._exchange.fetch_balance()
            result["balance"] = {
                k: v for k, v in balance.get("total", {}).items()
                if v and v > 0
            }
        except Exception:
            pass

        return result

    def get_balance(self) -> dict:
        if self._exchange is None:
            raise RuntimeError("Not connected. Call connect() first.")
        balance = self._exchange.fetch_balance()
        return {
            k: v for k, v in balance.get("total", {}).items()
            if v and v > 0
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict:
        if self._exchange is None:
            raise RuntimeError("Not connected. Call connect() first.")
        order = self._exchange.create_order(
            symbol, order_type, side, amount, price
        )
        logger.info(
            "CCXT order on %s: %s %s %s %.6f @ %s",
            self._exchange_id, side, order_type, symbol, amount, price,
        )
        return order

    def get_positions(self) -> list:
        if self._exchange is None:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            return self._exchange.fetch_positions()
        except Exception:
            # Not all exchanges support fetch_positions
            return []

    def disconnect(self) -> None:
        self._exchange = None
        self._connected = False
        logger.info("CCXTAdapter disconnected from %s", self._exchange_id)


# ---------------------------------------------------------------------------
# AlpacaAdapter — US Stocks
# ---------------------------------------------------------------------------

class AlpacaAdapter(BrokerAdapter):
    """Adapter for US stocks via Alpaca Trade API."""

    name = "alpaca"
    display_name = "Alpaca"
    supported_assets = [AssetClass.STOCKS]
    required_credentials = ["api_key", "api_secret"]

    _PAPER_URL = "https://paper-api.alpaca.markets"

    def __init__(self):
        self._api = None
        self._connected = False

    def connect(self, credentials: dict) -> bool:
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            logger.error(
                "alpaca-trade-api is not installed. "
                "Install with: pip install alpaca-trade-api"
            )
            return False

        base_url = credentials.get("base_url", self._PAPER_URL)
        try:
            self._api = tradeapi.REST(
                key_id=credentials["api_key"],
                secret_key=credentials["api_secret"],
                base_url=base_url,
            )
            # Quick connectivity check
            self._api.get_account()
            self._connected = True
            logger.info("AlpacaAdapter connected (base_url=%s)", base_url)
            return True
        except Exception as e:
            logger.error("Failed to connect to Alpaca: %s", e)
            return False

    def test_connection(self) -> dict:
        result = {
            "connected": False,
            "latency_ms": -1.0,
            "account_id": "",
            "balance": {},
        }
        if self._api is None:
            return result

        try:
            start = time.time()
            account = self._api.get_account()
            result["latency_ms"] = (time.time() - start) * 1000
            result["connected"] = True
            result["account_id"] = account.id
            result["balance"] = {
                "USD": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
            }
        except Exception as e:
            logger.warning("Alpaca connection test failed: %s", e)

        return result

    def get_balance(self) -> dict:
        if self._api is None:
            raise RuntimeError("Not connected. Call connect() first.")
        account = self._api.get_account()
        return {
            "USD": float(account.cash),
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict:
        if self._api is None:
            raise RuntimeError("Not connected. Call connect() first.")

        kwargs: Dict[str, Any] = {
            "symbol": symbol,
            "qty": amount,
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": "day",
        }
        if price is not None and order_type.lower() == "limit":
            kwargs["limit_price"] = price

        order = self._api.submit_order(**kwargs)
        logger.info(
            "Alpaca order: %s %s %s qty=%.4f @ %s",
            side, order_type, symbol, amount, price,
        )
        return {
            "id": order.id,
            "status": order.status,
            "symbol": order.symbol,
            "qty": order.qty,
            "side": order.side,
            "type": order.type,
        }

    def get_positions(self) -> list:
        if self._api is None:
            raise RuntimeError("Not connected. Call connect() first.")
        positions = self._api.list_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    def disconnect(self) -> None:
        self._api = None
        self._connected = False
        logger.info("AlpacaAdapter disconnected")


# ---------------------------------------------------------------------------
# OANDAAdapter — Forex
# ---------------------------------------------------------------------------

class OANDAAdapter(BrokerAdapter):
    """Adapter for forex via OANDA v20 REST API (through ccxt)."""

    name = "oanda"
    display_name = "OANDA"
    supported_assets = [AssetClass.FOREX]
    required_credentials = ["api_key", "account_id"]

    def __init__(self):
        self._exchange = None
        self._account_id: str = ""
        self._connected = False

    def connect(self, credentials: dict) -> bool:
        import ccxt

        self._account_id = credentials.get("account_id", "")
        config: Dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if credentials.get("api_key"):
            config["apiKey"] = credentials["api_key"]
        if self._account_id:
            config["uid"] = self._account_id

        try:
            self._exchange = ccxt.oanda(config)
            self._connected = True
            logger.info("OANDAAdapter connected (account=%s)", self._account_id)
            return True
        except Exception as e:
            logger.error("Failed to connect to OANDA: %s", e)
            return False

    def test_connection(self) -> dict:
        result = {
            "connected": False,
            "latency_ms": -1.0,
            "account_id": self._account_id,
            "balance": {},
        }
        if self._exchange is None:
            return result

        try:
            start = time.time()
            self._exchange.fetch_time()
            result["latency_ms"] = (time.time() - start) * 1000
            result["connected"] = True
        except Exception as e:
            logger.warning("OANDA connection test failed: %s", e)
            return result

        try:
            balance = self._exchange.fetch_balance()
            result["balance"] = {
                k: v for k, v in balance.get("total", {}).items()
                if v and v > 0
            }
        except Exception:
            pass

        return result

    def get_balance(self) -> dict:
        if self._exchange is None:
            raise RuntimeError("Not connected. Call connect() first.")
        balance = self._exchange.fetch_balance()
        return {
            k: v for k, v in balance.get("total", {}).items()
            if v and v > 0
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> dict:
        if self._exchange is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Convert EUR/USD -> EUR_USD for OANDA
        oanda_symbol = symbol.replace("/", "_")
        order = self._exchange.create_order(
            oanda_symbol, order_type, side, amount, price
        )
        logger.info(
            "OANDA order: %s %s %s %.4f @ %s",
            side, order_type, oanda_symbol, amount, price,
        )
        return order

    def get_positions(self) -> list:
        if self._exchange is None:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            return self._exchange.fetch_positions()
        except Exception:
            return []

    def disconnect(self) -> None:
        self._exchange = None
        self._connected = False
        logger.info("OANDAAdapter disconnected")


# ---------------------------------------------------------------------------
# Stubs — InteractiveBrokers, MetaTrader, Bloomberg
# ---------------------------------------------------------------------------

class InteractiveBrokersAdapter(BrokerAdapter):
    """Stub adapter for Interactive Brokers TWS / Gateway.

    Requires the ``ib_insync`` library and a running TWS or IB Gateway instance.
    Install: ``pip install ib_insync``
    """

    name = "interactive_brokers"
    display_name = "Interactive Brokers"
    supported_assets = [AssetClass.STOCKS, AssetClass.FOREX, AssetClass.CRYPTO]
    required_credentials = ["host", "port", "client_id"]

    def connect(self, credentials: dict) -> bool:
        raise NotImplementedError(
            "InteractiveBrokersAdapter is not yet implemented. "
            "To integrate IB:\n"
            "  1. Install ib_insync: pip install ib_insync\n"
            "  2. Run TWS or IB Gateway on the target host/port\n"
            "  3. Enable API connections in TWS settings\n"
            "  4. Implement connect() using ib_insync.IB().connect(host, port, clientId)\n"
        )

    def test_connection(self) -> dict:
        raise NotImplementedError("InteractiveBrokersAdapter is not yet implemented.")

    def get_balance(self) -> dict:
        raise NotImplementedError("InteractiveBrokersAdapter is not yet implemented.")

    def place_order(self, symbol, side, order_type, amount, price=None) -> dict:
        raise NotImplementedError("InteractiveBrokersAdapter is not yet implemented.")

    def get_positions(self) -> list:
        raise NotImplementedError("InteractiveBrokersAdapter is not yet implemented.")

    def disconnect(self) -> None:
        raise NotImplementedError("InteractiveBrokersAdapter is not yet implemented.")


class MetaTraderAdapter(BrokerAdapter):
    """Stub adapter for MetaTrader 4 / 5.

    Requires the ``MetaTrader5`` library and a running MT5 terminal (Windows only).
    Install: ``pip install MetaTrader5``
    """

    name = "metatrader"
    display_name = "MetaTrader 5"
    supported_assets = [AssetClass.FOREX, AssetClass.STOCKS]
    required_credentials = ["server", "login", "password", "path"]

    def connect(self, credentials: dict) -> bool:
        raise NotImplementedError(
            "MetaTraderAdapter is not yet implemented. "
            "To integrate MT5:\n"
            "  1. Install MetaTrader5: pip install MetaTrader5\n"
            "  2. Ensure MT5 terminal is installed and accessible at the given path\n"
            "  3. Note: MT5 library only works on Windows\n"
            "  4. Implement connect() using MetaTrader5.initialize(path) "
            "and MetaTrader5.login(login, password, server)\n"
        )

    def test_connection(self) -> dict:
        raise NotImplementedError("MetaTraderAdapter is not yet implemented.")

    def get_balance(self) -> dict:
        raise NotImplementedError("MetaTraderAdapter is not yet implemented.")

    def place_order(self, symbol, side, order_type, amount, price=None) -> dict:
        raise NotImplementedError("MetaTraderAdapter is not yet implemented.")

    def get_positions(self) -> list:
        raise NotImplementedError("MetaTraderAdapter is not yet implemented.")

    def disconnect(self) -> None:
        raise NotImplementedError("MetaTraderAdapter is not yet implemented.")


class BloombergAdapter(BrokerAdapter):
    """Stub adapter for Bloomberg Terminal.

    Requires the ``blpapi`` library and a running Bloomberg Terminal session.
    Install: ``pip install blpapi`` (requires Bloomberg C++ SDK)
    """

    name = "bloomberg"
    display_name = "Bloomberg Terminal"
    supported_assets = [AssetClass.STOCKS, AssetClass.FOREX, AssetClass.CRYPTO]
    required_credentials = []  # Connects via local Terminal session

    def connect(self, credentials: dict) -> bool:
        raise NotImplementedError(
            "BloombergAdapter is not yet implemented. "
            "To integrate Bloomberg:\n"
            "  1. Install blpapi: pip install blpapi "
            "(requires Bloomberg C++ SDK from Bloomberg BSDK)\n"
            "  2. Ensure Bloomberg Terminal is running on this machine\n"
            "  3. Implement connect() using blpapi.SessionOptions and blpapi.Session\n"
        )

    def test_connection(self) -> dict:
        raise NotImplementedError("BloombergAdapter is not yet implemented.")

    def get_balance(self) -> dict:
        raise NotImplementedError("BloombergAdapter is not yet implemented.")

    def place_order(self, symbol, side, order_type, amount, price=None) -> dict:
        raise NotImplementedError("BloombergAdapter is not yet implemented.")

    def get_positions(self) -> list:
        raise NotImplementedError("BloombergAdapter is not yet implemented.")

    def disconnect(self) -> None:
        raise NotImplementedError("BloombergAdapter is not yet implemented.")


# ---------------------------------------------------------------------------
# BrokerRegistry
# ---------------------------------------------------------------------------

# All known broker definitions
_BROKER_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "binance",
        "display_name": "Binance",
        "supported_assets": [AssetClass.CRYPTO],
        "status": "active",
        "required_credentials": ["api_key", "api_secret"],
        "factory": lambda: CCXTAdapter("binance"),
    },
    {
        "name": "coinbase",
        "display_name": "Coinbase",
        "supported_assets": [AssetClass.CRYPTO],
        "status": "active",
        "required_credentials": ["api_key", "api_secret", "passphrase"],
        "factory": lambda: CCXTAdapter("coinbase"),
    },
    {
        "name": "kraken",
        "display_name": "Kraken",
        "supported_assets": [AssetClass.CRYPTO],
        "status": "active",
        "required_credentials": ["api_key", "api_secret"],
        "factory": lambda: CCXTAdapter("kraken"),
    },
    {
        "name": "bybit",
        "display_name": "Bybit",
        "supported_assets": [AssetClass.CRYPTO],
        "status": "active",
        "required_credentials": ["api_key", "api_secret"],
        "factory": lambda: CCXTAdapter("bybit"),
    },
    {
        "name": "alpaca",
        "display_name": "Alpaca",
        "supported_assets": [AssetClass.STOCKS],
        "status": "active",
        "required_credentials": ["api_key", "api_secret"],
        "factory": lambda: AlpacaAdapter(),
    },
    {
        "name": "oanda",
        "display_name": "OANDA",
        "supported_assets": [AssetClass.FOREX],
        "status": "active",
        "required_credentials": ["api_key", "account_id"],
        "factory": lambda: OANDAAdapter(),
    },
    {
        "name": "interactive_brokers",
        "display_name": "Interactive Brokers",
        "supported_assets": [AssetClass.STOCKS, AssetClass.FOREX, AssetClass.CRYPTO],
        "status": "stub",
        "required_credentials": ["host", "port", "client_id"],
        "factory": lambda: InteractiveBrokersAdapter(),
    },
    {
        "name": "metatrader",
        "display_name": "MetaTrader 5",
        "supported_assets": [AssetClass.FOREX, AssetClass.STOCKS],
        "status": "stub",
        "required_credentials": ["server", "login", "password", "path"],
        "factory": lambda: MetaTraderAdapter(),
    },
    {
        "name": "bloomberg",
        "display_name": "Bloomberg Terminal",
        "supported_assets": [AssetClass.STOCKS, AssetClass.FOREX, AssetClass.CRYPTO],
        "status": "stub",
        "required_credentials": [],
        "factory": lambda: BloombergAdapter(),
    },
]

# Index by name for fast lookup
_BROKER_INDEX: Dict[str, Dict[str, Any]] = {b["name"]: b for b in _BROKER_DEFINITIONS}


class BrokerRegistry:
    """Registry of all available broker adapters."""

    @staticmethod
    def get_available_brokers() -> List[Dict[str, Any]]:
        """Return metadata for every registered broker.

        Returns:
            List of dicts with keys: name, display_name, supported_assets,
            status, required_credentials.
        """
        return [
            {
                "name": b["name"],
                "display_name": b["display_name"],
                "supported_assets": [a.value for a in b["supported_assets"]],
                "status": b["status"],
                "required_credentials": b["required_credentials"],
            }
            for b in _BROKER_DEFINITIONS
        ]

    @staticmethod
    def create_adapter(broker_name: str) -> BrokerAdapter:
        """Create and return a broker adapter by name.

        Args:
            broker_name: One of the registered broker names
                         (e.g. "binance", "alpaca", "oanda").

        Returns:
            An unconnected BrokerAdapter instance.

        Raises:
            ValueError: If broker_name is not registered.
        """
        defn = _BROKER_INDEX.get(broker_name.lower())
        if defn is None:
            available = ", ".join(sorted(_BROKER_INDEX.keys()))
            raise ValueError(
                f"Unknown broker: '{broker_name}'. "
                f"Available: {available}"
            )
        return defn["factory"]()

    @staticmethod
    def test_connection(broker_name: str, credentials: dict) -> dict:
        """Create an adapter, connect, test, and return diagnostics.

        Convenience method that handles the full lifecycle.

        Args:
            broker_name: Registered broker name.
            credentials: Dict of credentials for the broker.

        Returns:
            Connection test result dict.
        """
        adapter = BrokerRegistry.create_adapter(broker_name)
        try:
            connected = adapter.connect(credentials)
            if not connected:
                return {
                    "connected": False,
                    "latency_ms": -1.0,
                    "account_id": "",
                    "balance": {},
                    "error": "connect() returned False",
                }
            result = adapter.test_connection()
            return result
        except NotImplementedError as e:
            return {
                "connected": False,
                "latency_ms": -1.0,
                "account_id": "",
                "balance": {},
                "error": str(e),
            }
        except Exception as e:
            return {
                "connected": False,
                "latency_ms": -1.0,
                "account_id": "",
                "balance": {},
                "error": str(e),
            }
        finally:
            try:
                adapter.disconnect()
            except (NotImplementedError, Exception):
                pass

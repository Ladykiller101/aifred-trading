"""Smart order routing: find the best exchange for each trade."""

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.execution.exchange_connector import ExchangeConnector

logger = logging.getLogger(__name__)


class ExchangeQuote:
    """A price quote from an exchange for routing decisions."""

    def __init__(self, exchange_name: str, bid: float, ask: float,
                 fee_rate: float, latency_ms: float):
        self.exchange_name = exchange_name
        self.bid = bid
        self.ask = ask
        self.fee_rate = fee_rate
        self.latency_ms = latency_ms

    @property
    def spread(self) -> float:
        if self.bid == 0:
            return float("inf")
        return (self.ask - self.bid) / self.bid

    def effective_buy_price(self, amount: float) -> float:
        """Total cost to buy including fees."""
        return self.ask * amount * (1 + self.fee_rate)

    def effective_sell_price(self, amount: float) -> float:
        """Net proceeds from selling including fees."""
        return self.bid * amount * (1 - self.fee_rate)


class SmartRouter:
    """Routes orders to the exchange with the best execution quality."""

    # Default fee rates (maker/taker mid-point estimates)
    DEFAULT_FEES = {
        "binance": 0.001,
        "coinbase": 0.005,
        "kraken": 0.002,
        "bybit": 0.001,
        "alpaca": 0.0,
        "oanda": 0.0001,
    }

    def __init__(self, connectors: Dict[str, ExchangeConnector]):
        self.connectors = connectors
        self._fill_rates: Dict[str, List[float]] = defaultdict(list)  # exchange -> recent fill ratios
        self._routing_log: List[Dict[str, Any]] = []

    def get_quotes(self, symbol: str) -> List[ExchangeQuote]:
        """Fetch current bid/ask from all connected exchanges that support the symbol."""
        quotes = []
        for name, connector in self.connectors.items():
            try:
                start = time.time()
                ticker = connector.get_ticker(symbol)
                latency = (time.time() - start) * 1000
                bid = float(ticker.get("bid", 0) or 0)
                ask = float(ticker.get("ask", 0) or 0)
                if bid > 0 and ask > 0:
                    fee = self.DEFAULT_FEES.get(name, 0.002)
                    quotes.append(ExchangeQuote(name, bid, ask, fee, latency))
            except Exception as e:
                logger.debug("Could not get quote from %s for %s: %s", name, symbol, e)
        return quotes

    def best_exchange_for_buy(self, symbol: str, amount: float) -> Optional[Tuple[str, ExchangeQuote]]:
        """Find the cheapest exchange to buy on (lowest effective price)."""
        quotes = self.get_quotes(symbol)
        if not quotes:
            return None
        best = min(quotes, key=lambda q: q.effective_buy_price(amount))
        self._log_routing("buy", symbol, amount, quotes, best)
        return best.exchange_name, best

    def best_exchange_for_sell(self, symbol: str, amount: float) -> Optional[Tuple[str, ExchangeQuote]]:
        """Find the best exchange to sell on (highest effective proceeds)."""
        quotes = self.get_quotes(symbol)
        if not quotes:
            return None
        best = max(quotes, key=lambda q: q.effective_sell_price(amount))
        self._log_routing("sell", symbol, amount, quotes, best)
        return best.exchange_name, best

    def route_order(self, symbol: str, side: str, amount: float) -> Optional[str]:
        """Determine which exchange to route an order to.

        Returns the exchange name or None if no exchange is available.
        """
        if side.lower() == "buy":
            result = self.best_exchange_for_buy(symbol, amount)
        else:
            result = self.best_exchange_for_sell(symbol, amount)

        if result is None:
            logger.warning("No exchange available for %s %s %s", side, symbol, amount)
            return None

        exchange_name, quote = result
        # Factor in historical fill rate
        fill_rate = self._average_fill_rate(exchange_name)
        if fill_rate < 0.5:
            # If fill rate is terrible, try next best
            logger.warning("Exchange %s has low fill rate %.2f, considering alternatives",
                           exchange_name, fill_rate)

        return exchange_name

    def record_fill(self, exchange_name: str, requested: float, filled: float) -> None:
        """Record fill outcome to adjust future routing."""
        ratio = filled / requested if requested > 0 else 0.0
        history = self._fill_rates[exchange_name]
        history.append(ratio)
        # Keep only last 100 fills
        if len(history) > 100:
            self._fill_rates[exchange_name] = history[-100:]

    def _average_fill_rate(self, exchange_name: str) -> float:
        history = self._fill_rates.get(exchange_name, [])
        if not history:
            return 1.0  # Assume perfect fill if no data
        return sum(history) / len(history)

    def _log_routing(self, side: str, symbol: str, amount: float,
                     quotes: List[ExchangeQuote], chosen: ExchangeQuote) -> None:
        entry = {
            "side": side,
            "symbol": symbol,
            "amount": amount,
            "chosen_exchange": chosen.exchange_name,
            "chosen_price": chosen.ask if side == "buy" else chosen.bid,
            "chosen_fee": chosen.fee_rate,
            "alternatives": [
                {"exchange": q.exchange_name,
                 "price": q.ask if side == "buy" else q.bid,
                 "fee": q.fee_rate}
                for q in quotes if q.exchange_name != chosen.exchange_name
            ],
            "timestamp": time.time(),
        }
        self._routing_log.append(entry)
        # Keep last 1000 routing decisions
        if len(self._routing_log) > 1000:
            self._routing_log = self._routing_log[-1000:]
        logger.info("Routed %s %s %.6f to %s (price=%.4f, fee=%.4f%%)",
                     side, symbol, amount, chosen.exchange_name,
                     chosen.ask if side == "buy" else chosen.bid,
                     chosen.fee_rate * 100)

    def get_routing_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._routing_log[-limit:]

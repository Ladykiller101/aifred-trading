"""Unified market data provider for the AIFred trading system.

Fetches OHLCV data from multiple sources based on asset class:
- Crypto (e.g. BTC/USDT): Uses ccxt (Binance by default)
- Stocks (e.g. AAPL): Uses yfinance
- Forex (e.g. EUR/USD): Uses yfinance (EURUSD=X format) or OANDA via ccxt

Computes standard technical indicators on all returned data.
Includes a TTL-based cache to avoid excessive API calls.
"""

import logging
import re
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe mapping helpers
# ---------------------------------------------------------------------------

_YFINANCE_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1wk",
    "1M": "1mo",
}

_YFINANCE_PERIOD_MAP = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "5y",
    "1w": "10y",
    "1M": "max",
}


# ---------------------------------------------------------------------------
# Asset classification helpers
# ---------------------------------------------------------------------------

def _is_crypto(asset: str) -> bool:
    """Crypto assets contain '/' and have a quote currency (e.g. BTC/USDT)."""
    return "/" in asset and not _is_forex(asset)


def _is_forex(asset: str) -> bool:
    """Forex pairs are 3-letter/3-letter (e.g. EUR/USD)."""
    parts = asset.split("/")
    if len(parts) != 2:
        return False
    base, quote = parts
    return (
        len(base) == 3
        and len(quote) == 3
        and base.isalpha()
        and quote.isalpha()
        and base.upper() == base
        and quote.upper() == quote
    )


def _is_stock(asset: str) -> bool:
    """Stock tickers are single symbols up to 5 characters (e.g. AAPL, MSFT)."""
    return "/" not in asset and len(asset) <= 5 and asset.isalpha()


# ---------------------------------------------------------------------------
# Technical indicator computation
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and append them as columns.

    Expected input columns: open, high, low, close, volume.
    Added columns: atr, sma_20, sma_50, ema_12, ema_26, rsi,
                   macd, macd_signal, bb_upper, bb_lower.
    """
    if df is None or df.empty:
        return df

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # --- ATR (14-period) ---
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14, min_periods=1).mean()

    # --- Simple Moving Averages ---
    df["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    df["sma_50"] = close.rolling(window=50, min_periods=1).mean()

    # --- Exponential Moving Averages ---
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # --- MACD ---
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands (20-period, 2 std dev) ---
    bb_std = close.rolling(window=20, min_periods=1).std()
    df["bb_upper"] = df["sma_20"] + 2.0 * bb_std
    df["bb_lower"] = df["sma_20"] - 2.0 * bb_std

    return df


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _DataCache:
    """Simple TTL-based cache for market data."""

    def __init__(self, ttl_seconds: int = 60):
        self._ttl = ttl_seconds
        self._store: Dict[str, tuple] = {}  # key -> (timestamp, dataframe)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, df = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return df.copy()

    def put(self, key: str, df: pd.DataFrame) -> None:
        self._store[key] = (time.time(), df.copy())

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# MarketDataProvider
# ---------------------------------------------------------------------------

class MarketDataProvider:
    """Unified data provider that fetches OHLCV + indicators for any asset.

    Usage::

        provider = MarketDataProvider()
        orchestrator.set_data_provider(provider.get_data)
    """

    def __init__(
        self,
        default_exchange: str = "binance",
        cache_ttl: int = 60,
        min_candles: int = 200,
        ccxt_config: Optional[Dict[str, Any]] = None,
        use_oanda_for_forex: bool = False,
        oanda_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            default_exchange: CCXT exchange id for crypto data (default: binance).
            cache_ttl: Cache time-to-live in seconds.
            min_candles: Minimum number of candles to fetch.
            ccxt_config: Extra config dict passed to the ccxt exchange constructor.
            use_oanda_for_forex: If True, use OANDA via ccxt for forex instead
                                 of yfinance.
            oanda_config: CCXT config for OANDA (api key, account id, etc.).
        """
        self._default_exchange = default_exchange
        self._min_candles = max(min_candles, 200)
        self._cache = _DataCache(ttl_seconds=cache_ttl)
        self._ccxt_config = ccxt_config or {}
        self._use_oanda_for_forex = use_oanda_for_forex
        self._oanda_config = oanda_config or {}

        # Lazy-initialized exchange instances
        self._ccxt_exchange = None
        self._oanda_exchange = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_data(self, asset: str, timeframe: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with technical indicators for *asset*.

        This is the callback the orchestrator expects:
            ``provider.get_data(asset, timeframe) -> pd.DataFrame | None``

        Returns:
            DataFrame with columns: open, high, low, close, volume,
            atr, sma_20, sma_50, ema_12, ema_26, rsi, macd, macd_signal,
            bb_upper, bb_lower.  DatetimeIndex.
            Returns None on failure.
        """
        cache_key = f"{asset}|{timeframe}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for %s (%s)", asset, timeframe)
            return cached

        try:
            if _is_crypto(asset):
                df = self._fetch_crypto(asset, timeframe)
            elif _is_forex(asset):
                df = self._fetch_forex(asset, timeframe)
            elif _is_stock(asset):
                df = self._fetch_stock(asset, timeframe)
            else:
                logger.warning(
                    "Cannot classify asset '%s' — trying crypto fetch as fallback",
                    asset,
                )
                df = self._fetch_crypto(asset, timeframe)

            if df is None or df.empty:
                logger.warning("No data returned for %s (%s)", asset, timeframe)
                return None

            # Ensure minimum candle count
            if len(df) < self._min_candles:
                logger.warning(
                    "Only %d candles for %s (need %d) — returning available data",
                    len(df), asset, self._min_candles,
                )

            # Compute technical indicators
            df = compute_indicators(df)

            # Cache result
            self._cache.put(cache_key, df)

            logger.debug(
                "Fetched %d candles for %s (%s) with indicators",
                len(df), asset, timeframe,
            )
            return df

        except Exception as e:
            logger.warning("Failed to fetch data for %s (%s): %s", asset, timeframe, e)
            return None

    @property
    def callback(self) -> Callable:
        """Convenience property returning the callback for the orchestrator."""
        return self.get_data

    # ------------------------------------------------------------------
    # Private fetch methods
    # ------------------------------------------------------------------

    def _get_ccxt_exchange(self):
        """Lazy-init the default CCXT exchange."""
        if self._ccxt_exchange is None:
            import ccxt

            exchange_class = getattr(ccxt, self._default_exchange, None)
            if exchange_class is None:
                raise ValueError(
                    f"Unknown ccxt exchange: {self._default_exchange}"
                )
            config = {"enableRateLimit": True, "timeout": 30000}
            config.update(self._ccxt_config)
            self._ccxt_exchange = exchange_class(config)
        return self._ccxt_exchange

    def _get_oanda_exchange(self):
        """Lazy-init OANDA exchange via ccxt."""
        if self._oanda_exchange is None:
            import ccxt

            config = {"enableRateLimit": True, "timeout": 30000}
            config.update(self._oanda_config)
            self._oanda_exchange = ccxt.oanda(config)
        return self._oanda_exchange

    def _fetch_crypto(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch crypto OHLCV via ccxt."""
        import ccxt as ccxt_lib

        exchange = self._get_ccxt_exchange()

        try:
            ohlcv = exchange.fetch_ohlcv(
                asset, timeframe=timeframe, limit=self._min_candles
            )
        except (ccxt_lib.NetworkError, ccxt_lib.ExchangeNotAvailable) as e:
            logger.warning("Network error fetching %s from %s: %s",
                           asset, self._default_exchange, e)
            return None
        except ccxt_lib.BadSymbol:
            logger.warning("Symbol %s not found on %s", asset, self._default_exchange)
            return None

        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _fetch_stock(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch stock OHLCV via yfinance."""
        import yfinance as yf

        interval = _YFINANCE_INTERVAL_MAP.get(timeframe, "1h")
        period = _YFINANCE_PERIOD_MAP.get(timeframe, "730d")

        ticker = yf.Ticker(asset)
        hist = ticker.history(period=period, interval=interval)

        if hist is None or hist.empty:
            return None

        df = hist.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        # Ensure we have at least min_candles rows (trim from front if more)
        if len(df) > self._min_candles:
            df = df.iloc[-self._min_candles:]

        return df

    def _fetch_forex(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch forex OHLCV via yfinance or OANDA."""
        if self._use_oanda_for_forex:
            return self._fetch_forex_oanda(asset, timeframe)
        return self._fetch_forex_yfinance(asset, timeframe)

    def _fetch_forex_yfinance(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch forex via yfinance using the EURUSD=X ticker format."""
        import yfinance as yf

        # Convert EUR/USD -> EURUSD=X
        base, quote = asset.split("/")
        yf_symbol = f"{base}{quote}=X"

        interval = _YFINANCE_INTERVAL_MAP.get(timeframe, "1h")
        period = _YFINANCE_PERIOD_MAP.get(timeframe, "730d")

        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=period, interval=interval)

        if hist is None or hist.empty:
            return None

        df = hist.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        if len(df) > self._min_candles:
            df = df.iloc[-self._min_candles:]

        return df

    def _fetch_forex_oanda(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch forex via OANDA through ccxt."""
        import ccxt as ccxt_lib

        exchange = self._get_oanda_exchange()

        # OANDA uses underscore format: EUR_USD
        base, quote = asset.split("/")
        oanda_symbol = f"{base}_{quote}"

        try:
            ohlcv = exchange.fetch_ohlcv(
                oanda_symbol, timeframe=timeframe, limit=self._min_candles
            )
        except (ccxt_lib.NetworkError, ccxt_lib.ExchangeNotAvailable) as e:
            logger.warning("Network error fetching %s from OANDA: %s", asset, e)
            return None
        except ccxt_lib.BadSymbol:
            logger.warning("Symbol %s not found on OANDA", asset)
            return None

        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
        logger.debug("Market data cache cleared")

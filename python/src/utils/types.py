"""Shared types for inter-agent communication."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Direction(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TradeStatus(Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


class AssetClass(Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"


class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class Signal:
    """A trading signal from any analysis agent."""
    asset: str
    direction: Direction
    confidence: float  # 0-100
    source: str  # "technical", "sentiment", "ensemble"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeframe: str = "1h"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeProposal:
    """A proposed trade submitted to the risk gate."""
    signal: Signal
    asset: str
    asset_class: AssetClass
    direction: Direction
    entry_price: float
    position_size: float  # In units of the asset
    position_value: float  # In USD
    stop_loss: float
    take_profit: float
    order_type: OrderType = OrderType.LIMIT
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    """Risk gate decision on a trade proposal."""
    approved: bool
    proposal: TradeProposal
    adjusted_size: Optional[float] = None  # Risk-adjusted position size
    adjusted_stop: Optional[float] = None  # Risk-adjusted stop-loss
    reason: str = ""
    risk_score: float = 0.0  # 0-100, higher = riskier
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeResult:
    """Result of an executed trade."""
    proposal: TradeProposal
    status: TradeStatus
    fill_price: float = 0.0
    fill_size: float = 0.0
    slippage: float = 0.0  # As percentage
    fees: float = 0.0
    exchange: str = ""
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


@dataclass
class Position:
    """An open position being tracked."""
    asset: str
    asset_class: AssetClass
    side: str  # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    size: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    order_id: str = ""
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioState:
    """Current portfolio snapshot."""
    total_value: float
    cash: float
    positions: List[Position] = field(default_factory=list)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketData:
    """OHLCV market data for a single candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    asset: str = ""
    timeframe: str = "1h"


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    asset: str
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    source: str  # "finbert", "llm", "social", "composite"
    confidence: float  # 0-1
    sample_size: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FearGreedIndex:
    """Custom Fear & Greed composite index."""
    value: int  # 0-100 (extreme fear → extreme greed)
    label: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    components: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

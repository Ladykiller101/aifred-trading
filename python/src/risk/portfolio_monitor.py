"""Portfolio exposure tracking and limit enforcement."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.types import AssetClass, Position, PortfolioState

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    """Tracks open positions and enforces portfolio exposure limits."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        risk_cfg = config.get("risk", config)
        self.max_positions = risk_cfg.get("max_concurrent_positions", 10)
        self.max_asset_class_pct = risk_cfg.get("max_asset_class_exposure_pct", 30.0) / 100.0
        self.max_single_asset_pct = risk_cfg.get("max_single_asset_exposure_pct", 20.0) / 100.0

        self._positions: List[Position] = []
        self._total_value: float = 0.0
        self._cash: float = 0.0
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._total_pnl: float = 0.0

    def set_portfolio_value(self, total_value: float, cash: float) -> None:
        """Set the current portfolio value and cash balance."""
        self._total_value = total_value
        self._cash = cash

    def set_pnl(self, daily: float, weekly: float, total: float) -> None:
        """Update P&L tracking values."""
        self._daily_pnl = daily
        self._weekly_pnl = weekly
        self._total_pnl = total

    def add_position(self, position: Position) -> None:
        """Add a new tracked position."""
        self._positions.append(position)
        logger.info("Added position: %s %s %.4f units", position.side, position.asset, position.size)

    def remove_position(self, asset: str, order_id: str = "") -> Optional[Position]:
        """Remove a closed position by asset name (and optionally order_id)."""
        for i, p in enumerate(self._positions):
            if p.asset == asset and (not order_id or p.order_id == order_id):
                removed = self._positions.pop(i)
                logger.info("Removed position: %s %s", removed.side, removed.asset)
                return removed
        return None

    def update_position_price(self, asset: str, current_price: float) -> None:
        """Update current price and unrealized P&L for a position."""
        for p in self._positions:
            if p.asset == asset:
                p.current_price = current_price
                if p.side.upper() == "LONG":
                    p.unrealized_pnl = (current_price - p.entry_price) * p.size
                else:
                    p.unrealized_pnl = (p.entry_price - current_price) * p.size

    @property
    def open_positions(self) -> List[Position]:
        return list(self._positions)

    @property
    def position_count(self) -> int:
        return len(self._positions)

    def exposure_by_asset_class(self) -> Dict[str, float]:
        """Calculate total exposure by asset class as fraction of portfolio."""
        if self._total_value <= 0:
            return {}
        exposure: Dict[str, float] = defaultdict(float)
        for p in self._positions:
            value = abs(p.current_price * p.size)
            exposure[p.asset_class.value] += value
        return {k: v / self._total_value for k, v in exposure.items()}

    def exposure_by_asset(self) -> Dict[str, float]:
        """Calculate exposure per individual asset as fraction of portfolio."""
        if self._total_value <= 0:
            return {}
        exposure: Dict[str, float] = defaultdict(float)
        for p in self._positions:
            value = abs(p.current_price * p.size)
            exposure[p.asset] += value
        return {k: v / self._total_value for k, v in exposure.items()}

    def can_add_position(
        self,
        asset: str,
        asset_class: AssetClass,
        position_value: float,
    ) -> tuple:
        """Check if a new position can be added within limits.

        Args:
            asset: Asset symbol.
            asset_class: Asset class.
            position_value: Proposed position value in USD.

        Returns:
            (allowed: bool, reason: str)
        """
        # Check max concurrent positions
        if self.position_count >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        if self._total_value <= 0:
            return False, "Portfolio value not set"

        # Check asset class exposure
        class_exposure = self.exposure_by_asset_class()
        current_class = class_exposure.get(asset_class.value, 0.0)
        new_class = current_class + (position_value / self._total_value)
        if new_class > self.max_asset_class_pct:
            return False, (
                f"Asset class {asset_class.value} exposure would be "
                f"{new_class:.1%} (max {self.max_asset_class_pct:.1%})"
            )

        # Check single asset exposure
        asset_exposure = self.exposure_by_asset()
        current_asset = asset_exposure.get(asset, 0.0)
        new_asset = current_asset + (position_value / self._total_value)
        if new_asset > self.max_single_asset_pct:
            return False, (
                f"Asset {asset} exposure would be "
                f"{new_asset:.1%} (max {self.max_single_asset_pct:.1%})"
            )

        return True, "OK"

    def get_state(self) -> PortfolioState:
        """Return current portfolio state snapshot."""
        return PortfolioState(
            total_value=self._total_value,
            cash=self._cash,
            positions=list(self._positions),
            daily_pnl=self._daily_pnl,
            weekly_pnl=self._weekly_pnl,
            total_pnl=self._total_pnl,
            timestamp=datetime.utcnow(),
        )

    def set_max_positions(self, n: int) -> None:
        """Override max positions (used by volatility regime adjustments)."""
        self.max_positions = n
        logger.info("Max positions adjusted to %d", n)

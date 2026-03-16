"""Risk Management Agent -- capital preservation first, profit second.

This module provides the top-level RiskManagementAgent class that
orchestrates all risk subsystems: position sizing, stop management,
portfolio monitoring, volatility regime detection, drawdown management,
correlation tracking, and the risk gate.
"""

import logging
from typing import Any, Dict, List, Optional

from src.utils.types import (
    Position,
    PortfolioState,
    RiskDecision,
    TradeProposal,
    VolatilityRegime,
)
from src.risk.correlation_tracker import CorrelationTracker
from src.risk.drawdown_manager import DrawdownManager
from src.risk.portfolio_monitor import PortfolioMonitor
from src.risk.position_sizer import calculate_position_size, adjust_for_volatility_regime
from src.risk.risk_gate import RiskGate
from src.risk.risk_metrics import calculate_all_metrics, rolling_metrics
from src.risk.stop_manager import (
    calculate_stop_loss,
    calculate_take_profit,
    update_trailing_stop,
    check_hard_max_loss,
    calculate_bb_middle_exit,
    check_time_based_stop,
    calculate_partial_take_profit,
)
from src.risk.volatility_regime import (
    detect_regime,
    get_regime_adjustments,
    calculate_regime_score,
    RegimeTransitionDetector,
)

logger = logging.getLogger(__name__)


class RiskManagementAgent:
    """Top-level risk management agent that ties all subsystems together.

    Capital preservation first, profit second.
    If in doubt, reject the trade.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.portfolio_monitor = PortfolioMonitor(self.config)
        self.drawdown_manager = DrawdownManager(self.config)
        self.correlation_tracker = CorrelationTracker(self.config)
        self.risk_gate = RiskGate(
            portfolio_monitor=self.portfolio_monitor,
            drawdown_manager=self.drawdown_manager,
            correlation_tracker=self.correlation_tracker,
            config=self.config,
        )
        self._current_regime = VolatilityRegime.NORMAL
        self._regime_detector = RegimeTransitionDetector()
        self._returns_history: List[float] = []
        self._equity_curve: List[float] = []
        self._trade_results: List[float] = []

    def initialize(self, portfolio_value: float, cash: float) -> None:
        """Initialize the risk management system with starting values.

        Args:
            portfolio_value: Total portfolio value in USD.
            cash: Available cash in USD.
        """
        self.portfolio_monitor.set_portfolio_value(portfolio_value, cash)
        self.drawdown_manager.initialize(portfolio_value)
        self._equity_curve.append(portfolio_value)
        logger.info("Risk management initialized: portfolio=$%.2f cash=$%.2f",
                     portfolio_value, cash)

    def evaluate_trade(self, proposal: TradeProposal) -> RiskDecision:
        """Evaluate a trade proposal through the risk gate.

        This is the primary entry point. No trade should execute
        without passing through this method.

        Also checks drawdown manager heat check state -- if heat check
        is active, the risk gate will enforce A+ only mode.

        Args:
            proposal: The proposed trade.

        Returns:
            RiskDecision indicating approval/rejection with reasoning.
        """
        # If heat check is active, enforce through risk gate metadata
        if self.drawdown_manager.is_heat_check_active:
            from src.risk.risk_gate import classify_signal_tier
            tier = classify_signal_tier(proposal.confidence)
            if tier != "A+":
                return RiskDecision(
                    approved=False,
                    proposal=proposal,
                    reason=(
                        f"REJECTED: Heat check active (last 5 trades net negative). "
                        f"Only A+ signals allowed, got {tier} tier at "
                        f"{proposal.confidence:.0f}% confidence"
                    ),
                    risk_score=80.0,
                )

        portfolio = self.portfolio_monitor.get_state()
        return self.risk_gate.evaluate(proposal, portfolio)

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state snapshot."""
        return self.portfolio_monitor.get_state()

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics across rolling windows.

        Returns:
            Dict with metrics for 7d, 30d, 90d, and all-time windows.
        """
        if not self._returns_history or not self._equity_curve:
            return {"status": "insufficient_data"}
        return rolling_metrics(
            self._returns_history,
            self._equity_curve,
            self._trade_results,
        )

    def is_trading_paused(self) -> bool:
        """Check if trading is currently paused due to drawdown or cooldown."""
        return self.drawdown_manager.is_paused

    def update_position(
        self,
        position: Position,
        current_price: float,
        atr: float = 0.0,
        bb_middle: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update a position with the current market price.

        Updates P&L tracking, trailing stops, and checks for:
        - Partial take-profit at 1R
        - BB-middle exit for mean reversion strategies
        - Time-based stop for capital lock-up prevention

        Args:
            position: The position to update.
            current_price: Current market price.
            atr: Current ATR for trailing stop updates.
            bb_middle: Current Bollinger Band middle (for mean reversion exits).

        Returns:
            Dict with updated stop, and any exit signals triggered.
        """
        self.portfolio_monitor.update_position_price(position.asset, current_price)

        result = {
            "new_stop": position.stop_loss,
            "partial_tp": None,
            "bb_exit": None,
            "time_stop": None,
        }

        if atr > 0:
            result["new_stop"] = update_trailing_stop(position, current_price, atr, self.config)

        # Check partial take-profit at 1R
        if atr > 0:
            partial = calculate_partial_take_profit(position, current_price, atr, self.config)
            if partial:
                result["partial_tp"] = partial

        # Check BB-middle exit for mean reversion
        if bb_middle is not None:
            bb_exit = calculate_bb_middle_exit(
                position.entry_price, current_price, bb_middle,
                position.side, position.strategy,
            )
            if bb_exit is not None:
                result["bb_exit"] = bb_exit

        # Check time-based stop
        time_stop, time_reason = check_time_based_stop(position)
        if time_stop:
            result["time_stop"] = time_reason

        return result

    def record_trade_close(self, pnl: float, portfolio_value: float) -> None:
        """Record a closed trade and update tracking.

        Args:
            pnl: Trade P&L in USD.
            portfolio_value: Portfolio value after trade close.
        """
        self._trade_results.append(pnl)
        self._equity_curve.append(portfolio_value)
        if len(self._equity_curve) >= 2:
            ret = (self._equity_curve[-1] / self._equity_curve[-2]) - 1.0
            self._returns_history.append(ret)
        self.drawdown_manager.update(portfolio_value)
        self.drawdown_manager.record_trade_result(pnl)
        # Update risk gate streak tracking
        self.risk_gate.record_trade_outcome(pnl)

    def update_volatility_regime(
        self,
        vix: Optional[float] = None,
        atr_values: Optional[List[float]] = None,
        fear_greed_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update the current volatility regime.

        Returns regime info including score and any transition alerts.

        Args:
            vix: Current VIX value (for stocks).
            atr_values: Historical ATR values (for crypto/forex).
            fear_greed_index: Fear & Greed index 0-100.

        Returns:
            Dict with regime, score, and optional transition alert.
        """
        self._current_regime = detect_regime(
            vix=vix, atr_values=atr_values,
            fear_greed_index=fear_greed_index, config=self.config,
        )
        self.risk_gate.set_volatility_regime(self._current_regime)

        # Calculate composite regime score
        score = calculate_regime_score(vix, atr_values, fear_greed_index)

        # Detect transitions
        transition = self._regime_detector.record(self._current_regime, score)

        result = {
            "regime": self._current_regime,
            "score": score,
            "transition_alert": transition,
        }

        if transition and transition["danger_level"] in ("HIGH", "CRITICAL"):
            logger.warning(
                "DANGEROUS REGIME TRANSITION detected: %s",
                transition["message"],
            )

        return result

    def add_position(self, position: Position) -> None:
        """Track a new open position."""
        self.portfolio_monitor.add_position(position)

    def remove_position(self, asset: str, order_id: str = "") -> None:
        """Remove a closed position from tracking."""
        self.portfolio_monitor.remove_position(asset, order_id)

    def update_correlation_data(self, returns_dict: Dict[str, float]) -> None:
        """Feed new return data to the correlation tracker.

        Args:
            returns_dict: Dict of asset -> period return.
        """
        self.correlation_tracker.bulk_update(returns_dict)

    def get_drawdown_status(self) -> Dict[str, Any]:
        """Get current drawdown status and pause information."""
        return self.drawdown_manager.get_status()

    @property
    def volatility_regime(self) -> VolatilityRegime:
        return self._current_regime

    def get_position_size(
        self,
        portfolio_value: float,
        signal_confidence: float,
        win_rate: float = 0.55,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
    ) -> float:
        """Calculate a risk-adjusted position size.

        Incorporates tier-based sizing, streak adjustments, volatility
        regime adjustments, and recovery mode multiplier.

        Args:
            portfolio_value: Total portfolio value.
            signal_confidence: Signal confidence 0-100.
            win_rate: Historical win rate 0-1.
            avg_win: Average winning trade return.
            avg_loss: Average losing trade return.

        Returns:
            Position size in USD.
        """
        streak = self.risk_gate.streak_info
        size = calculate_position_size(
            portfolio_value, signal_confidence,
            win_rate, avg_win, avg_loss, self.config,
            consecutive_wins=streak["consecutive_wins"],
            consecutive_losses=streak["consecutive_losses"],
        )
        size = adjust_for_volatility_regime(size, self._current_regime.value)
        recovery = self.drawdown_manager.get_recovery_multiplier()
        size *= recovery
        return round(size, 2)

    def get_stops(
        self, entry: float, atr: float, side: str,
    ) -> Dict[str, Any]:
        """Calculate stop-loss and take-profit levels.

        Uses regime-adaptive stop distances.

        Args:
            entry: Entry price.
            atr: Current ATR.
            side: 'LONG' or 'SHORT'.

        Returns:
            Dict with stop_loss, take_profit_levels.
        """
        return {
            "stop_loss": calculate_stop_loss(
                entry, atr, side, self.config,
                volatility_regime=self._current_regime.value,
            ),
            "take_profit_levels": calculate_take_profit(entry, atr, side, self.config),
        }

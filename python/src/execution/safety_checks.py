"""Pre- and post-execution safety checks, circuit breaker, and dry-run mode."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.utils.types import (
    AssetClass, Direction, Position, PortfolioState,
    RiskDecision, TradeProposal, TradeResult, TradeStatus,
)

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Halts execution after consecutive failures."""

    def __init__(self, max_failures: int = 3, cooldown_seconds: int = 300):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._tripped_at: Optional[datetime] = None

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._tripped_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_failures:
            self._tripped_at = datetime.utcnow()
            logger.error("Circuit breaker TRIPPED after %d consecutive failures",
                         self._consecutive_failures)

    @property
    def is_tripped(self) -> bool:
        if self._tripped_at is None:
            return False
        elapsed = (datetime.utcnow() - self._tripped_at).total_seconds()
        if elapsed > self.cooldown_seconds:
            logger.info("Circuit breaker cooldown expired, resetting")
            self._consecutive_failures = 0
            self._tripped_at = None
            return False
        return True

    @property
    def failures(self) -> int:
        return self._consecutive_failures


class SafetyChecks:
    """Pre- and post-execution safety validation."""

    def __init__(self, config: Dict[str, Any]):
        risk_config = config.get("risk", {})
        exec_config = config.get("execution", {})
        self.max_position_pct = risk_config.get("max_position_pct", 3.0)
        self.max_risk_per_trade_pct = risk_config.get("max_risk_per_trade_pct", 1.5)
        self.max_concurrent_positions = risk_config.get("max_concurrent_positions", 10)
        self.max_daily_drawdown_pct = risk_config.get("max_daily_drawdown_pct", 5.0)
        self.slippage_tolerance_pct = exec_config.get("slippage_tolerance_pct", 0.1)

        max_failures = exec_config.get("max_consecutive_failures", 3)
        self.circuit_breaker = CircuitBreaker(max_failures=max_failures)
        self.dry_run = False
        self._paused = False
        self._pause_reason = ""
        self._valid_assets: Optional[List[str]] = None

    def set_valid_assets(self, assets: List[str]) -> None:
        """Set the list of tradeable assets."""
        self._valid_assets = [a.upper() for a in assets]

    def pause(self, reason: str = "") -> None:
        self._paused = True
        self._pause_reason = reason
        logger.warning("Execution PAUSED: %s", reason)

    def resume(self) -> None:
        self._paused = False
        self._pause_reason = ""
        logger.info("Execution RESUMED")

    # --- Pre-execution checks ---

    def pre_execution_check(self, proposal: TradeProposal,
                            risk_decision: RiskDecision,
                            portfolio: PortfolioState,
                            connector) -> tuple[bool, str]:
        """Run all pre-execution safety checks.

        Returns (passed, reason) where reason explains failure.
        """
        checks = [
            self._check_paused,
            self._check_circuit_breaker,
            lambda p, r, pf, c: self._check_approved(p, r),
            lambda p, r, pf, c: self._check_valid_asset(p),
            lambda p, r, pf, c: self._check_sufficient_balance(p, pf, c),
            lambda p, r, pf, c: self._check_position_limits(p, pf),
            lambda p, r, pf, c: self._check_drawdown_limits(pf),
        ]

        for check in checks:
            passed, reason = check(proposal, risk_decision, portfolio, connector)
            if not passed:
                logger.warning("Pre-execution check FAILED: %s (asset=%s)", reason, proposal.asset)
                return False, reason

        if self.dry_run:
            logger.info("[DRY RUN] All pre-checks passed for %s %s %.6f @ %.4f",
                        proposal.direction.value, proposal.asset,
                        proposal.position_size, proposal.entry_price)
            return False, "dry_run_mode"

        return True, ""

    def _check_paused(self, proposal, risk_decision, portfolio, connector) -> tuple[bool, str]:
        if self._paused:
            return False, f"execution_paused: {self._pause_reason}"
        return True, ""

    def _check_circuit_breaker(self, proposal, risk_decision, portfolio, connector) -> tuple[bool, str]:
        if self.circuit_breaker.is_tripped:
            return False, "circuit_breaker_tripped"
        return True, ""

    def _check_approved(self, proposal: TradeProposal,
                        risk_decision: RiskDecision) -> tuple[bool, str]:
        if not risk_decision.approved:
            return False, f"risk_rejected: {risk_decision.reason}"
        return True, ""

    def _check_valid_asset(self, proposal: TradeProposal) -> tuple[bool, str]:
        if self._valid_assets is not None:
            if proposal.asset.upper() not in self._valid_assets:
                return False, f"invalid_asset: {proposal.asset}"
        return True, ""

    def _check_sufficient_balance(self, proposal: TradeProposal,
                                  portfolio: PortfolioState,
                                  connector) -> tuple[bool, str]:
        required = proposal.position_value
        if required > portfolio.cash:
            return False, (f"insufficient_balance: need {required:.2f}, "
                           f"have {portfolio.cash:.2f}")
        return True, ""

    def _check_position_limits(self, proposal: TradeProposal,
                               portfolio: PortfolioState) -> tuple[bool, str]:
        if len(portfolio.positions) >= self.max_concurrent_positions:
            return False, (f"max_positions_reached: "
                           f"{len(portfolio.positions)}/{self.max_concurrent_positions}")

        if portfolio.total_value > 0:
            position_pct = (proposal.position_value / portfolio.total_value) * 100
            if position_pct > self.max_position_pct:
                return False, (f"position_too_large: {position_pct:.1f}% > "
                               f"{self.max_position_pct}%")
        return True, ""

    def _check_drawdown_limits(self, portfolio: PortfolioState) -> tuple[bool, str]:
        if portfolio.current_drawdown > self.max_daily_drawdown_pct:
            return False, (f"drawdown_limit_exceeded: {portfolio.current_drawdown:.1f}% > "
                           f"{self.max_daily_drawdown_pct}%")
        return True, ""

    # --- Post-execution checks ---

    def post_execution_check(self, result: TradeResult,
                             proposal: TradeProposal) -> tuple[bool, str]:
        """Validate execution result and flag issues."""
        if result.status == TradeStatus.FAILED:
            self.circuit_breaker.record_failure()
            return False, f"order_failed: {result.error}"

        if result.status == TradeStatus.CANCELLED:
            return False, "order_cancelled"

        self.circuit_breaker.record_success()

        # Check slippage
        if proposal.entry_price > 0:
            slippage_pct = abs(result.fill_price - proposal.entry_price) / proposal.entry_price * 100
            if slippage_pct > self.slippage_tolerance_pct:
                logger.warning("High slippage on %s: %.4f%% (tolerance: %.4f%%)",
                               proposal.asset, slippage_pct, self.slippage_tolerance_pct)

        # Verify fill
        if result.fill_size <= 0:
            return False, "zero_fill_size"

        return True, ""

    def should_place_stop_loss(self, proposal: TradeProposal) -> bool:
        """Determine if a stop-loss order should be placed immediately after fill."""
        return proposal.stop_loss > 0

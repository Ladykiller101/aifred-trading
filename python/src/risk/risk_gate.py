"""The Risk Gate -- no trade executes without approval from this module."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.types import (
    AssetClass,
    PortfolioState,
    RiskDecision,
    TradeProposal,
    VolatilityRegime,
)
from src.risk.correlation_tracker import CorrelationTracker
from src.risk.drawdown_manager import DrawdownManager
from src.risk.portfolio_monitor import PortfolioMonitor
from src.risk.position_sizer import adjust_for_volatility_regime, calculate_position_size
from src.risk.stop_manager import check_hard_max_loss
from src.risk.volatility_regime import detect_regime, get_regime_adjustments

logger = logging.getLogger(__name__)

# Signal tier definitions: minimum confidence thresholds
SIGNAL_TIERS = {
    "A+": 85,   # Elite setups only
    "A":  75,   # High-quality setups
    "B":  60,   # Medium quality -- rejected by default
    "C":  0,    # Low quality -- always rejected
}

# Volatility regime-based confidence adjustments
REGIME_CONFIDENCE_ADJUSTMENTS = {
    "low": -5,      # Can trade slightly lower confidence
    "normal": 0,    # Standard
    "high": +10,    # Require higher confidence
    "extreme": +20, # Much higher bar
}


def classify_signal_tier(confidence: float) -> str:
    """Classify a signal into a tier based on confidence."""
    if confidence >= SIGNAL_TIERS["A+"]:
        return "A+"
    elif confidence >= SIGNAL_TIERS["A"]:
        return "A"
    elif confidence >= SIGNAL_TIERS["B"]:
        return "B"
    return "C"


class RiskGate:
    """The gatekeeper -- every trade must pass through this class.

    Evaluates trade proposals against all risk rules and returns
    an approval/rejection decision with detailed reasoning.
    """

    def __init__(
        self,
        portfolio_monitor: PortfolioMonitor,
        drawdown_manager: DrawdownManager,
        correlation_tracker: CorrelationTracker,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.portfolio = portfolio_monitor
        self.drawdown = drawdown_manager
        self.correlation = correlation_tracker
        self.config = config or {}
        self._risk_cfg = self.config.get("risk", self.config)

        self._current_regime: VolatilityRegime = VolatilityRegime.NORMAL
        self._decision_log: List[Dict[str, Any]] = []

        # Consecutive loss tracking for cooldown
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0

    def set_volatility_regime(self, regime: VolatilityRegime) -> None:
        """Set the current volatility regime."""
        self._current_regime = regime
        adjustments = get_regime_adjustments(regime)
        self.portfolio.set_max_positions(adjustments["max_positions"])

    def record_trade_outcome(self, pnl: float) -> None:
        """Record a trade outcome for consecutive win/loss tracking.

        Args:
            pnl: Trade P&L (negative = loss).
        """
        if pnl < 0:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
        else:
            self._consecutive_wins += 1
            self._consecutive_losses = 0

    def _check_signal_tier(self, proposal: TradeProposal) -> tuple:
        """Check signal tier gating. Only A+ and A tier trades are allowed.

        Returns:
            (passed: bool, tier: str, reason: str)
        """
        tier = classify_signal_tier(proposal.confidence)

        # After 3 consecutive losses, require A+ with >85% confidence
        if self._consecutive_losses >= 3:
            if tier != "A+" or proposal.confidence <= 85:
                return False, tier, (
                    f"Consecutive loss cooldown active ({self._consecutive_losses} losses): "
                    f"requires A+ tier with >85% confidence, got {tier} tier at "
                    f"{proposal.confidence:.0f}%"
                )

        # Reject B and C tier signals entirely
        if tier in ("B", "C"):
            return False, tier, (
                f"Signal tier {tier} rejected (confidence {proposal.confidence:.0f}%): "
                f"only A+ and A tier trades are allowed"
            )

        return True, tier, f"{tier} tier signal at {proposal.confidence:.0f}% confidence"

    def _check_kill_zone(self, proposal: TradeProposal) -> tuple:
        """Compute a confidence multiplier based on time-of-day for this asset class.

        Instead of rejecting trades outright, returns a confidence multiplier
        that scales the signal strength based on historically optimal trading hours.

        Returns:
            (passed: bool, confidence_multiplier: float, reason: str)
        """
        now = datetime.utcnow()
        hour = now.hour
        asset_class = proposal.asset_class

        if asset_class == AssetClass.CRYPTO:
            # Crypto is 24/7 -- never reject, but slightly reduce confidence
            # during historically weaker 01-05 UTC window
            if 1 <= hour < 5:
                return True, 0.90, (
                    f"Crypto reduced confidence (0.90x) during 01-05 UTC window "
                    f"(hour {hour:02d} UTC)"
                )
            return True, 1.0, f"Crypto trading at full confidence ({hour:02d} UTC)"

        if asset_class == AssetClass.STOCKS:
            # Primary hours 13-20 UTC get a boost; other hours are reduced
            if 13 <= hour < 20:
                return True, 1.10, (
                    f"Stock primary hours boost (1.10x) during 13-20 UTC "
                    f"(hour {hour:02d} UTC)"
                )
            return True, 0.85, (
                f"Stock off-hours reduction (0.85x) outside 13-20 UTC "
                f"(hour {hour:02d} UTC)"
            )

        if asset_class == AssetClass.FOREX:
            # Kill zones (London 07-10, New York 12-15) get a boost
            if (7 <= hour < 10) or (12 <= hour < 15):
                return True, 1.15, (
                    f"Forex kill zone boost (1.15x) during optimal hours "
                    f"(hour {hour:02d} UTC)"
                )
            return True, 0.85, (
                f"Forex off-hours reduction (0.85x) outside kill zones "
                f"(hour {hour:02d} UTC)"
            )

        # Unknown asset class -- no adjustment
        return True, 1.0, f"No time-of-day adjustment for {asset_class.value}"

    def _check_momentum_filter(self, proposal: TradeProposal) -> tuple:
        """Reject counter-trend trades when ADX > 30 (strong trend).

        Returns:
            (passed: bool, reason: str)
        """
        adx = proposal.metadata.get("adx")
        trend_direction = proposal.metadata.get("trend_direction")

        if adx is None or trend_direction is None:
            return True, "No ADX/trend data available, momentum filter skipped"

        if adx <= 30:
            return True, f"ADX={adx:.1f} (weak/no trend), counter-trend trades allowed"

        # ADX > 30: strong trend. Check if trade is counter-trend.
        is_bullish_trend = trend_direction in ("up", "bullish", "LONG")
        is_buy = proposal.direction.value in ("BUY", "STRONG_BUY")

        if (is_bullish_trend and not is_buy) or (not is_bullish_trend and is_buy):
            return False, (
                f"Momentum filter: ADX={adx:.1f} indicates strong "
                f"{'bullish' if is_bullish_trend else 'bearish'} trend, "
                f"rejecting counter-trend {proposal.direction.value} trade"
            )

        return True, (
            f"ADX={adx:.1f} confirms trend alignment with "
            f"{proposal.direction.value} direction"
        )

    def evaluate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """Evaluate a trade proposal against all risk rules.

        Args:
            proposal: The trade proposal to evaluate.
            portfolio: Current portfolio state.

        Returns:
            RiskDecision with approval/rejection and detailed reasoning.
        """
        reasons = []
        risk_score = 0.0
        adjusted_size = proposal.position_value

        # 1. Check drawdown pause
        paused, pause_reason = self.drawdown.check_pause_rules()
        if paused:
            return self._reject(proposal, f"REJECTED: Trading paused: {pause_reason}", risk_score=100.0)

        # 2. Signal tier gating (highest impact filter)
        tier_ok, tier, tier_reason = self._check_signal_tier(proposal)
        if not tier_ok:
            return self._reject(proposal, f"REJECTED: {tier_reason}", risk_score=95.0)
        reasons.append(tier_reason)

        # 3. Time-of-day confidence scaling (never rejects, only adjusts confidence)
        kz_ok, kz_confidence_mult, kz_reason = self._check_kill_zone(proposal)
        reasons.append(kz_reason)

        # 4. Momentum filter (ADX counter-trend rejection)
        momentum_ok, momentum_reason = self._check_momentum_filter(proposal)
        if not momentum_ok:
            return self._reject(proposal, f"REJECTED: {momentum_reason}", risk_score=65.0)
        reasons.append(momentum_reason)

        # 5. Check volatility regime
        adjustments = get_regime_adjustments(self._current_regime)
        if adjustments["action"] == "close_all_or_hedge":
            return self._reject(
                proposal,
                f"REJECTED: Extreme volatility regime ({self._current_regime.value}): no new trades allowed",
                risk_score=100.0,
            )
        if adjustments["position_size_multiplier"] < 1.0:
            adjusted_size *= adjustments["position_size_multiplier"]
            reasons.append(
                f"Size reduced by {(1 - adjustments['position_size_multiplier']):.0%} "
                f"due to {self._current_regime.value} volatility"
            )
            risk_score += 20.0

        # 6. In HIGH volatility regime, require A+ signals only
        if self._current_regime == VolatilityRegime.HIGH and tier != "A+":
            return self._reject(
                proposal,
                f"REJECTED: High volatility regime requires A+ signals only, got {tier} tier",
                risk_score=85.0,
            )

        # 7. Apply recovery mode multiplier
        recovery_mult = self.drawdown.get_recovery_multiplier()
        if recovery_mult < 1.0:
            adjusted_size *= recovery_mult
            reasons.append(f"Recovery mode: size multiplied by {recovery_mult:.2f}")
            risk_score += 15.0

        # 8. Check portfolio exposure limits
        can_add, exposure_reason = self.portfolio.can_add_position(
            proposal.asset, proposal.asset_class, adjusted_size,
        )
        if not can_add:
            return self._reject(
                proposal,
                f"REJECTED: Exposure limit: {exposure_reason}",
                risk_score=80.0,
            )
        reasons.append(f"Portfolio exposure within limits")

        # 9. Check correlation limits
        held_assets = [p.asset for p in portfolio.positions]
        corr_ok, corr_reason = self.correlation.check_correlation_limit(
            proposal.asset, held_assets,
        )
        if not corr_ok:
            return self._reject(
                proposal,
                f"REJECTED: Correlation limit: {corr_reason}",
                risk_score=70.0,
            )
        reasons.append("No correlation conflict")

        # 10. Verify stop-loss is set
        if proposal.stop_loss <= 0:
            return self._reject(
                proposal,
                "REJECTED: No stop-loss set on proposal",
                risk_score=90.0,
            )

        # 11. Check hard max loss per trade
        stop_ok = check_hard_max_loss(
            proposal.entry_price,
            proposal.stop_loss,
            adjusted_size,
            portfolio.total_value,
            self.config,
        )
        if not stop_ok:
            max_risk_pct = self._risk_cfg.get("max_risk_per_trade_pct", 1.5) / 100.0
            max_risk_usd = portfolio.total_value * max_risk_pct
            stop_distance_pct = abs(proposal.entry_price - proposal.stop_loss) / proposal.entry_price
            if stop_distance_pct > 0:
                max_allowed_size = max_risk_usd / stop_distance_pct
                if max_allowed_size >= adjusted_size * 0.25:
                    adjusted_size = max_allowed_size
                    reasons.append(
                        f"Size reduced to ${adjusted_size:.2f} to respect "
                        f"{max_risk_pct:.1%} max risk per trade"
                    )
                    risk_score += 25.0
                else:
                    return self._reject(
                        proposal,
                        f"REJECTED: Stop-loss too wide: risk would be "
                        f"{stop_distance_pct:.2%} of position, "
                        f"exceeds max risk budget even at minimum size",
                        risk_score=85.0,
                    )

        # 12. Apply time-of-day confidence multiplier from kill zone check
        effective_confidence = proposal.confidence * kz_confidence_mult

        # 13. Volatility regime-based confidence threshold adjustment
        regime_key = self._current_regime.value
        regime_adj = REGIME_CONFIDENCE_ADJUSTMENTS.get(regime_key, 0)
        if regime_adj != 0:
            reasons.append(
                f"Regime confidence adjustment: {regime_adj:+d} points "
                f"({regime_key} volatility)"
            )

        # Apply regime adjustment to minimum confidence thresholds
        # by effectively raising/lowering the bar for the confidence check
        adjusted_confidence_for_check = effective_confidence - regime_adj

        # 14. Confidence-based risk scoring (using adjusted confidence)
        if adjusted_confidence_for_check < 50:
            risk_score += 15.0
            reasons.append(
                f"Low confidence warning: {effective_confidence:.0f}% "
                f"(effective, after {kz_confidence_mult:.2f}x time-of-day scaling)"
            )
        elif adjusted_confidence_for_check < 70:
            risk_score += 5.0

        # Build detailed reasoning string
        risk_score = min(risk_score, 100.0)

        # Calculate adjusted size as units (not USD)
        adjusted_units = None
        if adjusted_size != proposal.position_value and proposal.entry_price > 0:
            adjusted_units = adjusted_size / proposal.entry_price

        # Construct detailed approval reasoning
        regime_str = self._current_regime.value
        streak_info = ""
        if self._consecutive_wins > 0:
            streak_info = f", {self._consecutive_wins} consecutive wins"
        elif self._consecutive_losses > 0:
            streak_info = f", {self._consecutive_losses} consecutive losses"

        approval_summary = (
            f"Approved: {tier} tier, {proposal.confidence:.0f}% confidence, "
            f"risk score {risk_score:.1f}/100, {regime_str} volatility regime"
            f"{streak_info}"
        )
        reason_str = f"{approval_summary} | Details: {'; '.join(reasons)}"

        decision = RiskDecision(
            approved=True,
            proposal=proposal,
            adjusted_size=adjusted_units,
            reason=reason_str,
            risk_score=risk_score,
        )

        self._log_decision(decision, "APPROVED")
        return decision

    def _reject(
        self,
        proposal: TradeProposal,
        reason: str,
        risk_score: float = 100.0,
    ) -> RiskDecision:
        """Create a rejection decision."""
        decision = RiskDecision(
            approved=False,
            proposal=proposal,
            reason=reason,
            risk_score=risk_score,
        )
        self._log_decision(decision, "REJECTED")
        return decision

    def _log_decision(self, decision: RiskDecision, status: str) -> None:
        """Log every decision with full details."""
        tier = classify_signal_tier(decision.proposal.confidence)
        entry = {
            "status": status,
            "asset": decision.proposal.asset,
            "direction": decision.proposal.direction.value,
            "signal_tier": tier,
            "confidence": decision.proposal.confidence,
            "proposed_value": decision.proposal.position_value,
            "adjusted_size": decision.adjusted_size,
            "reason": decision.reason,
            "risk_score": decision.risk_score,
            "regime": self._current_regime.value,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._decision_log.append(entry)
        log_fn = logger.info if status == "APPROVED" else logger.warning
        log_fn("Risk Gate %s: %s %s [%s tier] (score=%.1f) - %s",
               status, decision.proposal.direction.value,
               decision.proposal.asset, tier,
               decision.risk_score, decision.reason)

    @property
    def decision_history(self) -> List[Dict[str, Any]]:
        return list(self._decision_log)

    @property
    def streak_info(self) -> Dict[str, int]:
        """Get current win/loss streak information."""
        return {
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
        }

"""Position sizing using Kelly Criterion with safety adjustments."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Tier-based Kelly multipliers: what fraction of calculated Kelly size to use
TIER_KELLY_MULTIPLIERS = {
    "A+": 1.00,  # 100% of Kelly size
    "A":  0.75,  # 75% of Kelly size
    "B":  0.50,  # 50% of Kelly size (if somehow passed the gate)
    "C":  0.25,  # 25% -- should never reach here
}


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate optimal Kelly fraction.

    f* = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win / avg_loss
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p
    f = (p * b - q) / b
    return max(f, 0.0)


def _classify_tier(confidence: float) -> str:
    """Classify confidence into signal tier."""
    if confidence >= 85:
        return "A+"
    elif confidence >= 75:
        return "A"
    elif confidence >= 60:
        return "B"
    return "C"


def streak_multiplier(
    consecutive_wins: int = 0,
    consecutive_losses: int = 0,
) -> float:
    """Calculate position size multiplier based on win/loss streaks.

    Win streaks (momentum sizing):
    - After 5+ consecutive wins: +30% size
    - After 3+ consecutive wins: +20% size

    Loss streaks (protective sizing):
    - After 6+ consecutive losses: -80% size (near stop trading)
    - After 4+ consecutive losses: -60% size
    - After 2+ consecutive losses: -40% size

    Args:
        consecutive_wins: Number of consecutive winning trades.
        consecutive_losses: Number of consecutive losing trades.

    Returns:
        Multiplier to apply to position size.
    """
    if consecutive_wins >= 5:
        return 1.30  # 30% boost for extended win streak
    if consecutive_wins >= 3:
        return 1.20  # 20% boost for win streak momentum
    if consecutive_losses >= 6:
        return 0.20  # 80% reduction -- near stop trading
    if consecutive_losses >= 4:
        return 0.40  # 60% reduction for severe loss streak
    if consecutive_losses >= 2:
        return 0.60  # 40% reduction for loss streak
    return 1.0


def calculate_position_size(
    portfolio_value: float,
    signal_confidence: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    config: Optional[Dict[str, Any]] = None,
    consecutive_wins: int = 0,
    consecutive_losses: int = 0,
) -> float:
    """Calculate risk-adjusted position size in USD.

    Applies tiered sizing based on signal quality and streak adjustments.

    Args:
        portfolio_value: Total portfolio value in USD.
        signal_confidence: Signal confidence 0-100.
        win_rate: Historical win rate 0-1.
        avg_win: Average winning trade return (absolute value).
        avg_loss: Average losing trade return (absolute value).
        config: Risk config dict (from default.yaml risk section).
        consecutive_wins: Number of consecutive winning trades.
        consecutive_losses: Number of consecutive losing trades.

    Returns:
        Position size in USD.
    """
    if config is None:
        config = {}

    risk_cfg = config.get("risk", config)
    max_position_pct = risk_cfg.get("max_position_pct", 3.0) / 100.0
    max_risk_per_trade_pct = risk_cfg.get("max_risk_per_trade_pct", 1.5) / 100.0
    kelly_fraction = risk_cfg.get("kelly_fraction", 0.5)

    # Kelly optimal fraction
    full_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
    # Apply fractional Kelly (half-Kelly default)
    fractional_kelly = full_kelly * kelly_fraction

    # Confidence scaling: map 0-100 confidence to 0.2-1.0 multiplier
    confidence_norm = max(0.0, min(signal_confidence, 100.0)) / 100.0
    confidence_multiplier = 0.2 + 0.8 * confidence_norm

    # Tier-based Kelly scaling: A+ gets full Kelly, A gets 75%, etc.
    tier = _classify_tier(signal_confidence)
    tier_mult = TIER_KELLY_MULTIPLIERS.get(tier, 0.5)

    # Win/loss streak adjustment
    streak_mult = streak_multiplier(consecutive_wins, consecutive_losses)

    # Base size from Kelly * tier * streak
    kelly_size = (
        portfolio_value * fractional_kelly * confidence_multiplier
        * tier_mult * streak_mult
    )

    # Cap at max position percentage
    max_position = portfolio_value * max_position_pct
    position_size = min(kelly_size, max_position)

    # Also ensure stop-loss risk stays within max risk per trade.
    # The caller should verify this with the actual stop distance;
    # here we provide an upper bound based on the risk budget.
    max_risk_budget = portfolio_value * max_risk_per_trade_pct
    # If the position size would risk more than the budget at a 2% stop distance
    # (a conservative assumption), clamp it.
    position_size = min(position_size, max_risk_budget / 0.02) if position_size > 0 else 0.0
    position_size = min(position_size, max_position)

    # Never negative
    position_size = max(position_size, 0.0)

    logger.debug(
        "Position sizing: portfolio=%.2f kelly=%.4f tier=%s(x%.2f) "
        "streak=x%.2f conf_mult=%.2f -> size=%.2f",
        portfolio_value, fractional_kelly, tier, tier_mult,
        streak_mult, confidence_multiplier, position_size,
    )
    return round(position_size, 2)


def adjust_for_volatility_regime(
    position_size: float,
    regime: str,
) -> float:
    """Reduce position size based on volatility regime.

    Args:
        position_size: Base position size in USD.
        regime: One of 'low', 'normal', 'high', 'extreme'.

    Returns:
        Adjusted position size.
    """
    multipliers = {
        "low": 1.05,    # Slightly larger in low vol (tighter stops compensate)
        "normal": 1.0,
        "high": 0.5,    # 50% reduction in high vol
        "extreme": 0.0,
    }
    mult = multipliers.get(regime, 1.0)
    return round(position_size * mult, 2)

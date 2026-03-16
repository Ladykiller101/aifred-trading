"""Dynamic stop-loss and take-profit management based on ATR."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.utils.types import Position

logger = logging.getLogger(__name__)

# Time-based stop: close position after this many hours if not profitable
DEFAULT_TIME_STOP_HOURS = 36


def calculate_stop_loss(
    entry: float,
    atr: float,
    side: str,
    config: Optional[Dict[str, Any]] = None,
    volatility_regime: str = "normal",
) -> float:
    """Calculate initial stop-loss based on ATR with regime adaptation.

    In LOW volatility: tighter stops (1.5x ATR) since moves are smaller.
    In HIGH volatility: wider stops (2.5x ATR) to avoid noise stop-outs.

    Args:
        entry: Entry price.
        atr: Current Average True Range value.
        side: 'LONG' or 'SHORT'.
        config: Risk config dict.
        volatility_regime: Current regime ('low', 'normal', 'high', 'extreme').

    Returns:
        Stop-loss price.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    base_atr_mult = risk_cfg.get("stop_loss_atr_multiplier", 2.0)

    # Regime-adaptive ATR multiplier
    regime_adjustments = {
        "low": 0.75,      # 1.5 ATR (tighter) if base is 2.0
        "normal": 1.0,    # 2.0 ATR (standard)
        "high": 1.25,     # 2.5 ATR (wider)
        "extreme": 1.5,   # 3.0 ATR (widest)
    }
    regime_mult = regime_adjustments.get(volatility_regime, 1.0)
    atr_mult = base_atr_mult * regime_mult

    if side.upper() == "LONG":
        stop = entry - (atr * atr_mult)
    else:
        stop = entry + (atr * atr_mult)

    logger.debug(
        "Stop loss: entry=%.4f atr=%.4f side=%s mult=%.1f (regime=%s) -> stop=%.4f",
        entry, atr, side, atr_mult, volatility_regime, stop,
    )
    return round(stop, 6)


def calculate_take_profit(
    entry: float,
    atr: float,
    side: str,
    config: Optional[Dict[str, Any]] = None,
    r_multiples: Optional[list] = None,
) -> list:
    """Calculate take-profit levels as R-multiples of risk.

    Args:
        entry: Entry price.
        atr: Current Average True Range value.
        side: 'LONG' or 'SHORT'.
        config: Risk config dict.
        r_multiples: List of R-multiples for partial exits. Default [2.0, 3.0].

    Returns:
        List of take-profit prices.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    atr_mult = risk_cfg.get("stop_loss_atr_multiplier", 2.0)

    if r_multiples is None:
        r_multiples = [2.0, 3.0]

    risk_per_unit = atr * atr_mult  # Distance from entry to stop

    targets = []
    for r in r_multiples:
        if side.upper() == "LONG":
            tp = entry + (risk_per_unit * r)
        else:
            tp = entry - (risk_per_unit * r)
        targets.append(round(tp, 6))

    logger.debug(
        "Take profit levels: entry=%.4f risk=%.4f side=%s -> %s",
        entry, risk_per_unit, side, targets,
    )
    return targets


def update_trailing_stop(
    position: Position,
    current_price: float,
    atr: float,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Update trailing stop-loss for an open position.

    Rules:
    - Move to breakeven after 1x ATR profit.
    - Trail at trailing_stop_atr_multiplier x ATR once in profit.
    - Never move stop backwards (only tighten).

    Args:
        position: Current open position.
        current_price: Current market price.
        atr: Current ATR value.
        config: Risk config dict.

    Returns:
        Updated stop-loss price.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    trail_mult = risk_cfg.get("trailing_stop_atr_multiplier", 1.5)

    entry = position.entry_price
    current_stop = position.stop_loss

    if position.side.upper() == "LONG":
        profit_distance = current_price - entry
        # Move to breakeven after 1x ATR profit
        if profit_distance >= atr:
            breakeven_stop = entry
            current_stop = max(current_stop, breakeven_stop)
        # Trail at trail_mult x ATR below current price
        if profit_distance > 0:
            trailing = current_price - (atr * trail_mult)
            current_stop = max(current_stop, trailing)
    else:
        profit_distance = entry - current_price
        if profit_distance >= atr:
            breakeven_stop = entry
            current_stop = min(current_stop, breakeven_stop)
        if profit_distance > 0:
            trailing = current_price + (atr * trail_mult)
            current_stop = min(current_stop, trailing)

    new_stop = round(current_stop, 6)
    logger.debug(
        "Trailing stop update: %s side=%s entry=%.4f current=%.4f atr=%.4f -> stop=%.4f",
        position.asset, position.side, entry, current_price, atr, new_stop,
    )
    return new_stop


def check_hard_max_loss(
    entry_price: float,
    stop_loss: float,
    position_value: float,
    portfolio_value: float,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if a trade's stop-loss risk exceeds the hard max loss limit.

    Args:
        entry_price: Entry price.
        stop_loss: Stop-loss price.
        position_value: Position value in USD.
        portfolio_value: Total portfolio value in USD.
        config: Risk config dict.

    Returns:
        True if within limits, False if exceeds max loss.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    max_risk_pct = risk_cfg.get("max_risk_per_trade_pct", 1.5) / 100.0

    if entry_price <= 0:
        return False

    stop_distance_pct = abs(entry_price - stop_loss) / entry_price
    risk_amount = position_value * stop_distance_pct
    max_risk = portfolio_value * max_risk_pct

    within_limits = risk_amount <= max_risk
    if not within_limits:
        logger.warning(
            "Hard max loss exceeded: risk=%.2f max=%.2f (%.2f%% of portfolio)",
            risk_amount, max_risk, (risk_amount / portfolio_value) * 100,
        )
    return within_limits


def calculate_bb_middle_exit(
    entry: float,
    current_price: float,
    bb_middle: float,
    side: str,
    strategy: str = "",
) -> Optional[float]:
    """Calculate Bollinger Band middle exit for mean reversion strategies.

    For mean reversion trades, exit at the BB middle line instead of a fixed TP.
    This has been proven in backtesting to improve profit factor.

    Args:
        entry: Entry price.
        current_price: Current market price.
        bb_middle: Current Bollinger Band middle (SMA 20).
        side: 'LONG' or 'SHORT'.
        strategy: Strategy name to check if mean reversion.

    Returns:
        BB middle exit price, or None if not applicable.
    """
    is_mean_reversion = "mean_reversion" in strategy.lower() or "mr" in strategy.lower()
    if not is_mean_reversion:
        return None

    if side.upper() == "LONG":
        # For mean reversion longs, TP at BB middle (price should revert up)
        if bb_middle > entry:
            logger.debug(
                "BB middle exit for LONG mean reversion: entry=%.4f bb_mid=%.4f",
                entry, bb_middle,
            )
            return round(bb_middle, 6)
    else:
        # For mean reversion shorts, TP at BB middle (price should revert down)
        if bb_middle < entry:
            logger.debug(
                "BB middle exit for SHORT mean reversion: entry=%.4f bb_mid=%.4f",
                entry, bb_middle,
            )
            return round(bb_middle, 6)

    return None


def check_time_based_stop(
    position: Position,
    max_hours: int = DEFAULT_TIME_STOP_HOURS,
) -> tuple:
    """Check if a position should be closed due to time-based stop.

    Close position after max_hours if not profitable to avoid capital lock-up.

    Args:
        position: Open position to check.
        max_hours: Maximum hours to hold an unprofitable position.

    Returns:
        (should_close: bool, reason: str)
    """
    now = datetime.utcnow()
    hold_duration = now - position.entry_time
    hours_held = hold_duration.total_seconds() / 3600.0

    if hours_held >= max_hours and position.unrealized_pnl <= 0:
        return True, (
            f"Time-based stop triggered: held for {hours_held:.1f}h "
            f"(max {max_hours}h) with unrealized P&L ${position.unrealized_pnl:.2f}"
        )

    return False, "OK"


def calculate_partial_take_profit(
    position: Position,
    current_price: float,
    atr: float,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Calculate partial take-profit at 1R profit.

    At 1R profit (risk distance), close 50% of the position and move
    stop to breakeven on the remainder.

    Args:
        position: Open position.
        current_price: Current market price.
        atr: Current ATR value.
        config: Risk config dict.

    Returns:
        Dict with partial TP instructions, or None if not triggered.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    atr_mult = risk_cfg.get("stop_loss_atr_multiplier", 2.0)

    risk_distance = atr * atr_mult  # 1R = distance from entry to stop
    entry = position.entry_price

    if position.side.upper() == "LONG":
        profit_distance = current_price - entry
        one_r_target = entry + risk_distance
    else:
        profit_distance = entry - current_price
        one_r_target = entry - risk_distance

    # Check if we've reached 1R profit
    if profit_distance >= risk_distance:
        # Check metadata to avoid triggering multiple times
        if position.metadata.get("partial_tp_taken"):
            return None

        return {
            "action": "partial_take_profit",
            "close_pct": 0.50,  # Close 50% of position
            "close_size": position.size * 0.50,
            "new_stop": entry,  # Move stop to breakeven
            "reason": (
                f"1R profit reached (${profit_distance:.2f} >= "
                f"${risk_distance:.2f} risk): closing 50%, "
                f"moving stop to breakeven at {entry:.4f}"
            ),
        }

    return None

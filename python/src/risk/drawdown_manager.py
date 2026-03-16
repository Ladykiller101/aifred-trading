"""Drawdown tracking, auto-pause rules, and anti-revenge-trading."""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DrawdownManager:
    """Tracks drawdown at daily/weekly/total levels and enforces pause rules."""

    # Default daily trade limit to prevent overtrading
    DEFAULT_MAX_DAILY_TRADES = 8
    # After this drawdown threshold, enter recovery mode sizing
    RECOVERY_DRAWDOWN_THRESHOLD = 0.05  # 5%
    # Number of trades at reduced size during recovery
    RECOVERY_TRADE_COUNT = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        risk_cfg = config.get("risk", config)
        self.max_daily_dd_pct = risk_cfg.get("max_daily_drawdown_pct", 5.0) / 100.0
        self.max_weekly_dd_pct = risk_cfg.get("max_weekly_drawdown_pct", 10.0) / 100.0
        self.max_daily_trades = risk_cfg.get("max_daily_trades", self.DEFAULT_MAX_DAILY_TRADES)

        self._peak_value: float = 0.0
        self._daily_start_value: float = 0.0
        self._weekly_start_value: float = 0.0
        self._current_value: float = 0.0

        self._pause_until: Optional[datetime] = None
        self._pause_reason: str = ""

        # Recovery mode tracking
        self._in_recovery: bool = False
        self._recovery_start_dd: float = 0.0
        self._recovery_trades_remaining: int = 0

        # Anti-revenge trading: track recent consecutive losses
        self._recent_results: deque = deque(maxlen=20)
        self._cooldown_until: Optional[datetime] = None

        # Daily trade count tracking
        self._daily_trade_count: int = 0
        self._last_trade_date: Optional[datetime] = None

        # Heat check: A+ only mode when recent trades are net negative
        self._a_plus_only_mode: bool = False

        # Drawdown history for analysis
        self._drawdown_log: List[Dict[str, Any]] = []

    def initialize(self, portfolio_value: float) -> None:
        """Initialize with starting portfolio value."""
        self._peak_value = portfolio_value
        self._daily_start_value = portfolio_value
        self._weekly_start_value = portfolio_value
        self._current_value = portfolio_value

    def update(self, current_value: float) -> None:
        """Update with latest portfolio value."""
        self._current_value = current_value
        if current_value > self._peak_value:
            self._peak_value = current_value

    def reset_daily(self, current_value: float) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self._daily_start_value = current_value
        self.update(current_value)

    def reset_weekly(self, current_value: float) -> None:
        """Reset weekly tracking (call at start of each trading week)."""
        self._weekly_start_value = current_value
        self.update(current_value)

    @property
    def daily_drawdown(self) -> float:
        """Current daily drawdown as positive fraction."""
        if self._daily_start_value <= 0:
            return 0.0
        dd = (self._daily_start_value - self._current_value) / self._daily_start_value
        return max(dd, 0.0)

    @property
    def weekly_drawdown(self) -> float:
        """Current weekly drawdown as positive fraction."""
        if self._weekly_start_value <= 0:
            return 0.0
        dd = (self._weekly_start_value - self._current_value) / self._weekly_start_value
        return max(dd, 0.0)

    @property
    def total_drawdown(self) -> float:
        """Drawdown from all-time peak as positive fraction."""
        if self._peak_value <= 0:
            return 0.0
        dd = (self._peak_value - self._current_value) / self._peak_value
        return max(dd, 0.0)

    def check_pause_rules(self) -> tuple:
        """Check if trading should be paused due to drawdown or limits.

        Returns:
            (should_pause: bool, reason: str)
        """
        now = datetime.utcnow()

        # Check existing pause
        if self._pause_until and now < self._pause_until:
            remaining = self._pause_until - now
            return True, f"Paused until {self._pause_until} ({self._pause_reason}), {remaining} remaining"

        # Check daily drawdown
        if self.daily_drawdown >= self.max_daily_dd_pct:
            self._pause_until = now + timedelta(hours=24)
            self._pause_reason = f"Daily drawdown {self.daily_drawdown:.2%} >= {self.max_daily_dd_pct:.2%}"
            self._log_drawdown_event("daily_pause", self.daily_drawdown)
            logger.warning("TRADING PAUSED: %s", self._pause_reason)
            return True, self._pause_reason

        # Check weekly drawdown
        if self.weekly_drawdown >= self.max_weekly_dd_pct:
            self._pause_until = now + timedelta(hours=72)
            self._pause_reason = f"Weekly drawdown {self.weekly_drawdown:.2%} >= {self.max_weekly_dd_pct:.2%}"
            self._log_drawdown_event("weekly_pause", self.weekly_drawdown)
            logger.warning("TRADING PAUSED: %s", self._pause_reason)
            return True, self._pause_reason

        # Check cooldown from anti-revenge trading
        if self._cooldown_until and now < self._cooldown_until:
            remaining = self._cooldown_until - now
            return True, f"Cooldown active (anti-revenge), {remaining} remaining"

        # Check daily trade limit
        self._refresh_daily_trade_count()
        if self._daily_trade_count >= self.max_daily_trades:
            return True, (
                f"Daily trade limit reached: {self._daily_trade_count}/{self.max_daily_trades} "
                f"trades today (prevents overtrading)"
            )

        # Clear expired pause
        if self._pause_until and now >= self._pause_until:
            logger.info("Trading pause expired, resuming")
            self._pause_until = None
            self._pause_reason = ""

        return False, "OK"

    @property
    def is_paused(self) -> bool:
        paused, _ = self.check_pause_rules()
        return paused

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result for anti-revenge trading detection.

        Also updates daily trade count and heat check state.

        Args:
            pnl: Trade P&L (negative = loss).
        """
        now = datetime.utcnow()
        self._recent_results.append({
            "pnl": pnl,
            "timestamp": now,
        })

        # Update daily trade count
        self._refresh_daily_trade_count()
        if self._last_trade_date is None or self._last_trade_date.date() != now.date():
            self._daily_trade_count = 1
        else:
            self._daily_trade_count += 1
        self._last_trade_date = now

        # Update heat check (A+ only mode)
        self._update_heat_check()

        # Check for recovery mode after >5% drawdown
        self._check_recovery_mode()

        self._check_revenge_trading()

    def _check_revenge_trading(self) -> None:
        """Detect rapid consecutive losses and force a cooldown."""
        if len(self._recent_results) < 3:
            return

        recent = list(self._recent_results)[-5:]
        consecutive_losses = 0
        for r in reversed(recent):
            if r["pnl"] < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            # Check if losses happened within a short time span
            loss_times = [
                r["timestamp"] for r in recent[-consecutive_losses:]
            ]
            time_span = loss_times[-1] - loss_times[0]
            if time_span < timedelta(hours=2):
                self._cooldown_until = datetime.utcnow() + timedelta(hours=4)
                logger.warning(
                    "ANTI-REVENGE TRADING: %d consecutive losses in %s. "
                    "Cooldown until %s",
                    consecutive_losses, time_span, self._cooldown_until,
                )
                self._log_drawdown_event("revenge_cooldown", self.daily_drawdown)

    def get_recovery_multiplier(self) -> float:
        """Get position size multiplier based on recovery mode.

        After >5% total drawdown, use 50% of normal size for the next 10 trades.
        After >10% total drawdown, gradually scale back up as recovery progresses.

        Returns:
            Multiplier 0.25-1.0.
        """
        # Recovery mode from recent drawdown (50% size for next N trades)
        if self._recovery_trades_remaining > 0:
            return 0.50

        dd = self.total_drawdown
        if dd > 0.10:
            if not self._in_recovery:
                self._in_recovery = True
                self._recovery_start_dd = dd
                logger.info("Entering deep recovery mode at %.2f%% drawdown", dd * 100)
            # Scale from 0.25 at peak drawdown to 1.0 at full recovery
            recovery_progress = 1.0 - (dd / self._recovery_start_dd) if self._recovery_start_dd > 0 else 0.0
            return 0.25 + 0.75 * max(0.0, min(recovery_progress, 1.0))

        if self._in_recovery and dd < 0.03:
            self._in_recovery = False
            logger.info("Exiting recovery mode, drawdown recovered to %.2f%%", dd * 100)

        return 1.0

    def _refresh_daily_trade_count(self) -> None:
        """Reset daily trade count if it's a new day."""
        now = datetime.utcnow()
        if self._last_trade_date is not None and self._last_trade_date.date() != now.date():
            self._daily_trade_count = 0

    def _update_heat_check(self) -> None:
        """Track P&L of last 5 trades. If net negative, switch to A+ only mode."""
        if len(self._recent_results) < 5:
            self._a_plus_only_mode = False
            return

        last_5 = list(self._recent_results)[-5:]
        net_pnl = sum(r["pnl"] for r in last_5)

        if net_pnl < 0 and not self._a_plus_only_mode:
            self._a_plus_only_mode = True
            logger.warning(
                "HEAT CHECK: Last 5 trades net P&L = $%.2f (negative). "
                "Switching to A+ only mode.",
                net_pnl,
            )
            self._log_drawdown_event("heat_check_activated", self.daily_drawdown)
        elif net_pnl >= 0 and self._a_plus_only_mode:
            self._a_plus_only_mode = False
            logger.info(
                "Heat check cleared: last 5 trades net P&L = $%.2f. "
                "Resuming normal signal acceptance.",
                net_pnl,
            )

    def _check_recovery_mode(self) -> None:
        """After >5% drawdown, activate recovery mode sizing for next 10 trades."""
        dd = self.total_drawdown
        if dd > self.RECOVERY_DRAWDOWN_THRESHOLD and self._recovery_trades_remaining <= 0:
            self._recovery_trades_remaining = self.RECOVERY_TRADE_COUNT
            logger.warning(
                "Recovery mode activated: %.2f%% drawdown exceeds %.2f%% threshold. "
                "Next %d trades at 50%% size.",
                dd * 100, self.RECOVERY_DRAWDOWN_THRESHOLD * 100,
                self.RECOVERY_TRADE_COUNT,
            )
            self._log_drawdown_event("recovery_mode_activated", dd)
        elif self._recovery_trades_remaining > 0:
            self._recovery_trades_remaining -= 1
            logger.info(
                "Recovery mode: %d trades remaining at reduced size",
                self._recovery_trades_remaining,
            )

    @property
    def is_heat_check_active(self) -> bool:
        """Whether the heat check has triggered A+ only mode."""
        return self._a_plus_only_mode

    @property
    def daily_trade_count(self) -> int:
        """Number of trades taken today."""
        self._refresh_daily_trade_count()
        return self._daily_trade_count

    def _log_drawdown_event(self, event_type: str, drawdown: float) -> None:
        """Log a drawdown event for later analysis."""
        entry = {
            "type": event_type,
            "drawdown": drawdown,
            "daily_dd": self.daily_drawdown,
            "weekly_dd": self.weekly_drawdown,
            "total_dd": self.total_drawdown,
            "portfolio_value": self._current_value,
            "peak_value": self._peak_value,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._drawdown_log.append(entry)
        logger.info("Drawdown event: %s", entry)

    @property
    def drawdown_history(self) -> List[Dict[str, Any]]:
        return list(self._drawdown_log)

    def get_status(self) -> Dict[str, Any]:
        """Get complete drawdown status including all anti-revenge trading state."""
        paused, reason = self.check_pause_rules()
        return {
            "daily_drawdown": self.daily_drawdown,
            "weekly_drawdown": self.weekly_drawdown,
            "total_drawdown": self.total_drawdown,
            "is_paused": paused,
            "pause_reason": reason,
            "in_recovery": self._in_recovery,
            "recovery_multiplier": self.get_recovery_multiplier(),
            "recovery_trades_remaining": self._recovery_trades_remaining,
            "recent_trade_count": len(self._recent_results),
            "daily_trade_count": self.daily_trade_count,
            "max_daily_trades": self.max_daily_trades,
            "heat_check_active": self._a_plus_only_mode,
        }

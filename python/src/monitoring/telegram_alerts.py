"""Telegram bot integration for trading alerts and notifications."""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AlertType(Enum):
    TRADE_EXECUTED = "trade_executed"
    STOP_LOSS_HIT = "stop_loss_hit"
    DAILY_SUMMARY = "daily_summary"
    MODEL_DEGRADATION = "model_degradation"
    SYSTEM_ERROR = "system_error"
    DRAWDOWN_WARNING = "drawdown_warning"


class TelegramAlerts:
    """Sends trading alerts via Telegram bot.

    Gracefully degrades if bot token is not configured.
    Implements rate limiting to avoid spam.
    """

    # Rate limiting: max messages per window
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_MAX = 20  # max messages per window

    def __init__(self, bot_token: str = "", chat_id: str = "",
                 alert_config: Optional[Dict[str, bool]] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._bot = None
        self._message_times: list[float] = []

        # Which alert types are enabled
        defaults = {
            "trade_executed": True,
            "stop_loss_hit": True,
            "daily_summary": True,
            "model_degradation": True,
            "drawdown_warning": True,
        }
        self._alert_config = {**defaults, **(alert_config or {})}

        if not self._enabled:
            logger.info("Telegram alerts DISABLED (no bot token/chat ID configured)")
        else:
            self._init_bot()

    def _init_bot(self) -> None:
        """Initialize the Telegram bot."""
        try:
            from telegram import Bot
            self._bot = Bot(token=self.bot_token)
            logger.info("Telegram bot initialized")
        except ImportError:
            logger.warning("python-telegram-bot not installed, Telegram alerts disabled")
            self._enabled = False
        except Exception as e:
            logger.error("Failed to initialize Telegram bot: %s", e)
            self._enabled = False

    def _is_rate_limited(self) -> bool:
        now = time.time()
        # Remove old timestamps
        self._message_times = [t for t in self._message_times
                               if now - t < self.RATE_LIMIT_WINDOW]
        return len(self._message_times) >= self.RATE_LIMIT_MAX

    def _record_message(self) -> None:
        self._message_times.append(time.time())

    def send_alert(self, message: str, alert_type: AlertType = AlertType.SYSTEM_ERROR) -> bool:
        """Send an alert message via Telegram.

        Returns True if sent successfully, False otherwise.
        """
        if not self._enabled:
            logger.debug("Telegram alert skipped (disabled): %s", alert_type.value)
            return False

        # Check if this alert type is enabled
        if not self._alert_config.get(alert_type.value, True):
            return False

        if self._is_rate_limited():
            logger.warning("Telegram rate limit reached, dropping alert")
            return False

        try:
            self._send_sync(message)
            self._record_message()
            return True
        except Exception as e:
            logger.error("Failed to send Telegram alert: %s", e)
            return False

    async def send_alert_async(self, message: str,
                               alert_type: AlertType = AlertType.SYSTEM_ERROR) -> bool:
        """Async version of send_alert."""
        if not self._enabled or not self._alert_config.get(alert_type.value, True):
            return False
        if self._is_rate_limited():
            return False
        try:
            await self._send_async(message)
            self._record_message()
            return True
        except Exception as e:
            logger.error("Failed to send async Telegram alert: %s", e)
            return False

    def _send_sync(self, message: str) -> None:
        """Send message synchronously."""
        if self._bot is None:
            return
        try:
            # python-telegram-bot v20+ is async-first; use a new event loop for sync
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="HTML",
            ))
            loop.close()
        except RuntimeError:
            # Already in an async context
            asyncio.create_task(self._send_async(message))

    async def _send_async(self, message: str) -> None:
        if self._bot is None:
            return
        await self._bot.send_message(
            chat_id=self.chat_id, text=message, parse_mode="HTML",
        )

    # --- Convenience methods for specific alert types ---

    def alert_trade_executed(self, asset: str, side: str, size: float,
                             price: float, exchange: str) -> bool:
        msg = (
            f"<b>Trade Executed</b>\n"
            f"Asset: {asset}\n"
            f"Side: {side.upper()}\n"
            f"Size: {size:.6f}\n"
            f"Price: ${price:,.4f}\n"
            f"Exchange: {exchange}"
        )
        return self.send_alert(msg, AlertType.TRADE_EXECUTED)

    def alert_stop_loss_hit(self, asset: str, entry_price: float,
                            stop_price: float, pnl: float) -> bool:
        emoji = "+" if pnl >= 0 else ""
        msg = (
            f"<b>Stop-Loss Hit</b>\n"
            f"Asset: {asset}\n"
            f"Entry: ${entry_price:,.4f}\n"
            f"Stop: ${stop_price:,.4f}\n"
            f"P&L: {emoji}${pnl:,.2f}"
        )
        return self.send_alert(msg, AlertType.STOP_LOSS_HIT)

    def alert_daily_summary(self, summary_text: str) -> bool:
        msg = f"<b>Daily P&L Summary</b>\n\n{summary_text}"
        return self.send_alert(msg, AlertType.DAILY_SUMMARY)

    def alert_model_degradation(self, model_name: str,
                                current_accuracy: float,
                                baseline_accuracy: float) -> bool:
        drop = baseline_accuracy - current_accuracy
        msg = (
            f"<b>Model Degradation Warning</b>\n"
            f"Model: {model_name}\n"
            f"Current accuracy: {current_accuracy:.1f}%\n"
            f"Baseline: {baseline_accuracy:.1f}%\n"
            f"Drop: {drop:.1f}%"
        )
        return self.send_alert(msg, AlertType.MODEL_DEGRADATION)

    def alert_system_error(self, subsystem: str, error: str) -> bool:
        msg = (
            f"<b>System Error</b>\n"
            f"Subsystem: {subsystem}\n"
            f"Error: {error}"
        )
        return self.send_alert(msg, AlertType.SYSTEM_ERROR)

    def alert_drawdown_warning(self, current_drawdown: float,
                               limit: float) -> bool:
        msg = (
            f"<b>Drawdown Warning</b>\n"
            f"Current drawdown: {current_drawdown:.2f}%\n"
            f"Limit: {limit:.2f}%\n"
            f"Action: Consider reducing exposure"
        )
        return self.send_alert(msg, AlertType.DRAWDOWN_WARNING)

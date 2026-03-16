"""Monitoring subsystem: trade logging, alerts, health checks, and reporting."""

import logging
from typing import Any, Dict, List, Optional

from src.utils.types import TradeResult
from src.monitoring.trade_logger import TradeLogger
from src.monitoring.system_health import SystemHealthMonitor
from src.monitoring.telegram_alerts import AlertType, TelegramAlerts
from src.monitoring.report_generator import ReportGenerator
from src.monitoring.model_tracker import ModelTracker

logger = logging.getLogger(__name__)


class MonitoringAgent:
    """Unified monitoring interface that coordinates all monitoring subsystems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        data_config = config.get("data", {})
        mon_config = config.get("monitoring", {})
        tg_config = mon_config.get("telegram", {})
        alert_config = mon_config.get("alerts", {})

        db_path = data_config.get("sqlite_path", "data/trading.db")

        self.trade_logger = TradeLogger(db_path=db_path)
        self.health_monitor = SystemHealthMonitor()
        self.telegram = TelegramAlerts(
            bot_token=tg_config.get("bot_token", ""),
            chat_id=tg_config.get("chat_id", ""),
            alert_config=alert_config,
        )
        self.report_generator = ReportGenerator(self.trade_logger)
        self.model_tracker = ModelTracker()

        logger.info("MonitoringAgent initialized")

    def log_trade(self, trade_result: TradeResult) -> int:
        """Log a trade result and send alert if configured."""
        row_id = self.trade_logger.log_trade(trade_result)

        # Send Telegram alert for executed trades
        if trade_result.status.value == "filled":
            proposal = trade_result.proposal
            side = "buy" if proposal.direction.value in ("BUY", "STRONG_BUY") else "sell"
            self.telegram.alert_trade_executed(
                asset=proposal.asset,
                side=side,
                size=trade_result.fill_size,
                price=trade_result.fill_price,
                exchange=trade_result.exchange,
            )

        return row_id

    def send_alert(self, message: str, alert_type: str = "system_error") -> bool:
        """Send an alert via Telegram."""
        try:
            at = AlertType(alert_type)
        except ValueError:
            at = AlertType.SYSTEM_ERROR
        return self.telegram.send_alert(message, at)

    def get_performance_report(self, period: str = "daily") -> str:
        """Generate and return a formatted performance report."""
        report = self.report_generator.generate_report(period)
        text = self.report_generator.format_text(report)

        # Optionally send to Telegram
        if period == "daily":
            self.telegram.alert_daily_summary(text)

        return text

    def check_system_health(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        health = self.health_monitor.get_all_health()

        # Alert if any subsystem is critical
        for name, status in health.get("subsystems", {}).items():
            if status.get("status") == "critical":
                self.telegram.alert_system_error(name, status.get("message", ""))

        return health

    def track_model_performance(self, model_name: str,
                                predictions: List[float],
                                actuals: List[float]) -> None:
        """Track prediction accuracy for a model and check for degradation."""
        metrics = self.model_tracker.track(model_name, predictions, actuals)

        # Check for degradation
        degraded = self.model_tracker.check_degradation()
        for d in degraded:
            if d["model"] == model_name:
                self.telegram.alert_model_degradation(
                    model_name=model_name,
                    current_accuracy=d["current_accuracy"],
                    baseline_accuracy=d["baseline_accuracy"],
                )


__all__ = ["MonitoringAgent"]

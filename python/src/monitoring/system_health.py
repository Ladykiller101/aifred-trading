"""System health monitoring: connectivity, latency, data freshness, error tracking."""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus:
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SubsystemHealth:
    """Health state for a single subsystem."""

    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.UNKNOWN
        self.latency_ms: float = 0.0
        self.last_check: Optional[datetime] = None
        self.last_success: Optional[datetime] = None
        self.error_count: int = 0
        self.message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "error_count": self.error_count,
            "message": self.message,
        }


class SystemHealthMonitor:
    """Monitors exchange APIs, data pipelines, and subsystem health."""

    STALE_DATA_THRESHOLD_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._subsystems: Dict[str, SubsystemHealth] = {}
        self._error_log: List[Dict[str, Any]] = []
        self._data_timestamps: Dict[str, datetime] = {}  # source -> last data time

    def register_subsystem(self, name: str) -> None:
        """Register a subsystem for health tracking."""
        self._subsystems[name] = SubsystemHealth(name)

    def check_exchange(self, name: str, connector) -> SubsystemHealth:
        """Check connectivity and latency for an exchange connector."""
        health = self._subsystems.get(name)
        if health is None:
            self.register_subsystem(name)
            health = self._subsystems[name]

        health.last_check = datetime.utcnow()
        try:
            latency = connector.ping()
            if latency < 0:
                health.status = HealthStatus.CRITICAL
                health.message = "Ping failed"
                health.error_count += 1
            elif latency > 5000:
                health.status = HealthStatus.WARNING
                health.message = f"High latency: {latency:.0f}ms"
                health.latency_ms = latency
                health.last_success = datetime.utcnow()
            else:
                health.status = HealthStatus.OK
                health.latency_ms = latency
                health.message = ""
                health.last_success = datetime.utcnow()
                health.error_count = 0
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.message = str(e)
            health.error_count += 1
            self._log_error(name, str(e))

        return health

    def record_data_timestamp(self, source: str, timestamp: Optional[datetime] = None) -> None:
        """Record when data was last received from a source."""
        self._data_timestamps[source] = timestamp or datetime.utcnow()

    def check_data_freshness(self, source: str) -> SubsystemHealth:
        """Check if data from a source is stale."""
        health = self._subsystems.get(f"data:{source}")
        if health is None:
            self.register_subsystem(f"data:{source}")
            health = self._subsystems[f"data:{source}"]

        health.last_check = datetime.utcnow()
        last_data = self._data_timestamps.get(source)

        if last_data is None:
            health.status = HealthStatus.UNKNOWN
            health.message = "No data received yet"
            return health

        age_seconds = (datetime.utcnow() - last_data).total_seconds()
        if age_seconds > self.STALE_DATA_THRESHOLD_SECONDS:
            health.status = HealthStatus.WARNING
            health.message = f"Data stale: {age_seconds:.0f}s old (threshold: {self.STALE_DATA_THRESHOLD_SECONDS}s)"
        else:
            health.status = HealthStatus.OK
            health.message = f"Data fresh: {age_seconds:.0f}s old"
            health.last_success = datetime.utcnow()

        return health

    def record_error(self, subsystem: str, error: str) -> None:
        """Record an error for tracking error rates."""
        health = self._subsystems.get(subsystem)
        if health:
            health.error_count += 1
            health.status = HealthStatus.WARNING
            health.message = error
        self._log_error(subsystem, error)

    def record_success(self, subsystem: str) -> None:
        """Record a successful operation."""
        health = self._subsystems.get(subsystem)
        if health:
            health.last_success = datetime.utcnow()
            health.status = HealthStatus.OK
            health.message = ""

    def get_health(self, subsystem: str) -> Optional[Dict[str, Any]]:
        health = self._subsystems.get(subsystem)
        return health.to_dict() if health else None

    def get_all_health(self) -> Dict[str, Any]:
        """Get health status for all subsystems."""
        statuses = {name: h.to_dict() for name, h in self._subsystems.items()}
        overall = HealthStatus.OK
        for h in self._subsystems.values():
            if h.status == HealthStatus.CRITICAL:
                overall = HealthStatus.CRITICAL
                break
            if h.status == HealthStatus.WARNING and overall != HealthStatus.CRITICAL:
                overall = HealthStatus.WARNING
        return {
            "overall": overall,
            "subsystems": statuses,
            "checked_at": datetime.utcnow().isoformat(),
        }

    def get_error_rates(self, window_minutes: int = 60) -> Dict[str, int]:
        """Get error counts per subsystem in the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        rates: Dict[str, int] = defaultdict(int)
        for entry in self._error_log:
            if entry["timestamp"] >= cutoff:
                rates[entry["subsystem"]] += 1
        return dict(rates)

    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._error_log[-limit:]

    def _log_error(self, subsystem: str, error: str) -> None:
        self._error_log.append({
            "subsystem": subsystem,
            "error": error,
            "timestamp": datetime.utcnow(),
        })
        # Keep last 1000 errors
        if len(self._error_log) > 1000:
            self._error_log = self._error_log[-1000:]
        logger.warning("Health error [%s]: %s", subsystem, error)

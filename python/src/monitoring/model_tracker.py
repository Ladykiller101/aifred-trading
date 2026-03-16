"""Track per-model prediction accuracy, detect degradation, and log experiments."""

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelMetrics:
    """Rolling metrics for a single model."""

    def __init__(self, name: str, window_size: int = 200):
        self.name = name
        self.window_size = window_size
        self._predictions: List[Tuple[float, float, datetime]] = []  # (predicted, actual, time)
        self.baseline_accuracy: Optional[float] = None
        self.profitable_contribution: float = 0.0

    def record(self, predicted: float, actual: float,
               timestamp: Optional[datetime] = None) -> None:
        ts = timestamp or datetime.utcnow()
        self._predictions.append((predicted, actual, ts))
        if len(self._predictions) > self.window_size:
            self._predictions = self._predictions[-self.window_size:]

    @property
    def accuracy(self) -> float:
        """Directional accuracy: how often prediction direction matches actual."""
        if len(self._predictions) < 5:
            return 0.0
        correct = sum(
            1 for pred, actual, _ in self._predictions
            if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) or (pred == 0 and actual == 0)
        )
        return correct / len(self._predictions) * 100

    @property
    def precision(self) -> float:
        """Precision: of trades we predicted positive, how many were actually positive."""
        positive_preds = [(p, a) for p, a, _ in self._predictions if p > 0]
        if not positive_preds:
            return 0.0
        true_positives = sum(1 for _, a in positive_preds if a > 0)
        return true_positives / len(positive_preds) * 100

    @property
    def recall(self) -> float:
        """Recall: of actual positives, how many did we predict."""
        actual_positives = [(p, a) for p, a, _ in self._predictions if a > 0]
        if not actual_positives:
            return 0.0
        true_positives = sum(1 for p, _ in actual_positives if p > 0)
        return true_positives / len(actual_positives) * 100

    @property
    def mse(self) -> float:
        """Mean squared error."""
        if not self._predictions:
            return 0.0
        return sum((p - a) ** 2 for p, a, _ in self._predictions) / len(self._predictions)

    @property
    def sample_count(self) -> int:
        return len(self._predictions)

    def set_baseline(self, accuracy: Optional[float] = None) -> None:
        """Set baseline accuracy. If None, uses current accuracy."""
        self.baseline_accuracy = accuracy if accuracy is not None else self.accuracy

    @property
    def is_degraded(self) -> bool:
        """Check if accuracy has dropped >10% from baseline."""
        if self.baseline_accuracy is None or self.sample_count < 20:
            return False
        return (self.baseline_accuracy - self.accuracy) > 10.0

    @property
    def degradation_amount(self) -> float:
        if self.baseline_accuracy is None:
            return 0.0
        return self.baseline_accuracy - self.accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "mse": self.mse,
            "sample_count": self.sample_count,
            "baseline_accuracy": self.baseline_accuracy,
            "is_degraded": self.is_degraded,
            "degradation": self.degradation_amount,
            "profitable_contribution": self.profitable_contribution,
        }


class ModelTracker:
    """Tracks and compares performance across multiple prediction models."""

    DEGRADATION_THRESHOLD = 10.0  # percentage points

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._models: Dict[str, ModelMetrics] = {}
        self._profit_attributions: Dict[str, List[float]] = defaultdict(list)

    def register_model(self, name: str, baseline_accuracy: Optional[float] = None) -> None:
        """Register a model for tracking."""
        metrics = ModelMetrics(name, self.window_size)
        if baseline_accuracy is not None:
            metrics.set_baseline(baseline_accuracy)
        self._models[name] = metrics
        logger.info("Registered model: %s (baseline=%.1f%%)",
                     name, baseline_accuracy or 0)

    def track(self, model_name: str, predictions: List[float],
              actuals: List[float]) -> ModelMetrics:
        """Record prediction-actual pairs for a model.

        Args:
            model_name: Name of the model
            predictions: List of predicted values (positive = up, negative = down)
            actuals: List of actual values
        """
        if model_name not in self._models:
            self.register_model(model_name)
        metrics = self._models[model_name]

        for pred, actual in zip(predictions, actuals):
            metrics.record(pred, actual)

        # Auto-set baseline after first batch if not set
        if metrics.baseline_accuracy is None and metrics.sample_count >= 50:
            metrics.set_baseline()
            logger.info("Auto-set baseline for %s: %.1f%%",
                        model_name, metrics.baseline_accuracy)

        return metrics

    def record_profit_attribution(self, model_name: str, pnl: float) -> None:
        """Attribute a trade's P&L to a model."""
        self._profit_attributions[model_name].append(pnl)
        if len(self._profit_attributions[model_name]) > 1000:
            self._profit_attributions[model_name] = self._profit_attributions[model_name][-1000:]
        if model_name in self._models:
            self._models[model_name].profitable_contribution = sum(
                self._profit_attributions[model_name]
            )

    def check_degradation(self) -> List[Dict[str, Any]]:
        """Check all models for degradation. Returns list of degraded models."""
        degraded = []
        for name, metrics in self._models.items():
            if metrics.is_degraded:
                degraded.append({
                    "model": name,
                    "current_accuracy": metrics.accuracy,
                    "baseline_accuracy": metrics.baseline_accuracy,
                    "drop": metrics.degradation_amount,
                    "sample_count": metrics.sample_count,
                })
                logger.warning("Model degradation detected: %s (%.1f%% -> %.1f%%)",
                               name, metrics.baseline_accuracy, metrics.accuracy)
        return degraded

    def get_model_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        metrics = self._models.get(model_name)
        return metrics.to_dict() if metrics else None

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {name: m.to_dict() for name, m in self._models.items()}

    def get_best_model(self) -> Optional[str]:
        """Return the model with highest accuracy."""
        if not self._models:
            return None
        return max(self._models, key=lambda n: self._models[n].accuracy)

    def get_most_profitable_model(self) -> Optional[str]:
        """Return the model that contributed most to profits."""
        if not self._profit_attributions:
            return None
        return max(self._profit_attributions,
                   key=lambda n: sum(self._profit_attributions[n]))

    def log_to_mlflow(self, run_name: Optional[str] = None) -> bool:
        """Log current metrics to MLflow for experiment tracking."""
        try:
            import mlflow
        except ImportError:
            logger.debug("MLflow not installed, skipping experiment logging")
            return False

        try:
            with mlflow.start_run(run_name=run_name or "model_tracker"):
                for name, metrics in self._models.items():
                    mlflow.log_metrics({
                        f"{name}_accuracy": metrics.accuracy,
                        f"{name}_precision": metrics.precision,
                        f"{name}_recall": metrics.recall,
                        f"{name}_mse": metrics.mse,
                        f"{name}_samples": metrics.sample_count,
                    })
                    if metrics.baseline_accuracy is not None:
                        mlflow.log_metric(f"{name}_baseline", metrics.baseline_accuracy)
            return True
        except Exception as e:
            logger.error("Failed to log to MLflow: %s", e)
            return False

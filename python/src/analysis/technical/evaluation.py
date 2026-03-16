"""Backtesting and evaluation framework for technical analysis models.

Provides performance metrics, comparison reports, and backtesting
simulation for individual models and the ensemble.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.types import Signal, Direction

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Performance metrics for a single model."""
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    n_samples: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ModelEvaluator:
    """Evaluates model predictions against actual outcomes."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
        """
        self.risk_free_rate = risk_free_rate

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> ModelMetrics:
        """Compute classification metrics.

        Args:
            y_true: True labels (0/1).
            y_pred: Predicted labels (0/1).
            y_proba: Optional predicted probabilities (n, 2).
            model_name: Model identifier.

        Returns:
            ModelMetrics with classification performance.
        """
        valid = (y_true >= 0) & (y_pred >= 0)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        n = len(y_true)
        if n == 0:
            return ModelMetrics(model_name=model_name, n_samples=0)

        accuracy = float(np.mean(y_true == y_pred))

        # Precision, recall, F1 for class 1 (up)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-8))

        return ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            n_samples=n,
        )

    def evaluate_trading_performance(
        self,
        signals: List[Signal],
        prices: pd.Series,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.02,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        model_name: str = "model",
    ) -> ModelMetrics:
        """Simulate trading and compute financial metrics.

        Args:
            signals: List of trading signals in chronological order.
            prices: Close prices aligned with signals.
            initial_capital: Starting capital.
            position_size_pct: Fraction of capital per trade.
            stop_loss_pct: Stop-loss percentage.
            take_profit_pct: Take-profit percentage.
            model_name: Model identifier.

        Returns:
            ModelMetrics with trading performance.
        """
        if len(signals) == 0 or len(prices) == 0:
            return ModelMetrics(model_name=model_name)

        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0.0
        trades = []
        equity_curve = [initial_capital]

        position = None  # {"entry_price": float, "direction": 1/-1, "size": float}

        price_arr = prices.values if isinstance(prices, pd.Series) else prices

        for i, signal in enumerate(signals):
            if i >= len(price_arr):
                break

            current_price = price_arr[i]

            # Check existing position for exit
            if position is not None:
                pnl_pct = (current_price / position["entry_price"] - 1) * position["direction"]

                hit_sl = pnl_pct <= -stop_loss_pct
                hit_tp = pnl_pct >= take_profit_pct

                if hit_sl or hit_tp:
                    realized_pnl = position["size"] * pnl_pct
                    capital += realized_pnl
                    trades.append({
                        "entry": position["entry_price"],
                        "exit": current_price,
                        "direction": position["direction"],
                        "pnl": realized_pnl,
                        "pnl_pct": pnl_pct,
                        "reason": "sl" if hit_sl else "tp",
                    })
                    position = None

            # Open new position on signal
            if position is None and signal.direction in (
                Direction.BUY, Direction.STRONG_BUY,
                Direction.SELL, Direction.STRONG_SELL,
            ):
                direction = 1 if signal.direction in (Direction.BUY, Direction.STRONG_BUY) else -1
                size = capital * position_size_pct
                position = {
                    "entry_price": current_price,
                    "direction": direction,
                    "size": size,
                }

            equity_curve.append(capital)
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Close any remaining position at last price
        if position is not None and len(price_arr) > 0:
            last_price = price_arr[-1]
            pnl_pct = (last_price / position["entry_price"] - 1) * position["direction"]
            realized_pnl = position["size"] * pnl_pct
            capital += realized_pnl
            trades.append({
                "entry": position["entry_price"],
                "exit": last_price,
                "direction": position["direction"],
                "pnl": realized_pnl,
                "pnl_pct": pnl_pct,
                "reason": "end",
            })

        # Compute metrics
        n_trades = len(trades)
        if n_trades == 0:
            return ModelMetrics(model_name=model_name, n_trades=0)

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        win_rate = len(wins) / n_trades
        total_return = (capital - initial_capital) / initial_capital

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1e-8
        profit_factor = gross_profit / max(gross_loss, 1e-8)

        # Sharpe ratio from equity curve
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) > 1 and np.std(returns) > 0:
            # Annualize assuming hourly data
            periods_per_year = 365 * 24
            sharpe = (np.mean(returns) - self.risk_free_rate / periods_per_year) / np.std(returns)
            sharpe *= np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        return ModelMetrics(
            model_name=model_name,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            n_trades=n_trades,
            n_samples=len(signals),
        )


class BacktestEngine:
    """Backtesting framework that runs the full analysis pipeline on historical data."""

    def __init__(self, evaluator: Optional[ModelEvaluator] = None):
        self.evaluator = evaluator or ModelEvaluator()
        self.results: Dict[str, ModelMetrics] = {}

    def backtest_model(
        self,
        model_name: str,
        predictions: np.ndarray,
        y_true: np.ndarray,
        prices: pd.Series,
        probabilities: Optional[np.ndarray] = None,
    ) -> ModelMetrics:
        """Backtest a single model's predictions.

        Args:
            model_name: Model identifier.
            predictions: Predicted labels.
            y_true: True labels.
            prices: Close prices.
            probabilities: Optional prediction probabilities.

        Returns:
            ModelMetrics combining classification and trading performance.
        """
        # Classification metrics
        clf_metrics = self.evaluator.evaluate_predictions(
            y_true, predictions, probabilities, model_name
        )

        # Convert predictions to signals for trading evaluation
        signals = []
        for i, pred in enumerate(predictions):
            if pred == 1:
                direction = Direction.BUY
            elif pred == 0:
                direction = Direction.SELL
            else:
                direction = Direction.HOLD

            conf = 50.0
            if probabilities is not None and i < len(probabilities):
                conf = float(abs(probabilities[i, 1] - 0.5) * 200) if probabilities.ndim == 2 else 50.0

            signals.append(Signal(
                asset="backtest",
                direction=direction,
                confidence=conf,
                source=model_name,
            ))

        trade_metrics = self.evaluator.evaluate_trading_performance(
            signals, prices, model_name=model_name
        )

        # Merge classification and trading metrics
        combined = ModelMetrics(
            model_name=model_name,
            accuracy=clf_metrics.accuracy,
            precision=clf_metrics.precision,
            recall=clf_metrics.recall,
            f1=clf_metrics.f1,
            sharpe_ratio=trade_metrics.sharpe_ratio,
            max_drawdown=trade_metrics.max_drawdown,
            total_return=trade_metrics.total_return,
            win_rate=trade_metrics.win_rate,
            profit_factor=trade_metrics.profit_factor,
            n_trades=trade_metrics.n_trades,
            n_samples=clf_metrics.n_samples,
        )

        self.results[model_name] = combined
        return combined

    def comparison_report(self) -> pd.DataFrame:
        """Generate a comparison table across all backtested models.

        Returns:
            DataFrame with one row per model and metrics as columns.
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for name, metrics in self.results.items():
            rows.append({
                "model": name,
                "accuracy": round(metrics.accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "sharpe": round(metrics.sharpe_ratio, 3),
                "max_drawdown": round(metrics.max_drawdown, 4),
                "total_return": round(metrics.total_return, 4),
                "win_rate": round(metrics.win_rate, 4),
                "profit_factor": round(metrics.profit_factor, 3),
                "n_trades": metrics.n_trades,
                "n_samples": metrics.n_samples,
            })

        df = pd.DataFrame(rows).set_index("model")
        return df.sort_values("sharpe", ascending=False)

    def summary(self) -> Dict[str, Any]:
        """Generate a summary dict of the best-performing model."""
        if not self.results:
            return {"status": "no_results"}

        report = self.comparison_report()
        best_model = report.index[0]

        return {
            "best_model": best_model,
            "best_sharpe": float(report.loc[best_model, "sharpe"]),
            "best_accuracy": float(report.loc[best_model, "accuracy"]),
            "best_win_rate": float(report.loc[best_model, "win_rate"]),
            "n_models_evaluated": len(self.results),
            "report": report.to_dict(),
        }

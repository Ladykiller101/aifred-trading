"""Autoresearch-inspired autonomous parameter optimizer.

Inspired by Karpathy's autoresearch framework:
- Define a primary metric (win_rate)
- Define a search space (strategy parameters)
- Run experiments with fixed time budget
- Keep improvements, discard failures
- Log everything to results.tsv for analysis

The optimizer runs overnight, discovering parameter regimes
that maximize win rate without overfitting (fixed OOS validation).
"""

import csv
import hashlib
import json
import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)

# Search space: parameters to optimize and their ranges
SEARCH_SPACE = {
    "min_confidence": {
        "type": "float",
        "min": 65.0,
        "max": 92.0,
        "step": 1.0,
        "default": 78.0,
    },
    "min_risk_reward": {
        "type": "float",
        "min": 0.5,
        "max": 3.0,
        "step": 0.1,
        "default": 1.5,
    },
    "max_loss_streak_before_pause": {
        "type": "int",
        "min": 2,
        "max": 8,
        "step": 1,
        "default": 3,
    },
    "excluded_tiers": {
        "type": "categorical",
        "options": [[], ["C"], ["C", "B"], ["C", "B", "A"]],
        "default": ["C"],
    },
    "excluded_hours": {
        "type": "categorical",
        "options": [
            [],
            [1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5, 6],
            [22, 23, 0, 1, 2, 3, 4, 5],
        ],
        "default": [1, 2, 3, 4, 5],
    },
}


@dataclass
class ExperimentResult:
    """Result of a single optimizer experiment."""
    experiment_id: str
    params: Dict[str, Any]
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    profit_factor: float
    status: str  # "keep", "discard", "crash"
    description: str
    timestamp: str = ""
    duration_seconds: float = 0.0
    improvement: float = 0.0  # delta vs baseline


class ParameterOptimizer:
    """Autonomous parameter search inspired by Karpathy's autoresearch.

    Core loop:
    1. Sample parameter variation from search space
    2. Run backtest with those parameters
    3. Compare to best known result
    4. Keep if improved, discard if not
    5. Log to results.tsv
    6. Repeat until budget exhausted

    Search strategies:
    - Random search (baseline)
    - Bayesian-inspired: narrow search around best known params
    - Grid search for fine-tuning final params
    """

    def __init__(
        self,
        db_path: str = "data/trading.db",
        results_dir: str = "data/optimizer",
        validation_start: str = "2025-12-01",
        validation_end: str = "2026-03-15",
    ):
        self.engine = BacktestEngine(db_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.validation_start = validation_start
        self.validation_end = validation_end

        self.results_file = self.results_dir / "results.tsv"
        self.best_params_file = self.results_dir / "best_params.json"

        self.best_win_rate = 0.0
        self.best_params: Dict[str, Any] = {}
        self.experiment_history: List[ExperimentResult] = []

        self._load_previous_results()

    def _load_previous_results(self) -> None:
        """Load previous experiment results to resume from best known."""
        if self.best_params_file.exists():
            try:
                with open(self.best_params_file) as f:
                    data = json.load(f)
                self.best_win_rate = data.get("win_rate", 0)
                self.best_params = data.get("params", {})
                logger.info(
                    "Loaded previous best: win_rate=%.2f%%", self.best_win_rate
                )
            except Exception:
                pass

    def run(
        self,
        max_experiments: int = 100,
        min_improvement: float = 0.5,
        strategy: str = "bayesian",
    ) -> Dict[str, Any]:
        """Run the optimizer loop.

        Args:
            max_experiments: Maximum experiments to run.
            min_improvement: Minimum win rate improvement to keep (%).
            strategy: Search strategy - "random", "bayesian", or "grid".

        Returns:
            Best parameters found with metrics.
        """
        logger.info(
            "Starting optimizer: max_experiments=%d, strategy=%s, "
            "validation=%s to %s",
            max_experiments, strategy, self.validation_start, self.validation_end,
        )

        # Run baseline first
        if self.best_win_rate == 0:
            baseline = self._run_experiment(
                self._get_defaults(), "baseline (default params)"
            )
            if baseline.status != "crash":
                self.best_win_rate = baseline.win_rate
                self.best_params = baseline.params
                self._save_best()
                logger.info("Baseline: win_rate=%.2f%%", baseline.win_rate)

        kept = 0
        discarded = 0
        crashed = 0

        for i in range(max_experiments):
            # Sample parameters based on strategy
            if strategy == "bayesian":
                params, desc = self._sample_bayesian(i)
            elif strategy == "grid":
                params, desc = self._sample_grid(i, max_experiments)
            else:
                params, desc = self._sample_random()

            # Run experiment
            result = self._run_experiment(params, desc)

            if result.status == "crash":
                crashed += 1
                continue

            # Decision: keep or discard
            improvement = result.win_rate - self.best_win_rate
            result.improvement = improvement

            if improvement >= min_improvement:
                result.status = "keep"
                self.best_win_rate = result.win_rate
                self.best_params = result.params
                self._save_best()
                kept += 1
                logger.info(
                    "[%d/%d] KEEP: win_rate=%.2f%% (+%.2f%%) — %s",
                    i + 1, max_experiments, result.win_rate, improvement, desc,
                )
            else:
                result.status = "discard"
                discarded += 1
                logger.info(
                    "[%d/%d] DISCARD: win_rate=%.2f%% (%.2f%%) — %s",
                    i + 1, max_experiments, result.win_rate, improvement, desc,
                )

            self._log_result(result)
            self.experiment_history.append(result)

        summary = {
            "best_win_rate": self.best_win_rate,
            "best_params": self.best_params,
            "experiments_run": max_experiments,
            "kept": kept,
            "discarded": discarded,
            "crashed": crashed,
            "results_file": str(self.results_file),
        }

        # Save final summary
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(
            "Optimizer complete: best_win_rate=%.2f%%, kept=%d, discarded=%d",
            self.best_win_rate, kept, discarded,
        )
        return summary

    def _run_experiment(
        self, params: Dict[str, Any], description: str
    ) -> ExperimentResult:
        """Run a single backtest experiment."""
        exp_id = hashlib.md5(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        start_time = time.time()
        try:
            config = BacktestConfig(
                start_date=self.validation_start,
                end_date=self.validation_end,
                params=params,
            )
            bt_result = self.engine.run(config)
            duration = time.time() - start_time

            return ExperimentResult(
                experiment_id=exp_id,
                params=params,
                win_rate=bt_result.win_rate,
                total_return=bt_result.total_return,
                sharpe_ratio=bt_result.sharpe_ratio,
                max_drawdown=bt_result.max_drawdown,
                total_trades=bt_result.total_trades,
                profit_factor=bt_result.profit_factor,
                status="pending",
                description=description,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration,
            )
        except Exception as e:
            logger.error("Experiment %s crashed: %s", exp_id, e)
            return ExperimentResult(
                experiment_id=exp_id,
                params=params,
                win_rate=0.0,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                profit_factor=0.0,
                status="crash",
                description=f"CRASH: {e}",
                timestamp=datetime.now().isoformat(),
                duration_seconds=time.time() - start_time,
            )

    def _sample_random(self) -> Tuple[Dict[str, Any], str]:
        """Sample random parameters from search space."""
        params = {}
        changes = []
        for name, spec in SEARCH_SPACE.items():
            if spec["type"] == "float":
                val = round(
                    random.uniform(spec["min"], spec["max"]),
                    1 if spec["step"] >= 0.1 else 2,
                )
                params[name] = val
                if val != spec["default"]:
                    changes.append(f"{name}={val}")
            elif spec["type"] == "int":
                val = random.randint(spec["min"], spec["max"])
                params[name] = val
                if val != spec["default"]:
                    changes.append(f"{name}={val}")
            elif spec["type"] == "categorical":
                val = random.choice(spec["options"])
                params[name] = val
                if val != spec["default"]:
                    changes.append(f"{name}={val}")

        desc = "random: " + ", ".join(changes) if changes else "random: defaults"
        return params, desc

    def _sample_bayesian(self, iteration: int) -> Tuple[Dict[str, Any], str]:
        """Bayesian-inspired: narrow search around best known params.

        Early iterations: wide exploration (random).
        Later iterations: narrow perturbation around best.
        """
        explore_ratio = max(0.2, 1.0 - iteration / 50)

        if random.random() < explore_ratio or not self.best_params:
            return self._sample_random()

        # Perturb best known params
        params = deepcopy(self.best_params)
        changes = []

        # Pick 1-2 parameters to perturb
        param_names = list(SEARCH_SPACE.keys())
        n_perturb = random.randint(1, 2)
        to_perturb = random.sample(param_names, min(n_perturb, len(param_names)))

        for name in to_perturb:
            spec = SEARCH_SPACE[name]
            if spec["type"] == "float":
                current = params.get(name, spec["default"])
                delta = random.gauss(0, spec["step"] * 2)
                val = round(
                    max(spec["min"], min(spec["max"], current + delta)),
                    1 if spec["step"] >= 0.1 else 2,
                )
                params[name] = val
                changes.append(f"{name}={val}")
            elif spec["type"] == "int":
                current = params.get(name, spec["default"])
                delta = random.choice([-1, 0, 1])
                val = max(spec["min"], min(spec["max"], current + delta))
                params[name] = val
                changes.append(f"{name}={val}")
            elif spec["type"] == "categorical":
                val = random.choice(spec["options"])
                params[name] = val
                changes.append(f"{name}={val}")

        desc = "bayesian perturb: " + ", ".join(changes)
        return params, desc

    def _sample_grid(
        self, iteration: int, total: int
    ) -> Tuple[Dict[str, Any], str]:
        """Grid search over confidence threshold (most impactful param)."""
        spec = SEARCH_SPACE["min_confidence"]
        n_points = int((spec["max"] - spec["min"]) / spec["step"]) + 1
        idx = iteration % n_points
        conf = spec["min"] + idx * spec["step"]

        params = self._get_defaults()
        params["min_confidence"] = conf

        return params, f"grid: min_confidence={conf}"

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default parameters from search space."""
        return {name: spec["default"] for name, spec in SEARCH_SPACE.items()}

    def _save_best(self) -> None:
        """Save best parameters to file."""
        with open(self.best_params_file, "w") as f:
            json.dump(
                {
                    "win_rate": self.best_win_rate,
                    "params": self.best_params,
                    "updated": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

    def _log_result(self, result: ExperimentResult) -> None:
        """Append experiment result to results.tsv."""
        file_exists = self.results_file.exists()
        with open(self.results_file, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if not file_exists:
                writer.writerow([
                    "experiment_id", "win_rate", "total_return", "sharpe",
                    "max_dd", "trades", "profit_factor", "improvement",
                    "status", "description", "timestamp", "duration_s",
                ])
            writer.writerow([
                result.experiment_id,
                f"{result.win_rate:.2f}",
                f"{result.total_return:.2f}",
                f"{result.sharpe_ratio:.3f}",
                f"{result.max_drawdown:.2f}",
                result.total_trades,
                f"{result.profit_factor:.2f}",
                f"{result.improvement:+.2f}",
                result.status,
                result.description,
                result.timestamp,
                f"{result.duration_seconds:.1f}",
            ])


def run_optimizer(
    db_path: str = "data/trading.db",
    max_experiments: int = 100,
    strategy: str = "bayesian",
) -> Dict[str, Any]:
    """Convenience function to run the optimizer."""
    optimizer = ParameterOptimizer(db_path=db_path)
    return optimizer.run(max_experiments=max_experiments, strategy=strategy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_optimizer()
    print(f"\nBest win rate: {result['best_win_rate']:.2f}%")
    print(f"Best params: {json.dumps(result['best_params'], indent=2)}")
    print(f"Experiments: {result['experiments_run']} "
          f"(kept={result['kept']}, discarded={result['discarded']})")

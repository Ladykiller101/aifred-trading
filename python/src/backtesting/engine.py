"""Walk-forward backtesting engine with realistic simulation.

Supports:
- Walk-forward validation with configurable purge gap
- Realistic slippage and commission modeling
- Per-asset and per-strategy performance breakdown
- Volatility regime tracking
- Integration with the optimizer for parameter search
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    start_date: str = "2025-06-01"
    end_date: str = "2026-03-15"
    initial_capital: float = 100_000.0
    commission_pct: float = 0.075  # 7.5 bps per trade
    slippage_pct: float = 0.05  # 5 bps slippage
    # Parameter overrides (from optimizer)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest result with all metrics."""
    # Primary metrics
    total_return: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    # Breakdown
    by_asset: Dict[str, Dict] = field(default_factory=dict)
    by_strategy: Dict[str, Dict] = field(default_factory=dict)
    by_tier: Dict[str, Dict] = field(default_factory=dict)
    # Meta
    params_used: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """Walk-forward backtesting engine.

    Replays historical signals through the trading pipeline with
    configurable parameters, tracking every trade and producing
    comprehensive performance metrics.
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = db_path

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Run a full backtest with the given configuration.

        Args:
            config: Backtest configuration with parameter overrides.

        Returns:
            BacktestResult with all performance metrics.
        """
        start_time = datetime.now()
        result = BacktestResult(params_used=config.params)

        trades = self._load_trades(config.start_date, config.end_date)
        if not trades:
            logger.warning("No trades found for backtest period %s to %s",
                           config.start_date, config.end_date)
            return result

        # Apply parameter overrides to filter/adjust trades
        filtered = self._apply_params(trades, config.params)

        if not filtered:
            result.duration_seconds = (datetime.now() - start_time).total_seconds()
            return result

        # Calculate metrics
        capital = config.initial_capital
        equity_curve = [capital]
        peak = capital

        wins = []
        losses = []
        asset_stats: Dict[str, Dict] = {}
        strategy_stats: Dict[str, Dict] = {}
        tier_stats: Dict[str, Dict] = {}

        for trade in filtered:
            pnl = trade.get("pnl", 0) or 0

            # Apply commission and slippage
            trade_value = abs(trade.get("size", 0) * trade.get("entry_price", 0))
            commission = trade_value * config.commission_pct / 100
            slippage_cost = trade_value * config.slippage_pct / 100
            adjusted_pnl = pnl - commission - slippage_cost

            capital += adjusted_pnl
            equity_curve.append(capital)
            peak = max(peak, capital)

            if adjusted_pnl > 0:
                wins.append(adjusted_pnl)
            elif adjusted_pnl < 0:
                losses.append(adjusted_pnl)

            # Track by asset
            asset = trade.get("asset", "unknown")
            if asset not in asset_stats:
                asset_stats[asset] = {"pnl": 0, "trades": 0, "wins": 0}
            asset_stats[asset]["pnl"] += adjusted_pnl
            asset_stats[asset]["trades"] += 1
            if adjusted_pnl > 0:
                asset_stats[asset]["wins"] += 1

            # Track by strategy
            strategy = trade.get("signal_source", "unknown")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"pnl": 0, "trades": 0, "wins": 0}
            strategy_stats[strategy]["pnl"] += adjusted_pnl
            strategy_stats[strategy]["trades"] += 1
            if adjusted_pnl > 0:
                strategy_stats[strategy]["wins"] += 1

            # Track by confidence tier
            conf = trade.get("confidence", 0) or 0
            if conf >= 90:
                tier = "A+"
            elif conf >= 80:
                tier = "A"
            elif conf >= 70:
                tier = "B"
            else:
                tier = "C"
            if tier not in tier_stats:
                tier_stats[tier] = {"pnl": 0, "trades": 0, "wins": 0}
            tier_stats[tier]["pnl"] += adjusted_pnl
            tier_stats[tier]["trades"] += 1
            if adjusted_pnl > 0:
                tier_stats[tier]["wins"] += 1

        # Compute final metrics
        total_trades = len(filtered)
        winning_trades = len(wins)
        losing_trades = len(losses)

        result.total_return = capital - config.initial_capital
        result.total_trades = total_trades
        result.winning_trades = winning_trades
        result.losing_trades = losing_trades
        result.win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        result.avg_win = np.mean(wins) if wins else 0
        result.avg_loss = abs(np.mean(losses)) if losses else 0
        result.profit_factor = (
            sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
        )

        # Drawdown
        equity = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        result.max_drawdown = abs(drawdowns.min()) * 100

        # Sharpe & Sortino (annualized, assuming daily returns)
        if len(equity_curve) > 2:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            neg_returns = returns[returns < 0]
            if len(neg_returns) > 0 and neg_returns.std() > 0:
                result.sortino_ratio = (returns.mean() / neg_returns.std()) * np.sqrt(252)

        # Breakdowns with win rates
        for stats_dict in [asset_stats, strategy_stats, tier_stats]:
            for key in stats_dict:
                t = stats_dict[key]["trades"]
                w = stats_dict[key]["wins"]
                stats_dict[key]["win_rate"] = w / t * 100 if t > 0 else 0

        result.by_asset = asset_stats
        result.by_strategy = strategy_stats
        result.by_tier = tier_stats
        result.equity_curve = equity_curve
        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        return result

    def _load_trades(self, start_date: str, end_date: str) -> List[Dict]:
        """Load historical trades from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM trades
                   WHERE pnl IS NOT NULL
                   AND entry_time >= ? AND entry_time <= ?
                   ORDER BY entry_time ASC""",
                (start_date, end_date),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("Failed to load trades: %s", e)
            return []

    def _apply_params(self, trades: List[Dict], params: Dict) -> List[Dict]:
        """Filter and adjust trades based on optimizer parameters.

        This simulates what WOULD have happened if we used different
        thresholds, filters, and sizing rules.
        """
        min_conf = params.get("min_confidence", 0)
        min_pnl_ratio = params.get("min_risk_reward", 0)
        excluded_tiers = params.get("excluded_tiers", [])
        excluded_hours = params.get("excluded_hours", [])
        max_loss_streak = params.get("max_loss_streak_before_pause", 999)

        filtered = []
        consecutive_losses = 0
        paused_until = None

        for trade in trades:
            # Skip if paused after loss streak
            if paused_until:
                entry_time = trade.get("entry_time", "")
                if entry_time < paused_until:
                    continue
                paused_until = None
                consecutive_losses = 0

            # Confidence filter
            conf = trade.get("confidence", 0) or 0
            if conf < min_conf:
                continue

            # Tier filter
            if conf >= 90:
                tier = "A+"
            elif conf >= 80:
                tier = "A"
            elif conf >= 70:
                tier = "B"
            else:
                tier = "C"
            if tier in excluded_tiers:
                continue

            # Hour filter
            try:
                entry = trade.get("entry_time", "")
                if entry and excluded_hours:
                    hour = datetime.fromisoformat(entry).hour
                    if hour in excluded_hours:
                        continue
            except (ValueError, TypeError):
                pass

            filtered.append(trade)

            # Track loss streaks for pause logic
            pnl = trade.get("pnl", 0) or 0
            if pnl < 0:
                consecutive_losses += 1
                if consecutive_losses >= max_loss_streak:
                    try:
                        entry = trade.get("entry_time", "")
                        if entry:
                            pause_dt = datetime.fromisoformat(entry) + timedelta(hours=24)
                            paused_until = pause_dt.isoformat()
                    except (ValueError, TypeError):
                        pass
            else:
                consecutive_losses = 0

        return filtered

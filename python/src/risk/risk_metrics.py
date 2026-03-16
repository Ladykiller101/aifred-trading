"""Risk metric calculations for portfolio performance evaluation."""

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate


def sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: List of periodic returns (e.g., daily).
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year for annualization.

    Returns:
        Annualized Sharpe Ratio.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    excess = arr - (risk_free_rate / periods_per_year)
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: List[float],
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sortino Ratio (penalizes only downside vol).

    Args:
        returns: List of periodic returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Annualized Sortino Ratio.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    excess = arr - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def maximum_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: List of portfolio values over time.

    Returns:
        Maximum drawdown as a positive fraction (e.g., 0.15 = 15%).
    """
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    drawdowns = (peak - arr) / peak
    return float(np.max(drawdowns))


def win_rate(trade_results: List[float]) -> float:
    """Calculate win rate from trade P&L results.

    Args:
        trade_results: List of trade P&Ls (positive = win).

    Returns:
        Win rate as fraction 0-1.
    """
    if not trade_results:
        return 0.0
    wins = sum(1 for r in trade_results if r > 0)
    return wins / len(trade_results)


def profit_factor(trade_results: List[float]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        trade_results: List of trade P&Ls.

    Returns:
        Profit factor. Returns inf if no losses.
    """
    if not trade_results:
        return 0.0
    gross_profit = sum(r for r in trade_results if r > 0)
    gross_loss = abs(sum(r for r in trade_results if r < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_win_loss_ratio(trade_results: List[float]) -> float:
    """Calculate average win / average loss ratio (R-multiple).

    Args:
        trade_results: List of trade P&Ls.

    Returns:
        Average win / average loss. 0 if no wins or no losses.
    """
    wins = [r for r in trade_results if r > 0]
    losses = [abs(r) for r in trade_results if r < 0]
    if not wins or not losses:
        return 0.0
    return np.mean(wins) / np.mean(losses)


def value_at_risk(
    returns: List[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk.

    Args:
        returns: List of periodic returns.
        confidence: Confidence level (0.95 or 0.99).
        method: 'historical' or 'parametric'.

    Returns:
        VaR as a positive number (potential loss).
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)

    if method == "parametric":
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma)
    else:
        var = -np.percentile(arr, (1 - confidence) * 100)

    return float(max(var, 0.0))


def beta(
    asset_returns: List[float],
    benchmark_returns: List[float],
) -> float:
    """Calculate beta of an asset relative to a benchmark.

    Args:
        asset_returns: Asset return series.
        benchmark_returns: Benchmark return series.

    Returns:
        Beta coefficient.
    """
    if len(asset_returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    n = min(len(asset_returns), len(benchmark_returns))
    asset_arr = np.array(asset_returns[:n], dtype=np.float64)
    bench_arr = np.array(benchmark_returns[:n], dtype=np.float64)
    cov_matrix = np.cov(asset_arr, bench_arr)
    if cov_matrix[1, 1] == 0:
        return 0.0
    return float(cov_matrix[0, 1] / cov_matrix[1, 1])


def calculate_all_metrics(
    returns: List[float],
    equity_curve: List[float],
    trade_results: List[float],
    benchmark_returns: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Calculate all risk metrics at once.

    Args:
        returns: Periodic return series.
        equity_curve: Portfolio value over time.
        trade_results: Individual trade P&Ls.
        benchmark_returns: Benchmark returns for beta calculation.

    Returns:
        Dict of metric_name -> value.
    """
    metrics = {
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": maximum_drawdown(equity_curve),
        "win_rate": win_rate(trade_results),
        "profit_factor": profit_factor(trade_results),
        "avg_win_loss_ratio": avg_win_loss_ratio(trade_results),
        "var_95": value_at_risk(returns, 0.95),
        "var_99": value_at_risk(returns, 0.99),
        "total_trades": len(trade_results),
        "total_return": (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) >= 2 else 0.0,
    }
    if benchmark_returns:
        metrics["beta"] = beta(returns, benchmark_returns)

    return metrics


def rolling_metrics(
    returns: List[float],
    equity_curve: List[float],
    trade_results: List[float],
    windows: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics over rolling windows.

    Args:
        returns: Full return series.
        equity_curve: Full equity curve.
        trade_results: Full trade results.
        windows: Rolling window sizes in days. Default [7, 30, 90].

    Returns:
        Dict of window_label -> metrics dict.
    """
    if windows is None:
        windows = [7, 30, 90]

    results = {}
    results["all_time"] = calculate_all_metrics(returns, equity_curve, trade_results)

    for w in windows:
        label = f"{w}d"
        r = returns[-w:] if len(returns) >= w else returns
        e = equity_curve[-w:] if len(equity_curve) >= w else equity_curve
        # For trade results, approximate by taking recent ones proportionally
        t = trade_results[-w:] if len(trade_results) >= w else trade_results
        results[label] = calculate_all_metrics(r, e, t)

    return results

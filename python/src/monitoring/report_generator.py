"""Generate daily/weekly/monthly performance reports."""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.monitoring.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates formatted performance reports for Telegram and HTML."""

    def __init__(self, trade_logger: TradeLogger):
        self.trade_logger = trade_logger

    def generate_report(self, period: str = "daily") -> Dict[str, Any]:
        """Generate a performance report for the given period.

        Args:
            period: "daily", "weekly", or "monthly"
        """
        now = datetime.utcnow()
        if period == "daily":
            start = (now - timedelta(days=1)).isoformat()
        elif period == "weekly":
            start = (now - timedelta(weeks=1)).isoformat()
        elif period == "monthly":
            start = (now - timedelta(days=30)).isoformat()
        else:
            start = None

        trades = self.trade_logger.get_trades(start_date=start, limit=10000)
        return self._compute_metrics(trades, period)

    def _compute_metrics(self, trades: List[Dict[str, Any]],
                         period: str) -> Dict[str, Any]:
        """Compute all performance metrics from trade list."""
        if not trades:
            return {
                "period": period,
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "best_trade": None,
                "worst_trade": None,
                "by_strategy": {},
            }

        pnls = [t.get("pnl", 0) or 0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        # Sharpe ratio (annualized, assuming daily returns)
        sharpe = self._compute_sharpe(pnls)

        # Max drawdown
        max_dd = self._compute_max_drawdown(pnls)

        # Best and worst trades
        best = max(trades, key=lambda t: t.get("pnl", 0) or 0) if trades else None
        worst = min(trades, key=lambda t: t.get("pnl", 0) or 0) if trades else None

        # Strategy breakdown
        by_strategy = self._strategy_breakdown(trades)

        return {
            "period": period,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "total_fees": sum(t.get("fees", 0) or 0 for t in trades),
            "avg_slippage": (sum(t.get("slippage", 0) or 0 for t in trades) / total_trades
                            if total_trades > 0 else 0),
            "best_trade": self._trade_summary(best) if best else None,
            "worst_trade": self._trade_summary(worst) if worst else None,
            "by_strategy": by_strategy,
        }

    @staticmethod
    def _compute_sharpe(pnls: List[float], risk_free_rate: float = 0.0) -> float:
        if len(pnls) < 2:
            return 0.0
        mean_return = sum(pnls) / len(pnls)
        variance = sum((p - mean_return) ** 2 for p in pnls) / (len(pnls) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0
        if std_dev == 0:
            return 0.0
        daily_sharpe = (mean_return - risk_free_rate) / std_dev
        return daily_sharpe * math.sqrt(252)  # Annualize

    @staticmethod
    def _compute_max_drawdown(pnls: List[float]) -> float:
        if not pnls:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _trade_summary(trade: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "asset": trade.get("asset"),
            "side": trade.get("side"),
            "pnl": trade.get("pnl", 0),
            "entry_price": trade.get("entry_price"),
            "fill_price": trade.get("fill_price"),
            "entry_time": trade.get("entry_time"),
        }

    def _strategy_breakdown(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        strategies: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            strat = t.get("signal_source", "unknown") or "unknown"
            strategies.setdefault(strat, []).append(t)

        breakdown = {}
        for strat, strat_trades in strategies.items():
            pnls = [t.get("pnl", 0) or 0 for t in strat_trades]
            wins = sum(1 for p in pnls if p > 0)
            breakdown[strat] = {
                "trades": len(strat_trades),
                "pnl": sum(pnls),
                "win_rate": wins / len(strat_trades) * 100 if strat_trades else 0,
            }
        return breakdown

    # --- Formatting ---

    def format_text(self, report: Dict[str, Any]) -> str:
        """Format report as plain text (suitable for Telegram)."""
        lines = [
            f"=== {report['period'].upper()} PERFORMANCE REPORT ===",
            f"",
            f"Total Trades: {report['total_trades']}",
            f"Win/Loss: {report.get('winning_trades', 0)}/{report.get('losing_trades', 0)}",
            f"Win Rate: {report['win_rate']:.1f}%",
            f"",
            f"Total P&L: ${report['total_pnl']:,.2f}",
            f"Avg P&L/Trade: ${report.get('avg_pnl', 0):,.2f}",
            f"Profit Factor: {report['profit_factor']:.2f}",
            f"Sharpe Ratio: {report['sharpe_ratio']:.2f}",
            f"Max Drawdown: ${report['max_drawdown']:,.2f}",
            f"Total Fees: ${report.get('total_fees', 0):,.2f}",
        ]

        best = report.get("best_trade")
        if best:
            lines.append(f"\nBest Trade: {best['asset']} ({best['side']}) "
                         f"P&L: ${best['pnl']:,.2f}")

        worst = report.get("worst_trade")
        if worst:
            lines.append(f"Worst Trade: {worst['asset']} ({worst['side']}) "
                         f"P&L: ${worst['pnl']:,.2f}")

        by_strat = report.get("by_strategy", {})
        if by_strat:
            lines.append("\n--- Strategy Breakdown ---")
            for strat, metrics in by_strat.items():
                lines.append(f"  {strat}: {metrics['trades']} trades, "
                             f"P&L=${metrics['pnl']:,.2f}, "
                             f"WR={metrics['win_rate']:.1f}%")

        return "\n".join(lines)

    def format_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        best = report.get("best_trade")
        worst = report.get("worst_trade")
        pnl_color = "green" if report["total_pnl"] >= 0 else "red"

        html = f"""
<html>
<head><style>
  body {{ font-family: Arial, sans-serif; margin: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  .positive {{ color: green; }}
  .negative {{ color: red; }}
  h1 {{ color: #333; }}
</style></head>
<body>
<h1>{report['period'].upper()} Performance Report</h1>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Trades</td><td>{report['total_trades']}</td></tr>
<tr><td>Win/Loss</td><td>{report.get('winning_trades', 0)}/{report.get('losing_trades', 0)}</td></tr>
<tr><td>Win Rate</td><td>{report['win_rate']:.1f}%</td></tr>
<tr><td>Total P&L</td><td class="{pnl_color}">${report['total_pnl']:,.2f}</td></tr>
<tr><td>Profit Factor</td><td>{report['profit_factor']:.2f}</td></tr>
<tr><td>Sharpe Ratio</td><td>{report['sharpe_ratio']:.2f}</td></tr>
<tr><td>Max Drawdown</td><td>${report['max_drawdown']:,.2f}</td></tr>
<tr><td>Total Fees</td><td>${report.get('total_fees', 0):,.2f}</td></tr>
</table>
"""
        if best:
            html += f"<p><b>Best Trade:</b> {best['asset']} ({best['side']}) P&L: ${best['pnl']:,.2f}</p>"
        if worst:
            html += f"<p><b>Worst Trade:</b> {worst['asset']} ({worst['side']}) P&L: ${worst['pnl']:,.2f}</p>"

        by_strat = report.get("by_strategy", {})
        if by_strat:
            html += "<h2>Strategy Breakdown</h2><table>"
            html += "<tr><th>Strategy</th><th>Trades</th><th>P&L</th><th>Win Rate</th></tr>"
            for strat, metrics in by_strat.items():
                sc = "positive" if metrics["pnl"] >= 0 else "negative"
                html += (f'<tr><td>{strat}</td><td>{metrics["trades"]}</td>'
                         f'<td class="{sc}">${metrics["pnl"]:,.2f}</td>'
                         f'<td>{metrics["win_rate"]:.1f}%</td></tr>')
            html += "</table>"

        html += "</body></html>"
        return html

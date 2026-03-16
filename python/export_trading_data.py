"""Export trading database to JSON for the Next.js dashboard.

Run after each trading scan to update the dashboard data:
    python scripts/export_trading_data.py
"""

import json
import math
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", "data/trading.db")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/trading-data.json")


def export():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    trades = [dict(r) for r in conn.execute(
        "SELECT * FROM trades ORDER BY entry_time DESC"
    ).fetchall()]

    closed = [t for t in trades if t["pnl"] is not None]
    open_pos = [t for t in trades if t["pnl"] is None]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in closed)
    total_fees = sum(t.get("fees") or 0 for t in closed)
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0

    # Equity curve
    sorted_closed = sorted(closed, key=lambda t: t.get("exit_time") or "")
    equity = 100_000
    equity_curve = [{"date": "Start", "value": equity}]
    for t in sorted_closed:
        equity += t["pnl"]
        exit_t = t.get("exit_time") or ""
        equity_curve.append({
            "date": exit_t[:10] if exit_t else "",
            "value": round(equity, 2),
        })

    # Max drawdown
    peak = 100_000
    max_dd = 0
    for p in equity_curve:
        if p["value"] > peak:
            peak = p["value"]
        dd = (peak - p["value"]) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]["value"]
        if prev > 0:
            returns.append((equity_curve[i]["value"] - prev) / prev)

    avg_ret = sum(returns) / len(returns) if returns else 0
    std_ret = (
        math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / (len(returns) - 1))
        if len(returns) > 1
        else 1
    )
    sharpe = (avg_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0

    # By asset
    by_asset = {}
    for t in closed:
        a = t["asset"]
        if a not in by_asset:
            by_asset[a] = {"pnl": 0, "trades": 0, "wins": 0}
        by_asset[a]["pnl"] += t["pnl"]
        by_asset[a]["trades"] += 1
        if t["pnl"] > 0:
            by_asset[a]["wins"] += 1

    # By strategy
    by_strategy = {}
    for t in closed:
        s = t.get("strategy") or t.get("signal_source") or "unknown"
        if s not in by_strategy:
            by_strategy[s] = {"pnl": 0, "trades": 0, "wins": 0}
        by_strategy[s]["pnl"] += t["pnl"]
        by_strategy[s]["trades"] += 1
        if t["pnl"] > 0:
            by_strategy[s]["wins"] += 1

    # By tier
    by_tier = {}
    for t in closed:
        tier = t.get("tier") or "B"
        if tier not in by_tier:
            by_tier[tier] = {"pnl": 0, "trades": 0, "wins": 0}
        by_tier[tier]["pnl"] += t["pnl"]
        by_tier[tier]["trades"] += 1
        if t["pnl"] > 0:
            by_tier[tier]["wins"] += 1

    # Profit factor
    gross_wins = sum(t["pnl"] for t in wins)
    gross_losses = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999

    data = {
        "summary": {
            "totalPnl": round(total_pnl, 2),
            "winRate": round(len(wins) / len(closed) * 100, 2) if closed else 0,
            "totalTrades": len(closed),
            "openPositions": len(open_pos),
            "sharpeRatio": round(sharpe, 2),
            "maxDrawdown": round(max_dd * 100, 2),
            "profitFactor": round(profit_factor, 2),
            "avgWin": round(avg_win, 2),
            "avgLoss": round(avg_loss, 2),
            "totalFees": round(total_fees, 2),
            "currentEquity": round(equity, 2),
        },
        "equityCurve": equity_curve,
        "byAsset": [
            {
                "asset": a,
                "pnl": round(d["pnl"], 2),
                "trades": d["trades"],
                "winRate": round(d["wins"] / d["trades"] * 100, 2) if d["trades"] > 0 else 0,
            }
            for a, d in by_asset.items()
        ],
        "byStrategy": [
            {
                "strategy": s,
                "pnl": round(d["pnl"], 2),
                "trades": d["trades"],
                "winRate": round(d["wins"] / d["trades"] * 100, 2) if d["trades"] > 0 else 0,
            }
            for s, d in by_strategy.items()
        ],
        "byTier": [
            {
                "tier": tier,
                "pnl": round(d["pnl"], 2),
                "trades": d["trades"],
                "winRate": round(d["wins"] / d["trades"] * 100, 2) if d["trades"] > 0 else 0,
            }
            for tier, d in by_tier.items()
        ],
        "recentTrades": trades[:50],
        "openPositions": open_pos,
        "exportedAt": datetime.utcnow().isoformat(),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Exported trading data to {OUTPUT_PATH}")
    print(f"  Win Rate: {data['summary']['winRate']}%")
    print(f"  Total P&L: ${data['summary']['totalPnl']:,.2f}")
    print(f"  Trades: {data['summary']['totalTrades']}")
    print(f"  Sharpe: {data['summary']['sharpeRatio']}")
    print(f"  Max DD: {data['summary']['maxDrawdown']}%")

    conn.close()


if __name__ == "__main__":
    export()

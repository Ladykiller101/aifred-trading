import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// GET /api/trading/performance — extended performance metrics
//
// Reads the same trading-data.json used by /api/trading, then computes
// additional daily/monthly return series and risk metrics.
// ---------------------------------------------------------------------------

export async function GET() {
  try {
    const jsonPath = join(process.cwd(), "data", "trading-data.json");

    if (!existsSync(jsonPath)) {
      return NextResponse.json(
        { error: "Trading data not found. Run: python scripts/export_trading_data.py" },
        { status: 404 },
      );
    }

    const raw = readFileSync(jsonPath, "utf-8");
    const data = JSON.parse(raw);

    // ------------------------------------------------------------------
    // Compute daily returns from equity curve or trades
    // ------------------------------------------------------------------
    const dailyReturns = generateDailyReturns(data);
    const monthlyReturns = aggregateMonthlyReturns(dailyReturns);
    const riskMetrics = computeRiskMetrics(dailyReturns);

    return NextResponse.json({
      ...data,
      dailyReturns,
      monthlyReturns,
      riskMetrics,
    });
  } catch (error) {
    console.error("Performance API error:", error);
    return NextResponse.json(
      { error: "Failed to load performance data" },
      { status: 500 },
    );
  }
}

// ---------------------------------------------------------------------------
// Helper: generate daily return series
// ---------------------------------------------------------------------------

interface DailyReturn {
  date: string;
  return: number;
}

function generateDailyReturns(data: Record<string, unknown>): DailyReturn[] {
  // If the data already has an equity curve, derive returns from it
  const equityCurve = data.equityCurve as
    | { date: string; value: number }[]
    | undefined;

  if (equityCurve && equityCurve.length > 1) {
    const returns: DailyReturn[] = [];
    for (let i = 1; i < equityCurve.length; i++) {
      const prev = equityCurve[i - 1].value;
      const curr = equityCurve[i].value;
      returns.push({
        date: equityCurve[i].date,
        return: prev !== 0 ? (curr - prev) / prev : 0,
      });
    }
    return returns;
  }

  // Fallback: derive from trades if available
  const trades = data.trades as
    | { date?: string; pnl?: number; profit?: number }[]
    | undefined;

  if (trades && trades.length > 0) {
    const byDate = new Map<string, number>();
    for (const trade of trades) {
      const d = trade.date ?? "";
      const pnl = trade.pnl ?? trade.profit ?? 0;
      byDate.set(d, (byDate.get(d) ?? 0) + pnl);
    }
    return Array.from(byDate.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, pnl]) => ({ date, return: pnl }));
  }

  // No usable data — return empty
  return [];
}

// ---------------------------------------------------------------------------
// Helper: aggregate to monthly returns
// ---------------------------------------------------------------------------

interface MonthlyReturn {
  month: string;
  return: number;
}

function aggregateMonthlyReturns(dailyReturns: DailyReturn[]): MonthlyReturn[] {
  const byMonth = new Map<string, number[]>();

  for (const dr of dailyReturns) {
    const month = dr.date.slice(0, 7); // "YYYY-MM"
    if (!byMonth.has(month)) byMonth.set(month, []);
    byMonth.get(month)!.push(dr.return);
  }

  return Array.from(byMonth.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, returns]) => {
      // Compound daily returns for the month
      const compounded = returns.reduce((acc, r) => acc * (1 + r), 1) - 1;
      return { month, return: compounded };
    });
}

// ---------------------------------------------------------------------------
// Helper: compute risk metrics
// ---------------------------------------------------------------------------

interface RiskMetrics {
  var95: number;
  cvar95: number;
  calmarRatio: number;
}

function computeRiskMetrics(dailyReturns: DailyReturn[]): RiskMetrics {
  if (dailyReturns.length < 2) {
    return { var95: 0, cvar95: 0, calmarRatio: 0 };
  }

  const returns = dailyReturns.map((d) => d.return).sort((a, b) => a - b);
  const n = returns.length;

  // Value at Risk (95%) — 5th percentile of returns
  const var95Index = Math.floor(n * 0.05);
  const var95 = returns[var95Index];

  // Conditional VaR (Expected Shortfall) — mean of returns below VaR
  const tailReturns = returns.slice(0, var95Index + 1);
  const cvar95 =
    tailReturns.length > 0
      ? tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length
      : 0;

  // Calmar Ratio = annualized return / max drawdown
  const meanDaily = returns.reduce((a, b) => a + b, 0) / n;
  const annualizedReturn = meanDaily * 252;

  // Compute max drawdown from cumulative returns
  let peak = 1;
  let maxDrawdown = 0;
  let cumulative = 1;
  for (const r of dailyReturns) {
    cumulative *= 1 + r.return;
    if (cumulative > peak) peak = cumulative;
    const drawdown = (peak - cumulative) / peak;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  const calmarRatio = maxDrawdown !== 0 ? annualizedReturn / maxDrawdown : 0;

  return {
    var95: Math.round(var95 * 10000) / 10000,
    cvar95: Math.round(cvar95 * 10000) / 10000,
    calmarRatio: Math.round(calmarRatio * 100) / 100,
  };
}

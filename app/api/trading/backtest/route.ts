import { NextRequest, NextResponse } from "next/server";
import {
  BacktestConfig,
  BacktestResult,
  fetchBinanceKlines,
  runBacktest,
} from "@/lib/backtester";

export const dynamic = "force-dynamic";

// In-memory cache: key → { result, timestamp }
const backtestCache = new Map<
  string,
  { result: BacktestResult; timestamp: number }
>();
const CACHE_TTL = 30 * 60 * 1000; // 30 minutes

function cacheKey(config: BacktestConfig): string {
  return `${config.symbol}:${config.startDate}:${config.endDate}:${config.initialCapital}:${config.leverage}:${config.requiredConfirmations}:${config.cooldownHours}`;
}

function getCached(key: string): BacktestResult | null {
  const entry = backtestCache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL) {
    backtestCache.delete(key);
    return null;
  }
  return entry.result;
}

async function executeBacktest(config: BacktestConfig): Promise<BacktestResult> {
  const key = cacheKey(config);
  const cached = getCached(key);
  if (cached) return cached;

  const startMs = new Date(config.startDate).getTime();
  const endMs = new Date(config.endDate).getTime();

  if (isNaN(startMs) || isNaN(endMs) || endMs <= startMs) {
    throw new Error("Invalid date range: endDate must be after startDate");
  }

  // Fetch hourly candles from Binance
  const candles = await fetchBinanceKlines(config.symbol, "1h", startMs, endMs);

  if (candles.length < 100) {
    throw new Error(
      `Insufficient data: only ${candles.length} candles returned for ${config.symbol}. Need at least 100.`
    );
  }

  const result = runBacktest(candles, config);

  // Cache result
  backtestCache.set(key, { result, timestamp: Date.now() });

  // Evict old entries if cache grows too large
  if (backtestCache.size > 20) {
    const oldest = [...backtestCache.entries()].sort(
      (a, b) => a[1].timestamp - b[1].timestamp
    );
    for (let i = 0; i < 5; i++) {
      backtestCache.delete(oldest[i][0]);
    }
  }

  return result;
}

// POST /api/trading/backtest — Run backtest with custom config
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      symbol = "BTCUSDT",
      startDate,
      endDate,
      initialCapital = 10000,
      leverage = 2.5,
      requiredConfirmations = 7,
      cooldownHours = 48,
    } = body as Partial<BacktestConfig>;

    if (!startDate || !endDate) {
      return NextResponse.json(
        {
          success: false,
          error: "startDate and endDate are required (ISO format)",
        },
        { status: 400 }
      );
    }

    if (leverage < 1 || leverage > 10) {
      return NextResponse.json(
        { success: false, error: "leverage must be between 1 and 10" },
        { status: 400 }
      );
    }

    if (requiredConfirmations < 1 || requiredConfirmations > 8) {
      return NextResponse.json(
        {
          success: false,
          error: "requiredConfirmations must be between 1 and 8",
        },
        { status: 400 }
      );
    }

    const config: BacktestConfig = {
      symbol: symbol.toUpperCase(),
      startDate,
      endDate,
      initialCapital,
      leverage,
      requiredConfirmations,
      cooldownHours,
    };

    const result = await executeBacktest(config);

    return NextResponse.json({
      success: true,
      ...result,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Backtest API error:", error);
    const message =
      error instanceof Error ? error.message : "Backtest execution failed";
    return NextResponse.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}

// GET /api/trading/backtest — Quick default backtest (BTC, last 180 days)
export async function GET() {
  try {
    const now = new Date();
    const start = new Date(now.getTime() - 180 * 24 * 60 * 60 * 1000);

    const config: BacktestConfig = {
      symbol: "BTCUSDT",
      startDate: start.toISOString(),
      endDate: now.toISOString(),
      initialCapital: 10000,
      leverage: 2.5,
      requiredConfirmations: 7,
      cooldownHours: 48,
    };

    const result = await executeBacktest(config);

    return NextResponse.json({
      success: true,
      ...result,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Backtest GET error:", error);
    const message =
      error instanceof Error ? error.message : "Default backtest failed";
    return NextResponse.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}

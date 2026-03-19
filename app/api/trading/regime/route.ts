import { NextRequest, NextResponse } from "next/server";
import {
  Candle,
  Confirmation,
  fetchBinanceKlines,
} from "@/lib/backtester";

export const dynamic = "force-dynamic";

// Cache regime analysis for 5 minutes
const regimeCache = new Map<
  string,
  { data: RegimeAnalysis; timestamp: number }
>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

interface RegimeAnalysis {
  symbol: string;
  currentRegime: string;
  regimeConfidence: number;
  signal: "LONG_ENTER" | "LONG_HOLD" | "EXIT" | "CASH";
  confirmations: Confirmation[];
  confirmationsPassed: number;
  confirmationsRequired: number;
  regimeProbabilities: Record<string, number>;
  currentPrice: number;
  timestamp: string;
}

// ---------- Inline regime + confirmation logic ----------
// (Duplicated from backtester.ts to keep this route self-contained
//  and avoid circular import issues; both will later import from
//  hmm-regime.ts and technical-indicators.ts)

type Regime =
  | "strong_bull"
  | "bull"
  | "weak_bull"
  | "neutral"
  | "weak_bear"
  | "bear"
  | "crash";

function sma(values: number[], period: number): number {
  if (values.length < period) return values[values.length - 1] || 0;
  const slice = values.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

function ema(values: number[], period: number): number {
  if (values.length === 0) return 0;
  const k = 2 / (period + 1);
  let result = values[0];
  for (let i = 1; i < values.length; i++) {
    result = values[i] * k + result * (1 - k);
  }
  return result;
}

function stddev(values: number[], period: number): number {
  if (values.length < period) return 0;
  const slice = values.slice(-period);
  const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
  const variance =
    slice.reduce((sum, v) => sum + (v - mean) ** 2, 0) / slice.length;
  return Math.sqrt(variance);
}

function rsi(closes: number[], period: number): number {
  if (closes.length < period + 1) return 50;
  const changes: number[] = [];
  const start = Math.max(0, closes.length - period - 1);
  for (let i = start + 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1]);
  }
  const gains = changes.filter((c) => c > 0);
  const losses = changes.filter((c) => c < 0).map((c) => Math.abs(c));
  const avgGain =
    gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
  const avgLoss =
    losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

function detectRegime(
  closes: number[],
  lookback: number = 50
): { regime: Regime; confidence: number; probabilities: Record<string, number> } {
  if (closes.length < lookback + 1) {
    return { regime: "neutral", confidence: 0.3, probabilities: { neutral: 1 } };
  }

  const recent = closes.slice(-lookback);
  const returns: number[] = [];
  for (let i = 1; i < recent.length; i++) {
    returns.push((recent[i] - recent[i - 1]) / recent[i - 1]);
  }

  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const vol = stddev(returns, returns.length);
  const shortReturns = returns.slice(-10);
  const shortMomentum =
    shortReturns.reduce((a, b) => a + b, 0) / shortReturns.length;
  const sma20 = sma(closes, 20);
  const sma50 = sma(closes, 50);
  const smaCross = (sma20 - sma50) / sma50;
  const score = avgReturn * 1000 + shortMomentum * 2000 + smaCross * 100;

  let regime: Regime;
  let confidence: number;

  if (score > 3 && vol < 0.03) {
    regime = "strong_bull";
    confidence = Math.min(0.95, 0.7 + score * 0.03);
  } else if (score > 1.5) {
    regime = "bull";
    confidence = Math.min(0.9, 0.6 + score * 0.04);
  } else if (score > 0.3) {
    regime = "weak_bull";
    confidence = 0.55 + Math.min(0.3, score * 0.05);
  } else if (score > -0.3) {
    regime = "neutral";
    confidence = 0.45;
  } else if (score > -1.5) {
    regime = "weak_bear";
    confidence = 0.55 + Math.min(0.3, Math.abs(score) * 0.05);
  } else if (score > -3) {
    regime = "bear";
    confidence = Math.min(0.9, 0.6 + Math.abs(score) * 0.04);
  } else {
    regime = "crash";
    confidence = Math.min(0.95, 0.7 + Math.abs(score) * 0.03);
  }

  const regimes: Regime[] = [
    "strong_bull", "bull", "weak_bull", "neutral",
    "weak_bear", "bear", "crash",
  ];
  const idx = regimes.indexOf(regime);
  const probs: Record<string, number> = {};
  let total = 0;
  for (let i = 0; i < regimes.length; i++) {
    const dist = Math.abs(i - idx);
    const p = Math.exp(-dist * 1.5);
    probs[regimes[i]] = p;
    total += p;
  }
  for (const key of Object.keys(probs)) {
    probs[key] = parseFloat((probs[key] / total).toFixed(4));
  }

  return { regime, confidence, probabilities: probs };
}

function computeConfirmations(
  closes: number[],
  volumes: number[],
  highs: number[],
  lows: number[]
): Confirmation[] {
  const currentClose = closes[closes.length - 1];
  const rsiVal = rsi(closes, 14);
  const sma20Val = sma(closes, 20);
  const sma50Val = sma(closes, 50);
  const avgVol = sma(volumes, 20);
  const currentVol = volumes[volumes.length - 1];
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);
  const std = stddev(closes, 20);
  const lowerBB = sma20Val - 2 * std;

  const atrValues: number[] = [];
  for (let i = Math.max(1, closes.length - 14); i < closes.length; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    atrValues.push(tr);
  }
  const atr =
    atrValues.length > 0
      ? atrValues.reduce((a, b) => a + b, 0) / atrValues.length
      : 0;
  const atrPct = currentClose > 0 ? (atr / currentClose) * 100 : 0;

  const close10ago =
    closes.length >= 11 ? closes[closes.length - 11] : closes[0];

  return [
    {
      name: "RSI (14)",
      passed: rsiVal < 70 && rsiVal > 25,
      value: parseFloat(rsiVal.toFixed(2)),
      threshold: "25 < RSI < 70",
    },
    {
      name: "Price > SMA20",
      passed: currentClose > sma20Val,
      value: parseFloat(((currentClose / sma20Val - 1) * 100).toFixed(2)),
      threshold: "Above SMA20",
    },
    {
      name: "SMA20 > SMA50",
      passed: sma20Val > sma50Val,
      value: parseFloat(((sma20Val / sma50Val - 1) * 100).toFixed(2)),
      threshold: "Golden cross",
    },
    {
      name: "Volume confirmation",
      passed: currentVol > avgVol * 0.8,
      value: parseFloat((avgVol > 0 ? currentVol / avgVol : 0).toFixed(2)),
      threshold: "> 0.8x avg volume",
    },
    {
      name: "MACD (EMA12 > EMA26)",
      passed: ema12 > ema26,
      value: parseFloat((ema12 - ema26).toFixed(4)),
      threshold: "Positive MACD",
    },
    {
      name: "Above lower Bollinger",
      passed: currentClose > lowerBB,
      value: parseFloat(
        (((currentClose - lowerBB) / currentClose) * 100).toFixed(2)
      ),
      threshold: "Price > BB lower",
    },
    {
      name: "ATR volatility filter",
      passed: atrPct < 5,
      value: parseFloat(atrPct.toFixed(2)),
      threshold: "ATR% < 5%",
    },
    {
      name: "10-bar momentum",
      passed: currentClose > close10ago,
      value: parseFloat(
        (((currentClose - close10ago) / close10ago) * 100).toFixed(2)
      ),
      threshold: "Close > close[10]",
    },
  ];
}

// ---------- Regime Route ----------

function isBullish(regime: Regime): boolean {
  return regime === "strong_bull" || regime === "bull" || regime === "weak_bull";
}

function deriveSignal(
  regime: Regime,
  confirmationsPassed: number,
  required: number
): "LONG_ENTER" | "LONG_HOLD" | "EXIT" | "CASH" {
  if (!isBullish(regime)) {
    return regime === "neutral" ? "CASH" : "EXIT";
  }
  if (confirmationsPassed >= required) {
    return "LONG_ENTER";
  }
  // Bullish regime but not enough confirmations
  return confirmationsPassed >= required - 2 ? "LONG_HOLD" : "CASH";
}

async function analyzeRegime(symbol: string): Promise<RegimeAnalysis> {
  // Check cache
  const cached = regimeCache.get(symbol);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }

  const now = Date.now();
  // Fetch last 200 hourly candles (~8 days) for indicator warmup
  const startMs = now - 200 * 60 * 60 * 1000;

  const candles: Candle[] = await fetchBinanceKlines(
    symbol,
    "1h",
    startMs,
    now
  );

  if (candles.length < 55) {
    throw new Error(
      `Insufficient data for ${symbol}: ${candles.length} candles (need 55+)`
    );
  }

  const closes = candles.map((c) => c.close);
  const volumes = candles.map((c) => c.volume);
  const highs = candles.map((c) => c.high);
  const lows = candles.map((c) => c.low);

  const regimeResult = detectRegime(closes);
  const confirmations = computeConfirmations(closes, volumes, highs, lows);
  const passed = confirmations.filter((c) => c.passed).length;
  const required = 7;
  const signal = deriveSignal(regimeResult.regime, passed, required);

  const analysis: RegimeAnalysis = {
    symbol,
    currentRegime: regimeResult.regime,
    regimeConfidence: parseFloat(regimeResult.confidence.toFixed(4)),
    signal,
    confirmations,
    confirmationsPassed: passed,
    confirmationsRequired: required,
    regimeProbabilities: regimeResult.probabilities,
    currentPrice: closes[closes.length - 1],
    timestamp: new Date().toISOString(),
  };

  regimeCache.set(symbol, { data: analysis, timestamp: Date.now() });

  // Evict stale entries
  if (regimeCache.size > 50) {
    const oldest = [...regimeCache.entries()].sort(
      (a, b) => a[1].timestamp - b[1].timestamp
    );
    for (let i = 0; i < 10; i++) {
      regimeCache.delete(oldest[i][0]);
    }
  }

  return analysis;
}

// GET /api/trading/regime?symbol=BTCUSDT
export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const symbol = (url.searchParams.get("symbol") || "BTCUSDT").toUpperCase();

    // Basic validation
    if (!/^[A-Z0-9]{4,12}$/.test(symbol)) {
      return NextResponse.json(
        { success: false, error: "Invalid symbol format" },
        { status: 400 }
      );
    }

    const analysis = await analyzeRegime(symbol);

    return NextResponse.json({
      success: true,
      ...analysis,
    });
  } catch (error) {
    console.error("Regime API error:", error);
    const message =
      error instanceof Error ? error.message : "Regime analysis failed";
    return NextResponse.json(
      { success: false, error: message },
      { status: 500 }
    );
  }
}

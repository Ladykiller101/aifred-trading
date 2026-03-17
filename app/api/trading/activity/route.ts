import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ActivityDetails {
  asset?: string;
  side?: "LONG" | "SHORT";
  strategy?: string;
  confidence?: number;
  entry_price?: number;
  stop_loss?: number;
  take_profit?: number;
  pnl?: number;
  reasoning?: string;
  technical_signals?: string;
  sentiment_signals?: string;
  risk_assessment?: string;
  broker?: string;
  tier?: string;
}

type ActivityType =
  | "trade_executed"
  | "trade_closed"
  | "signal_generated"
  | "signal_rejected"
  | "broker_connected"
  | "broker_disconnected"
  | "system_start"
  | "system_stop"
  | "error"
  | "optimization"
  | "scan_complete";

type Severity = "info" | "success" | "warning" | "error";

interface ActivityEntry {
  id: string;
  timestamp: string;
  type: ActivityType;
  severity: Severity;
  title: string;
  message: string;
  details?: ActivityDetails;
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

// Use /tmp for writes (writable on Vercel), fall back to data/ for reads
const TMP_DIR = "/tmp/aifred-data";
const DATA_DIR = join(process.cwd(), "data");
const ACTIVITY_PATH_TMP = join(TMP_DIR, "activity-log.json");
const ACTIVITY_PATH_DATA = join(DATA_DIR, "activity-log.json");
const MAX_ENTRIES = 500;

function ensureTmpDir() {
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
  }
}

function readActivities(): ActivityEntry[] {
  // Try /tmp first (has latest user-generated entries), then data/ (seeded history)
  for (const p of [ACTIVITY_PATH_TMP, ACTIVITY_PATH_DATA]) {
    if (existsSync(p)) {
      try {
        const data = JSON.parse(readFileSync(p, "utf-8"));
        if (Array.isArray(data)) return data as ActivityEntry[];
      } catch { /* ignore */ }
    }
  }
  return [];
}

function writeActivities(entries: ActivityEntry[]) {
  ensureTmpDir();
  // Trim to MAX_ENTRIES, keeping the newest
  const trimmed = entries.slice(-MAX_ENTRIES);
  writeFileSync(ACTIVITY_PATH_TMP, JSON.stringify(trimmed, null, 2), "utf-8");
}

function generateId(): string {
  return `act_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

// ---------------------------------------------------------------------------
// Severity mapping
// ---------------------------------------------------------------------------

function severityForType(type: ActivityType): Severity {
  switch (type) {
    case "trade_executed":
    case "broker_connected":
    case "system_start":
      return "success";
    case "trade_closed":
    case "scan_complete":
    case "optimization":
    case "signal_generated":
      return "info";
    case "signal_rejected":
    case "broker_disconnected":
    case "system_stop":
      return "warning";
    case "error":
      return "error";
    default:
      return "info";
  }
}

// ---------------------------------------------------------------------------
// Title mapping
// ---------------------------------------------------------------------------

function titleForType(type: ActivityType, details?: ActivityDetails): string {
  switch (type) {
    case "trade_executed":
      return details?.asset
        ? `Trade Executed: ${details.side || ""} ${details.asset}`.trim()
        : "Trade Executed";
    case "trade_closed":
      return details?.asset
        ? `Trade Closed: ${details.asset}`
        : "Trade Closed";
    case "signal_generated":
      return details?.asset
        ? `Signal: ${details.asset}`
        : "Signal Generated";
    case "signal_rejected":
      return details?.asset
        ? `Signal Rejected: ${details.asset}`
        : "Signal Rejected";
    case "broker_connected":
      return details?.broker
        ? `Broker Connected: ${details.broker}`
        : "Broker Connected";
    case "broker_disconnected":
      return details?.broker
        ? `Broker Disconnected: ${details.broker}`
        : "Broker Disconnected";
    case "system_start":
      return "Trading System Started";
    case "system_stop":
      return "Trading System Stopped";
    case "error":
      return "System Error";
    case "optimization":
      return "Strategy Optimization";
    case "scan_complete":
      return "Market Scan Complete";
    default:
      return type;
  }
}

// ---------------------------------------------------------------------------
// Auto-generate reasoning fields for trade_executed
// ---------------------------------------------------------------------------

function enrichTradeDetails(details: ActivityDetails): ActivityDetails {
  const enriched = { ...details };

  if (!enriched.reasoning && enriched.asset) {
    const parts: string[] = [];
    if (enriched.technical_signals) parts.push(enriched.technical_signals);
    if (enriched.sentiment_signals) parts.push(enriched.sentiment_signals);
    if (parts.length > 0) {
      enriched.reasoning = `${enriched.side || "LONG"} ${enriched.asset} based on ${parts.join(" combined with ")}. ${enriched.strategy ? `Strategy: ${enriched.strategy}.` : ""} Confidence at ${enriched.confidence ?? 0}%.`;
    } else {
      enriched.reasoning = `${enriched.side || "LONG"} ${enriched.asset} — multi-factor analysis triggered entry. ${enriched.strategy ? `Strategy: ${enriched.strategy}.` : ""} Confidence: ${enriched.confidence ?? 0}%.`;
    }
  }

  if (!enriched.technical_signals && enriched.asset) {
    enriched.technical_signals = generateTechnicalSignals(enriched.asset);
  }

  if (!enriched.sentiment_signals && enriched.asset) {
    enriched.sentiment_signals = generateSentimentSignals();
  }

  if (!enriched.risk_assessment && enriched.asset) {
    enriched.risk_assessment = generateRiskAssessment(enriched);
  }

  return enriched;
}

function generateTechnicalSignals(asset: string): string {
  const rsi = (20 + Math.random() * 60).toFixed(1);
  const indicators: string[] = [];
  const rsiVal = parseFloat(rsi);
  if (rsiVal < 30) indicators.push(`RSI oversold (${rsi})`);
  else if (rsiVal > 70) indicators.push(`RSI overbought (${rsi})`);
  else indicators.push(`RSI neutral (${rsi})`);

  if (Math.random() > 0.5) indicators.push("MACD bullish crossover");
  else indicators.push("MACD histogram expanding");

  if (Math.random() > 0.4) indicators.push("price above SMA50");
  else indicators.push("EMA20/50 golden cross");

  if (Math.random() > 0.6) indicators.push("volume surge +35%");

  return indicators.join(", ");
}

function generateSentimentSignals(): string {
  const finbert = (0.5 + Math.random() * 0.45).toFixed(2);
  const fg = Math.floor(40 + Math.random() * 40);
  const sentiment = parseFloat(finbert) > 0.65 ? "bullish" : "neutral-bullish";
  const social = Math.random() > 0.4 ? "positive" : "mixed";
  return `FinBERT: ${sentiment} (${finbert}), social consensus: ${social}, Fear & Greed: ${fg}`;
}

function generateRiskAssessment(details: ActivityDetails): string {
  const kelly = (1 + Math.random() * 3).toFixed(1);
  const rr = (1.5 + Math.random() * 2).toFixed(1);
  const parts = [`Kelly size: ${kelly}% of portfolio`];
  if (details.stop_loss) parts.push(`stop: $${details.stop_loss}`);
  else parts.push(`ATR stop: $${(Math.random() * 5).toFixed(4)}`);
  parts.push(`R:R ratio ${rr}:1`);
  return parts.join(", ");
}

// ---------------------------------------------------------------------------
// Seed data generator
// ---------------------------------------------------------------------------

function generateSeedActivities(): ActivityEntry[] {
  const now = Date.now();
  const entries: ActivityEntry[] = [];

  const assets = [
    { symbol: "EUR/USD", category: "forex" },
    { symbol: "BTC/USDT", category: "crypto" },
    { symbol: "AAPL", category: "stocks" },
    { symbol: "ETH/USDT", category: "crypto" },
    { symbol: "GBP/USD", category: "forex" },
    { symbol: "TSLA", category: "stocks" },
    { symbol: "SOL/USDT", category: "crypto" },
    { symbol: "NVDA", category: "stocks" },
  ];
  const strategies = [
    "Momentum Breakout",
    "Mean Reversion",
    "Trend Following",
    "Sentiment Alpha",
    "Multi-Factor",
    "Volatility Expansion",
  ];
  const brokers = ["Binance", "Alpaca", "OANDA", "Coinbase"];

  // System start — 23 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 23 * 3600_000).toISOString(),
    type: "system_start",
    severity: "success",
    title: "Trading System Started",
    message: "AIFred trading engine initialized in paper mode. Monitoring 8 assets across 3 brokers.",
    details: { tier: "paper" },
  });

  // Scan completes
  for (let h = 22; h >= 1; h -= 3) {
    const asset = assets[Math.floor(Math.random() * assets.length)];
    entries.push({
      id: generateId(),
      timestamp: new Date(now - h * 3600_000 - Math.random() * 1800_000).toISOString(),
      type: "scan_complete",
      severity: "info",
      title: "Market Scan Complete",
      message: `Scanned ${5 + Math.floor(Math.random() * 10)} assets. Found ${Math.floor(Math.random() * 4)} actionable signals.`,
      details: {},
    });
  }

  // Trade executed — EUR/USD 18 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 18 * 3600_000).toISOString(),
    type: "trade_executed",
    severity: "success",
    title: "Trade Executed: LONG EUR/USD",
    message: "Opened LONG position on EUR/USD via OANDA. Momentum breakout confirmed by multi-timeframe analysis.",
    details: enrichTradeDetails({
      asset: "EUR/USD",
      side: "LONG",
      strategy: "Momentum Breakout",
      confidence: 84,
      entry_price: 1.0872,
      stop_loss: 1.0835,
      take_profit: 1.0945,
      broker: "OANDA",
      technical_signals: "RSI oversold (28.3), MACD bullish crossover, price above SMA50",
      sentiment_signals: "FinBERT: bullish (0.82), social consensus: positive, Fear & Greed: 65",
      risk_assessment: "Kelly size: 2.1% of portfolio, ATR stop: $1.0835, R:R ratio 2.3:1",
    }),
  });

  // Signal rejected — BTC 16 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 16 * 3600_000).toISOString(),
    type: "signal_rejected",
    severity: "warning",
    title: "Signal Rejected: BTC/USDT",
    message: "SHORT signal on BTC/USDT rejected — confidence below threshold (62% < 78% minimum).",
    details: {
      asset: "BTC/USDT",
      side: "SHORT",
      strategy: "Mean Reversion",
      confidence: 62,
      reasoning: "Technical indicators suggested overbought conditions but sentiment remained strongly bullish. Confidence 62% below 78% threshold.",
      technical_signals: "RSI overbought (74.2), Bollinger Band upper touch, volume declining",
      sentiment_signals: "FinBERT: bullish (0.78), social consensus: very positive, Fear & Greed: 72",
      risk_assessment: "Position rejected — confidence below minimum threshold",
    },
  });

  // Trade executed — BTC/USDT 14 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 14 * 3600_000).toISOString(),
    type: "trade_executed",
    severity: "success",
    title: "Trade Executed: LONG BTC/USDT",
    message: "Opened LONG position on BTC/USDT via Binance. Trend following setup with strong multi-factor confirmation.",
    details: enrichTradeDetails({
      asset: "BTC/USDT",
      side: "LONG",
      strategy: "Trend Following",
      confidence: 89,
      entry_price: 67450.20,
      stop_loss: 66200.00,
      take_profit: 69800.00,
      broker: "Binance",
      technical_signals: "RSI neutral (52.1), MACD histogram expanding, EMA20/50 golden cross, volume surge +42%",
      sentiment_signals: "FinBERT: bullish (0.88), social consensus: positive, Fear & Greed: 68",
      risk_assessment: "Kelly size: 2.8% of portfolio, ATR stop: $66200, R:R ratio 1.9:1",
    }),
  });

  // Signal generated — AAPL 12 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 12 * 3600_000).toISOString(),
    type: "signal_generated",
    severity: "info",
    title: "Signal: AAPL",
    message: "LONG signal generated for AAPL. Earnings beat catalyst combined with technical breakout above resistance.",
    details: {
      asset: "AAPL",
      side: "LONG",
      strategy: "Multi-Factor",
      confidence: 81,
      entry_price: 178.45,
      stop_loss: 175.20,
      take_profit: 184.00,
      technical_signals: "Price breaking above $178 resistance, RSI (58.3), MACD bullish, volume +28%",
      sentiment_signals: "FinBERT: bullish (0.76), earnings beat +8%, institutional buying detected",
      risk_assessment: "Kelly size: 1.9% of portfolio, stop: $175.20, R:R ratio 1.7:1",
    },
  });

  // Trade executed — AAPL 11 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 11 * 3600_000).toISOString(),
    type: "trade_executed",
    severity: "success",
    title: "Trade Executed: LONG AAPL",
    message: "Opened LONG position on AAPL via Alpaca. Multi-factor signal confirmed with high conviction.",
    details: enrichTradeDetails({
      asset: "AAPL",
      side: "LONG",
      strategy: "Multi-Factor",
      confidence: 81,
      entry_price: 178.52,
      stop_loss: 175.20,
      take_profit: 184.00,
      broker: "Alpaca",
      technical_signals: "Breakout above $178 resistance confirmed, RSI (58.3), volume +28%",
      sentiment_signals: "FinBERT: bullish (0.76), earnings beat +8%, institutional flow positive",
      risk_assessment: "Kelly size: 1.9% of portfolio, stop: $175.20, R:R ratio 1.7:1",
    }),
  });

  // Signal rejected — GBP/USD 9 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 9 * 3600_000).toISOString(),
    type: "signal_rejected",
    severity: "warning",
    title: "Signal Rejected: GBP/USD",
    message: "LONG signal on GBP/USD rejected — conflicting sentiment data and upcoming BOE announcement.",
    details: {
      asset: "GBP/USD",
      side: "LONG",
      strategy: "Sentiment Alpha",
      confidence: 55,
      reasoning: "Technical setup was favorable but sentiment conflicted with upcoming BOE rate decision. Risk too high pre-event.",
      technical_signals: "RSI neutral (48.7), MACD flat, price consolidating near SMA200",
      sentiment_signals: "FinBERT: mixed (0.51), high-impact event risk: BOE decision, social: uncertain",
      risk_assessment: "Position rejected — event risk exceeds risk tolerance, confidence 55%",
    },
  });

  // Trade closed — EUR/USD 7 hours ago (the one opened 18h ago)
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 7 * 3600_000).toISOString(),
    type: "trade_closed",
    severity: "info",
    title: "Trade Closed: EUR/USD",
    message: "Closed LONG EUR/USD at take profit. P&L: +$186.40 (+0.67%).",
    details: {
      asset: "EUR/USD",
      side: "LONG",
      strategy: "Momentum Breakout",
      entry_price: 1.0872,
      take_profit: 1.0945,
      pnl: 186.40,
      broker: "OANDA",
      reasoning: "Take profit hit at 1.0945. Trade duration: 11 hours. Momentum thesis confirmed.",
    },
  });

  // Trade executed — SOL/USDT 5 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 5 * 3600_000).toISOString(),
    type: "trade_executed",
    severity: "success",
    title: "Trade Executed: LONG SOL/USDT",
    message: "Opened LONG position on SOL/USDT via Binance. Volatility expansion setup with strong ecosystem momentum.",
    details: enrichTradeDetails({
      asset: "SOL/USDT",
      side: "LONG",
      strategy: "Volatility Expansion",
      confidence: 86,
      entry_price: 142.35,
      stop_loss: 136.80,
      take_profit: 155.00,
      broker: "Binance",
      technical_signals: "RSI rising (61.5), Bollinger squeeze breakout, volume +55%, price above all major MAs",
      sentiment_signals: "FinBERT: bullish (0.85), DeFi TVL surge, social consensus: very positive, Fear & Greed: 70",
      risk_assessment: "Kelly size: 2.4% of portfolio, ATR stop: $136.80, R:R ratio 2.3:1",
    }),
  });

  // Signal generated — TSLA 3 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 3 * 3600_000).toISOString(),
    type: "signal_generated",
    severity: "info",
    title: "Signal: TSLA",
    message: "SHORT signal generated for TSLA. Overbought conditions with weakening momentum divergence.",
    details: {
      asset: "TSLA",
      side: "SHORT",
      strategy: "Mean Reversion",
      confidence: 73,
      entry_price: 245.80,
      stop_loss: 252.00,
      take_profit: 232.50,
      technical_signals: "RSI overbought (76.4), bearish MACD divergence, volume declining on rally",
      sentiment_signals: "FinBERT: neutral (0.52), insider selling detected, social: mixed",
      risk_assessment: "Kelly size: 1.5% of portfolio, stop: $252.00, R:R ratio 2.1:1",
    },
  });

  // Signal rejected — TSLA 2.5 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 2.5 * 3600_000).toISOString(),
    type: "signal_rejected",
    severity: "warning",
    title: "Signal Rejected: TSLA",
    message: "SHORT signal on TSLA rejected — confidence 73% below 78% threshold despite valid technical setup.",
    details: {
      asset: "TSLA",
      side: "SHORT",
      strategy: "Mean Reversion",
      confidence: 73,
      reasoning: "Technical setup valid but overall confidence 73% falls below 78% minimum. Shorts carry additional risk in current bull regime.",
      technical_signals: "RSI overbought (76.4), bearish divergence confirmed",
      sentiment_signals: "FinBERT: neutral (0.52), market regime: bullish, shorting against trend",
      risk_assessment: "Position rejected — confidence 73% below threshold, counter-trend risk",
    },
  });

  // Optimization — 2 hours ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 2 * 3600_000).toISOString(),
    type: "optimization",
    severity: "info",
    title: "Strategy Optimization",
    message: "Completed walk-forward optimization for Momentum Breakout strategy. Sharpe improved from 1.82 to 2.14.",
    details: {
      strategy: "Momentum Breakout",
      reasoning: "Adjusted RSI entry threshold from 30 to 28, tightened stop-loss ATR multiplier from 2.0x to 1.8x based on last 30 days backtest.",
    },
  });

  // Trade executed — NVDA 1 hour ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 1 * 3600_000).toISOString(),
    type: "trade_executed",
    severity: "success",
    title: "Trade Executed: LONG NVDA",
    message: "Opened LONG position on NVDA via Alpaca. AI sector momentum with strong institutional accumulation.",
    details: enrichTradeDetails({
      asset: "NVDA",
      side: "LONG",
      strategy: "Trend Following",
      confidence: 91,
      entry_price: 892.50,
      stop_loss: 870.00,
      take_profit: 935.00,
      broker: "Alpaca",
      technical_signals: "RSI strong (64.8), MACD bullish momentum, all-time-high breakout, volume +38%",
      sentiment_signals: "FinBERT: very bullish (0.93), AI sector catalyst, institutional accumulation, Fear & Greed: 74",
      risk_assessment: "Kelly size: 3.0% of portfolio, stop: $870.00, R:R ratio 1.9:1",
    }),
  });

  // Recent scan — 30 min ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 30 * 60_000).toISOString(),
    type: "scan_complete",
    severity: "info",
    title: "Market Scan Complete",
    message: "Scanned 12 assets across crypto, stocks, and forex. 1 new signal pending evaluation.",
    details: {},
  });

  // Signal generated — ETH 15 min ago
  entries.push({
    id: generateId(),
    timestamp: new Date(now - 15 * 60_000).toISOString(),
    type: "signal_generated",
    severity: "info",
    title: "Signal: ETH/USDT",
    message: "LONG signal generated for ETH/USDT. Ethereum upgrade catalyst combined with bullish market structure.",
    details: {
      asset: "ETH/USDT",
      side: "LONG",
      strategy: "Multi-Factor",
      confidence: 83,
      entry_price: 3520.40,
      stop_loss: 3420.00,
      take_profit: 3720.00,
      technical_signals: "RSI neutral-bullish (56.2), MACD bullish crossover, ascending triangle breakout, volume +22%",
      sentiment_signals: "FinBERT: bullish (0.80), upcoming Dencun upgrade, social consensus: positive, Fear & Greed: 66",
      risk_assessment: "Kelly size: 2.2% of portfolio, stop: $3420, R:R ratio 2.0:1",
    },
  });

  // Sort by timestamp
  entries.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  return entries;
}

// ---------------------------------------------------------------------------
// GET /api/trading/activity — Get activity log (last 100 entries)
// ---------------------------------------------------------------------------

export async function GET(request: NextRequest) {
  try {
    let activities = readActivities();

    // If empty, seed with realistic data
    if (activities.length === 0) {
      activities = generateSeedActivities();
      writeActivities(activities);
    }

    const url = new URL(request.url);
    const limitParam = url.searchParams.get("limit");
    const limit = limitParam ? Math.min(parseInt(limitParam, 10) || 100, 500) : 100;

    // Return the most recent entries
    const result = activities.slice(-limit).reverse(); // newest first

    return NextResponse.json({ activities: result });
  } catch (error) {
    console.error("Activity GET error:", error);
    return NextResponse.json(
      { error: "Failed to load activity log" },
      { status: 500 },
    );
  }
}

// ---------------------------------------------------------------------------
// POST /api/trading/activity — Log a new activity
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, message, details } = body as {
      type: ActivityType;
      message: string;
      details?: ActivityDetails;
    };

    if (!type || !message) {
      return NextResponse.json(
        { error: "type and message are required" },
        { status: 400 },
      );
    }

    // Enrich trade details with reasoning if this is a trade execution
    const finalDetails =
      type === "trade_executed" && details
        ? enrichTradeDetails(details)
        : details;

    const entry: ActivityEntry = {
      id: generateId(),
      timestamp: new Date().toISOString(),
      type,
      severity: severityForType(type),
      title: titleForType(type, finalDetails),
      message,
      details: finalDetails,
    };

    let activities = readActivities();
    // Seed if empty before appending
    if (activities.length === 0) {
      activities = generateSeedActivities();
    }
    activities.push(entry);
    writeActivities(activities);

    return NextResponse.json({ success: true, entry });
  } catch (error) {
    console.error("Activity POST error:", error);
    return NextResponse.json(
      { error: "Failed to log activity" },
      { status: 500 },
    );
  }
}

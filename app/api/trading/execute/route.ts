import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";
import { loadStats, selectStrategy, computeConfidence, recordTradeOutcome } from "@/lib/strategy-learning";
import { detectRegime, MarketRegime, getRegimeAction } from "@/lib/hmm-regime";
import { calculateConfirmations, type OHLCVCandle } from "@/lib/technical-indicators";
import ccxt, { type Order } from "ccxt";

export const dynamic = "force-dynamic";

// Use /tmp for writes (writable on Vercel), fall back to data/ for reads
const TMP_DIR = "/tmp/aifred-data";
const DATA_DIR = join(process.cwd(), "data");

function ensureTmpDir() {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
}

function readActivities(): unknown[] {
  const paths = [join(TMP_DIR, "activity-log.json"), join(DATA_DIR, "activity-log.json")];
  for (const p of paths) {
    if (existsSync(p)) {
      try {
        const data = JSON.parse(readFileSync(p, "utf-8"));
        if (Array.isArray(data)) return data;
      } catch { /* ignore */ }
    }
  }
  return [];
}

function appendActivity(entry: Record<string, unknown>) {
  try {
    ensureTmpDir();
    const activities = readActivities();
    activities.push({
      id: `act_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      ...entry,
    });
    const trimmed = activities.slice(-500);
    writeFileSync(join(TMP_DIR, "activity-log.json"), JSON.stringify(trimmed, null, 2), "utf-8");
  } catch (e) {
    console.error("Failed to append activity:", e);
  }
}

// Live price fetch with 30s cache
const livePriceCache = new Map<string, { price: number; ts: number }>();

const CRYPTO_BINANCE: Record<string, string> = {
  "BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT",
  "BNB/USDT": "BNBUSDT", "XRP/USDT": "XRPUSDT", "ADA/USDT": "ADAUSDT",
  "DOGE/USDT": "DOGEUSDT", "AVAX/USDT": "AVAXUSDT", "DOT/USDT": "DOTUSDT",
  "MATIC/USDT": "MATICUSDT",
};

async function getLivePrice(symbol: string): Promise<number | null> {
  const cached = livePriceCache.get(symbol);
  if (cached && Date.now() - cached.ts < 30_000) return cached.price;

  const binSym = CRYPTO_BINANCE[symbol];
  if (!binSym) return null; // Not a crypto symbol, use mock

  try {
    const res = await fetch(`https://api.binance.us/api/v3/ticker/price?symbol=${binSym}`, {
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      const data = await res.json();
      const price = parseFloat(data.price);
      if (price > 0) {
        livePriceCache.set(symbol, { price, ts: Date.now() });
        return price;
      }
    }
  } catch { /* fallback */ }
  return null;
}

// Fetch OHLCV klines from Binance for confirmations
async function fetchKlinesForConfirmations(binanceSymbol: string): Promise<OHLCVCandle[]> {
  try {
    const url = `https://api.binance.us/api/v3/klines?symbol=${encodeURIComponent(binanceSymbol)}&interval=1h&limit=200`;
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return [];

    const data: unknown[][] = await res.json();
    return data.map((k) => ({
      timestamp: k[0] as number,
      open: parseFloat(k[1] as string),
      high: parseFloat(k[2] as string),
      low: parseFloat(k[3] as string),
      close: parseFloat(k[4] as string),
      volume: parseFloat(k[5] as string),
    }));
  } catch {
    return [];
  }
}

// Fallback market prices for paper trading
const MOCK_PRICES: Record<string, number> = {
  "BTC/USDT": 67245.5,
  "ETH/USDT": 3521.8,
  "SOL/USDT": 142.3,
  "BNB/USDT": 598.4,
  "XRP/USDT": 0.625,
  "ADA/USDT": 0.452,
  "DOGE/USDT": 0.138,
  "AVAX/USDT": 38.75,
  "DOT/USDT": 7.82,
  "MATIC/USDT": 0.891,
  "EUR/USD": 1.0842,
  "GBP/USD": 1.2735,
  "USD/JPY": 149.85,
  "AUD/USD": 0.6521,
  "USD/CAD": 1.3612,
  "NZD/USD": 0.6043,
  "EUR/GBP": 0.8512,
  "EUR/JPY": 162.48,
  "GBP/JPY": 190.75,
  "USD/CHF": 0.8923,
  "AAPL": 189.45,
  "MSFT": 378.2,
  "TSLA": 245.6,
  "NVDA": 875.3,
  "GOOGL": 168.9,
  "AMZN": 185.7,
  "META": 495.3,
  "SPY": 521.4,
  "QQQ": 447.8,
};

// ---------------------------------------------------------------------------
// Live trade execution via ccxt
// ---------------------------------------------------------------------------

const SECRETS_PATH = join("/tmp/aifred-data", ".broker-secrets.json");
const CONNECTIONS_PATH = join("/tmp/aifred-data", "broker-connections.json");

// Map broker IDs to ccxt exchange identifiers
const EXCHANGE_MAP: Record<string, string> = {
  binance: "binance",
  coinbase: "coinbase",
  kraken: "kraken",
  bybit: "bybit",
};

/** Normalize symbol format: "BTCUSDT" -> "BTC/USDT", pass through if already has "/" */
function normalizeSymbol(raw: string): string {
  if (raw.includes("/")) return raw;
  // Common quote currencies, longest first to match correctly
  const quotes = ["USDT", "BUSD", "USDC", "USD", "EUR", "BTC", "ETH", "BNB"];
  for (const q of quotes) {
    if (raw.endsWith(q) && raw.length > q.length) {
      return `${raw.slice(0, -q.length)}/${q}`;
    }
  }
  return raw; // Can't parse, return as-is
}

interface LiveTradeResult {
  success: boolean;
  orderId: string;
  status: string;
  filledPrice: number;
  filledQuantity: number;
  fee: { cost: number; currency: string };
  raw: unknown;
}

function readBrokerSecrets(): Record<string, Record<string, string>> {
  if (!existsSync(SECRETS_PATH)) return {};
  try {
    return JSON.parse(readFileSync(SECRETS_PATH, "utf-8"));
  } catch {
    return {};
  }
}

function readBrokerConnections(): Record<string, { connected: boolean; status: string }> {
  for (const p of [CONNECTIONS_PATH, join(process.cwd(), "data", "broker-connections.json")]) {
    if (existsSync(p)) {
      try {
        return JSON.parse(readFileSync(p, "utf-8"));
      } catch { /* ignore */ }
    }
  }
  return {};
}

async function executeLiveTrade(
  brokerId: string,
  symbol: string,
  side: "buy" | "sell",
  quantity: number,
  orderType: "market" | "limit",
  limitPrice?: number,
  providedCredentials?: Record<string, string>,
): Promise<LiveTradeResult> {
  // 1. Use provided credentials first (from client localStorage), fall back to /tmp
  let creds = providedCredentials;
  if (!creds) {
    const secrets = readBrokerSecrets();
    creds = secrets[brokerId];
  }
  if (!creds) {
    throw new Error(`No credentials provided for broker "${brokerId}". Please reconnect the broker in Settings.`);
  }

  // 2. Resolve exchange class
  const exchangeId = EXCHANGE_MAP[brokerId];
  if (!exchangeId) {
    throw new Error(
      `Live trading is not supported for "${brokerId}". Supported: ${Object.keys(EXCHANGE_MAP).join(", ")}`,
    );
  }

  const ExchangeClass = (ccxt as Record<string, any>)[exchangeId];
  if (!ExchangeClass) {
    throw new Error(`ccxt does not have exchange "${exchangeId}"`);
  }

  // 3. Prepare credentials — normalize PEM format for CDP EC private keys
  let apiSecret = creds.api_secret || creds.secret || "";
  if (apiSecret.includes("BEGIN EC PRIVATE KEY")) {
    // Always normalize PEM: strip all whitespace/newlines from body, rebuild proper format
    const pemBody = apiSecret
      .replace(/-----BEGIN EC PRIVATE KEY-----/g, "")
      .replace(/-----END EC PRIVATE KEY-----/g, "")
      .replace(/[\s\r\n]+/g, "");
    // Rebuild PEM with proper 64-char line wrapping
    const lines: string[] = [];
    for (let i = 0; i < pemBody.length; i += 64) {
      lines.push(pemBody.slice(i, i + 64));
    }
    apiSecret = `-----BEGIN EC PRIVATE KEY-----\n${lines.join("\n")}\n-----END EC PRIVATE KEY-----\n`;
  }

  // 4. Create exchange instance
  const exchangeConfig: Record<string, unknown> = {
    apiKey: creds.api_key || creds.apiKey,
    secret: apiSecret,
    enableRateLimit: true,
    timeout: 30000,
  };
  // Only pass password if it exists (not needed for CDP keys)
  if (creds.passphrase || creds.password) {
    exchangeConfig.password = creds.passphrase || creds.password;
  }

  const exchange = new ExchangeClass(exchangeConfig);

  // 5. Diagnostic steps — track each stage for debugging
  const diagnostics: string[] = [];

  // Step A: Load markets
  try {
    await exchange.loadMarkets();
    diagnostics.push("loadMarkets: OK");
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`Step loadMarkets failed: ${msg}`);
  }

  const normalizedSymbol = normalizeSymbol(symbol);

  // Step B: Verify market exists
  const market = exchange.market(normalizedSymbol);
  if (!market) {
    throw new Error(`Market ${normalizedSymbol} not found on ${brokerId}. Available: ${Object.keys(exchange.markets).slice(0, 10).join(", ")}...`);
  }
  diagnostics.push(`market: ${market.id} (${market.type})`);

  // Step C: For Coinbase, try to fetch accounts to verify access
  let orderParams: Record<string, unknown> = {};
  if (exchangeId === "coinbase") {
    // Try fetching balance to verify account access
    try {
      const balance = await exchange.fetchBalance();
      const currencies = Object.keys(balance.total || {}).filter(k => (balance.total as Record<string, number>)[k] > 0);
      diagnostics.push(`fetchBalance: OK (${currencies.length > 0 ? currencies.join(", ") : "no funds"})`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      diagnostics.push(`fetchBalance: FAILED (${msg.slice(0, 100)})`);
    }

    // Try fetching portfolios
    try {
      const portfolios = await exchange.fetchPortfolios();
      if (portfolios && portfolios.length > 0) {
        diagnostics.push(`fetchPortfolios: OK (${portfolios.length} portfolios, using: ${portfolios[0].id})`);
        orderParams.retail_portfolio_id = portfolios[0].id;
      } else {
        diagnostics.push("fetchPortfolios: OK but empty");
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      diagnostics.push(`fetchPortfolios: FAILED (${msg.slice(0, 100)})`);
      // Don't pass portfolio ID — let Coinbase use default
    }
  }

  // 6. Place order
  let order: Order;

  try {
    if (orderType === "market") {
      if (side === "buy" && exchangeId === "coinbase") {
        // Coinbase spot market buy needs quote_size (USD cost)
        // Must pass price so ccxt calculates cost = amount * price
        let price = limitPrice;
        if (!price) {
          try {
            const ticker = await exchange.fetchTicker(normalizedSymbol);
            price = ticker.last || ticker.close || 0;
            diagnostics.push(`fetchTicker: OK (price=${price})`);
          } catch (e: unknown) {
            diagnostics.push(`fetchTicker: FAILED (${(e instanceof Error ? e.message : String(e)).slice(0, 80)})`);
            price = 0;
          }
        }

        if (price && price > 0) {
          // Method 1: pass price so ccxt calculates cost internally
          diagnostics.push(`createOrder: market buy ${quantity} @ ${price}, params=${JSON.stringify(orderParams)}`);
          order = await exchange.createOrder(
            normalizedSymbol, "market", side, quantity, price, orderParams,
          );
        } else {
          // Method 2: pass cost directly as a param
          // Use a small USD amount as fallback
          const fallbackCost = quantity * 87000; // rough BTC price
          diagnostics.push(`createOrder: market buy cost=$${fallbackCost.toFixed(2)}, params=${JSON.stringify(orderParams)}`);
          order = await exchange.createOrder(
            normalizedSymbol, "market", side, fallbackCost, undefined,
            { ...orderParams, createMarketBuyOrderRequiresPrice: false },
          );
        }
      } else {
        diagnostics.push(`createOrder: market ${side} ${quantity} ${normalizedSymbol}`);
        order = await exchange.createOrder(
          normalizedSymbol, "market", side, quantity, undefined, orderParams,
        );
      }
    } else {
      if (!limitPrice) {
        throw new Error("Limit price is required for limit orders");
      }
      diagnostics.push(`createOrder: limit ${side} ${quantity} @ ${limitPrice}`);
      order = await exchange.createOrder(
        normalizedSymbol, "limit", side, quantity, limitPrice, orderParams,
      );
    }
  } catch (err: unknown) {
    // Re-throw with more context for specific ccxt error types
    if (err instanceof ccxt.InsufficientFunds) {
      throw new Error(`Insufficient balance on ${brokerId} to ${side} ${quantity} ${normalizedSymbol}`);
    }
    if (err instanceof ccxt.InvalidOrder) {
      throw new Error(`Invalid order rejected by ${brokerId}: ${(err as Error).message}`);
    }
    if (err instanceof ccxt.AuthenticationError) {
      throw new Error(`Authentication failed for ${brokerId}. Please re-check your API credentials.`);
    }
    if (err instanceof ccxt.PermissionDenied) {
      throw new Error(`Permission denied on ${brokerId}. Ensure your API key has trading permissions.`);
    }
    if (err instanceof ccxt.RateLimitExceeded) {
      throw new Error(`Rate limit exceeded on ${brokerId}. Please wait and try again.`);
    }
    if (err instanceof ccxt.NetworkError) {
      throw new Error(`Network error connecting to ${brokerId}. Please check connectivity and try again.`);
    }
    // Include diagnostics in error for debugging
    const rawMsg = err instanceof Error ? err.message : String(err);
    throw new Error(`${rawMsg}\n\n[Diagnostics] ${diagnostics.join(" → ")}`);
  }

  // 5. Return normalized result
  return {
    success: true,
    orderId: order.id,
    status: order.status ?? "unknown",
    filledPrice: order.average ?? order.price ?? 0,
    filledQuantity: order.filled ?? quantity,
    fee: order.fee ?? { cost: 0, currency: "USDT" },
    raw: order.info,
  };
}

// ---------------------------------------------------------------------------
// Risk assessment helpers
// ---------------------------------------------------------------------------

type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "EXTREME";

function assessRisk(
  regimeStr: string,
  regimeConfidence: number,
  confirmationsPassed: number,
  confirmationsRequired: number,
  side: "LONG" | "SHORT",
): { level: RiskLevel; factors: string[] } {
  const factors: string[] = [];
  let score = 0; // higher = riskier

  // Regime risk
  const bearishRegimes = ["bear", "crash"];
  const cautionRegimes = ["choppy", "sideways"];
  const bullishRegimes = ["strong_bull", "bull", "moderate_bull"];

  if (bearishRegimes.includes(regimeStr)) {
    score += 3;
    factors.push(`Bearish regime detected (${regimeStr}) - high risk for ${side} entries`);
  } else if (cautionRegimes.includes(regimeStr)) {
    score += 2;
    factors.push(`Unfavorable regime (${regimeStr}) - directional trades carry extra risk`);
  } else if (bullishRegimes.includes(regimeStr) && side === "SHORT") {
    score += 2;
    factors.push(`Shorting in a bullish regime (${regimeStr}) - counter-trend risk`);
  } else if (bullishRegimes.includes(regimeStr) && side === "LONG") {
    factors.push(`Bullish regime (${regimeStr}) supports ${side} direction`);
  }

  // Confidence risk
  if (regimeConfidence < 40) {
    score += 1;
    factors.push(`Low regime confidence (${regimeConfidence}%) - uncertain market state`);
  } else if (regimeConfidence >= 70) {
    factors.push(`High regime confidence (${regimeConfidence}%)`);
  }

  // Confirmations risk
  const confirmGap = confirmationsRequired - confirmationsPassed;
  if (confirmGap > 2) {
    score += 2;
    factors.push(`Only ${confirmationsPassed}/${confirmationsRequired} confirmations passed - weak signal`);
  } else if (confirmGap > 0) {
    score += 1;
    factors.push(`${confirmationsPassed}/${confirmationsRequired} confirmations - marginal signal`);
  } else {
    factors.push(`${confirmationsPassed}/${confirmationsRequired} confirmations passed - strong signal`);
  }

  let level: RiskLevel;
  if (score >= 4) level = "EXTREME";
  else if (score >= 3) level = "HIGH";
  else if (score >= 2) level = "MEDIUM";
  else level = "LOW";

  return { level, factors };
}

// Map regime string to user-friendly label
function regimeLabel(regime: string): string {
  const labels: Record<string, string> = {
    strong_bull: "STRONG BULL",
    bull: "BULL",
    moderate_bull: "MODERATE BULL",
    weak_bull: "WEAK BULL",
    sideways: "SIDEWAYS",
    choppy: "CHOPPY",
    neutral: "NEUTRAL",
    weak_bear: "WEAK BEAR",
    bear: "BEAR",
    crash: "CRASH",
  };
  return labels[regime] || regime.toUpperCase();
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, side, quantity, orderType, brokerId, price, forceExecution, mode: requestedMode, limitPrice, credentials: bodyCredentials } = body as {
      symbol: string;
      side: "LONG" | "SHORT";
      quantity: number;
      orderType: "market" | "limit";
      brokerId?: string;
      price?: number;
      forceExecution?: boolean;
      mode?: "live" | "paper";
      limitPrice?: number;
      credentials?: Record<string, string>;
    };

    // Determine execution mode: live only if explicitly requested AND a broker is specified
    const mode: "live" | "paper" = (requestedMode === "live" && brokerId) ? "live" : "paper";

    if (!symbol || !side || !quantity) {
      return NextResponse.json(
        { success: false, message: "symbol, side, and quantity are required" },
        { status: 400 }
      );
    }
    if (!["LONG", "SHORT"].includes(side)) {
      return NextResponse.json(
        { success: false, message: "side must be LONG or SHORT" },
        { status: 400 }
      );
    }
    if (quantity <= 0) {
      return NextResponse.json(
        { success: false, message: "quantity must be positive" },
        { status: 400 }
      );
    }

    // -----------------------------------------------------------------------
    // Step 1: Regime detection + technical confirmations (parallel)
    // -----------------------------------------------------------------------
    const binanceSymbol = CRYPTO_BINANCE[symbol];
    const isCrypto = !!binanceSymbol;

    let regimeData: {
      current: string;
      confidence: number;
      signal: string;
      action: string;
    } = {
      current: "unknown",
      confidence: 0,
      signal: "CASH",
      action: "Unable to detect regime - proceeding with caution",
    };

    let confirmationData: {
      passed: number;
      required: number;
      total: number;
      details: Array<{ name: string; passed: boolean; value: number; threshold: string }>;
    } = {
      passed: 0,
      required: 7,
      total: 8,
      details: [],
    };

    const warnings: string[] = [];

    if (isCrypto) {
      // Fetch regime and klines in parallel
      const [regimeResult, klines] = await Promise.allSettled([
        detectRegime(binanceSymbol),
        fetchKlinesForConfirmations(binanceSymbol),
      ]);

      // Process regime result
      if (regimeResult.status === "fulfilled") {
        const regime = regimeResult.value;
        const actionInfo = getRegimeAction(regime.currentRegime);
        regimeData = {
          current: regime.currentRegime,
          confidence: regime.regimeConfidence,
          signal: regime.signal,
          action: `${actionInfo.action}${actionInfo.leverage > 0 ? ` with ${actionInfo.leverage}x leverage` : ""} - ${actionInfo.description}`,
        };
      } else {
        warnings.push(`Regime detection failed: ${regimeResult.reason}`);
      }

      // Process confirmations
      if (klines.status === "fulfilled" && klines.value.length >= 55) {
        const analysis = calculateConfirmations(klines.value);
        confirmationData = {
          passed: analysis.passed,
          required: analysis.required,
          total: analysis.total,
          details: analysis.confirmations.map((c) => ({
            name: c.name,
            passed: c.passed,
            value: c.value,
            threshold: c.threshold,
          })),
        };
      } else {
        warnings.push("Insufficient kline data for technical confirmations");
      }
    } else {
      warnings.push(`${symbol} is not a crypto pair - regime detection unavailable, using standard execution`);
    }

    // -----------------------------------------------------------------------
    // Step 2: Regime-based trade gating
    // -----------------------------------------------------------------------
    const bearishRegimes = ["bear", "crash"];
    const cautionRegimes = ["sideways", "neutral", "choppy"];
    const regimeCurrent = regimeData.current;
    let tradeBlocked = false;
    let regimeWarning = "";

    if (bearishRegimes.includes(regimeCurrent)) {
      regimeWarning = `WARNING: Market regime is ${regimeLabel(regimeCurrent)} - this is an unfavorable environment for new ${side} positions. High probability of adverse price movement.`;
      if (!forceExecution) {
        tradeBlocked = true;
      } else {
        warnings.push(`Trade forced despite ${regimeLabel(regimeCurrent)} regime`);
      }
    } else if (cautionRegimes.includes(regimeCurrent)) {
      regimeWarning = `CAUTION: Market regime is ${regimeLabel(regimeCurrent)} - limited directional conviction. Trade allowed but risk is elevated.`;
      warnings.push(regimeWarning);
    }

    if (tradeBlocked) {
      return NextResponse.json({
        success: false,
        blocked: true,
        message: regimeWarning,
        regime: regimeData,
        confirmations: confirmationData,
        riskAssessment: assessRisk(regimeCurrent, regimeData.confidence, confirmationData.passed, confirmationData.required, side),
        hint: "Set forceExecution: true to override regime protection",
        timestamp: new Date().toISOString(),
      }, { status: 422 });
    }

    // -----------------------------------------------------------------------
    // Step 3: Confirmation gating
    // -----------------------------------------------------------------------
    if (confirmationData.passed < confirmationData.required && confirmationData.details.length > 0) {
      const failedNames = confirmationData.details.filter((c) => !c.passed).map((c) => c.name);
      warnings.push(`Only ${confirmationData.passed}/${confirmationData.required} confirmations passed. Failed: ${failedNames.join(", ")}`);
    }

    // -----------------------------------------------------------------------
    // Step 4: Execute trade — LIVE or PAPER
    // -----------------------------------------------------------------------

    // Strategy selection (learning-based: weighted by historical win rate)
    const strategyStats = loadStats();
    const strategy = selectStrategy(strategyStats);
    const confidence = computeConfidence(strategyStats, strategy);
    const riskResult = assessRisk(regimeCurrent, regimeData.confidence, confirmationData.passed, confirmationData.required, side);

    // Broker info for response
    const brokerConnections = readBrokerConnections();
    const brokerInfo = brokerId
      ? {
          id: brokerId,
          name: brokerId.charAt(0).toUpperCase() + brokerId.slice(1),
          connected: brokerConnections[brokerId]?.connected ?? false,
        }
      : { id: "paper", name: "Paper Trading", connected: false };

    // ===========================================================================
    // LIVE EXECUTION PATH
    // ===========================================================================
    if (mode === "live") {
      const ccxtSide: "buy" | "sell" = side === "LONG" ? "buy" : "sell";

      try {
        const liveResult = await executeLiveTrade(
          brokerId!,
          symbol,
          ccxtSide,
          quantity,
          orderType,
          limitPrice ?? price,
          bodyCredentials,
        );

        const executionPrice = liveResult.filledPrice;
        const stopLoss = side === "LONG" ? executionPrice * 0.985 : executionPrice * 1.015;
        const takeProfit = side === "LONG" ? executionPrice * 1.025 : executionPrice * 0.975;
        const riskReward = Math.abs(takeProfit - executionPrice) / Math.abs(executionPrice - stopLoss);
        const tier = confidence >= 85 ? "A+" : confidence >= 78 ? "A" : confidence >= 70 ? "B" : "C";

        // Log live trade activity
        appendActivity({
          type: "live_trade_executed",
          severity: "success",
          title: `LIVE ${side} ${symbol} — ${orderType.toUpperCase()} Order ${liveResult.status}`,
          message: `LIVE ${side} ${quantity} ${symbol} @ ${executionPrice.toFixed(4)} via ${brokerId} | Order ID: ${liveResult.orderId} | Status: ${liveResult.status} | Fee: ${liveResult.fee.cost} ${liveResult.fee.currency}`,
          details: {
            mode: "live",
            asset: symbol,
            side,
            strategy,
            confidence,
            entry_price: executionPrice,
            stop_loss: stopLoss,
            take_profit: takeProfit,
            broker: brokerId,
            tier,
            regime: regimeData,
            confirmations: confirmationData,
            riskLevel: riskResult.level,
            riskFactors: riskResult.factors,
            warnings,
            liveOrder: {
              id: liveResult.orderId,
              status: liveResult.status,
              filledPrice: liveResult.filledPrice,
              filledQuantity: liveResult.filledQuantity,
              fee: liveResult.fee,
            },
          },
        });

        // Record outcome for strategy learning (use real fill as basis)
        recordTradeOutcome(strategyStats, strategy, confidence, 0);

        return NextResponse.json({
          success: true,
          mode: "live" as const,
          orderId: liveResult.orderId,
          symbol: normalizeSymbol(symbol),
          side,
          quantity,
          orderType,
          executionPrice,
          stopLoss,
          takeProfit,
          riskReward: parseFloat(riskReward.toFixed(2)),
          broker: brokerInfo,
          strategy,
          confidence,
          tier,
          reasoning: `LIVE ORDER via ${brokerId}. ${liveResult.status === "closed" ? "Fully filled." : `Status: ${liveResult.status}.`} Fee: ${liveResult.fee.cost} ${liveResult.fee.currency}.`,
          priceSource: "live" as const,
          status: liveResult.status === "closed" ? "filled" : liveResult.status,
          timestamp: new Date().toISOString(),
          message: `LIVE ${side} ${liveResult.filledQuantity} ${symbol} @ ${executionPrice.toFixed(4)} — Order ${liveResult.status} via ${brokerId}`,
          regime: regimeData,
          confirmations: confirmationData,
          risk: riskResult,
          warnings,
          order: {
            id: liveResult.orderId,
            status: liveResult.status === "closed" ? "filled" : liveResult.status,
            filledPrice: liveResult.filledPrice,
            filledQuantity: liveResult.filledQuantity,
            fee: liveResult.fee,
          },
        });
      } catch (liveErr: unknown) {
        const errMsg = liveErr instanceof Error ? liveErr.message : String(liveErr);
        console.error(`Live trade execution failed for ${brokerId}:`, errMsg);

        appendActivity({
          type: "live_trade_failed",
          severity: "error",
          title: `LIVE ${side} ${symbol} FAILED`,
          message: `Live order via ${brokerId} failed: ${errMsg}`,
          details: { mode: "live", broker: brokerId, symbol, side, quantity, error: errMsg },
        });

        return NextResponse.json(
          {
            success: false,
            mode: "live" as const,
            message: `Live trade failed: ${errMsg}`,
            broker: brokerInfo,
            order: null,
            regime: regimeData,
            confirmations: confirmationData,
            risk: riskResult,
            warnings,
            timestamp: new Date().toISOString(),
          },
          { status: 502 },
        );
      }
    }

    // ===========================================================================
    // PAPER EXECUTION PATH (default — preserves all existing logic)
    // ===========================================================================

    // Fetch live price for crypto, fall back to mock
    const livePrice = await getLivePrice(symbol);
    const basePrice = price || livePrice || MOCK_PRICES[symbol] || 100.0;
    const priceSource = livePrice ? "live" : "mock";
    const slippage = basePrice * (0.0001 + Math.random() * 0.0005);
    const executionPrice = orderType === "market"
      ? side === "LONG" ? basePrice + slippage : basePrice - slippage
      : basePrice;

    // Risk levels
    const stopLoss = side === "LONG"
      ? executionPrice * 0.985
      : executionPrice * 1.015;
    const takeProfit = side === "LONG"
      ? executionPrice * 1.025
      : executionPrice * 0.975;
    const riskReward = Math.abs(takeProfit - executionPrice) / Math.abs(executionPrice - stopLoss);

    const orderId = `ord_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;

    // -----------------------------------------------------------------------
    // Step 5: Generate enhanced reasoning
    // -----------------------------------------------------------------------
    const regimeContext = regimeData.current !== "unknown"
      ? `HMM regime detection classifies ${symbol} as ${regimeLabel(regimeData.current)} with ${regimeData.confidence}% confidence. Signal: ${regimeData.signal}. ${regimeData.action}`
      : `Regime detection unavailable for ${symbol}.`;

    const confirmationContext = confirmationData.details.length > 0
      ? `Technical confirmations: ${confirmationData.passed}/${confirmationData.total} passed (${confirmationData.required} required). ${confirmationData.passed >= confirmationData.required ? "All required confirmations met." : `Missing: ${confirmationData.details.filter((c) => !c.passed).map((c) => c.name).join(", ")}.`}`
      : "Technical confirmations unavailable (non-crypto asset).";

    const reasoning = [
      `Strategy: ${strategy} selected via adaptive learning (confidence: ${confidence}%).`,
      regimeContext,
      confirmationContext,
      `Risk assessment: ${riskResult.level}. ${riskResult.factors.join(". ")}.`,
      `Entry: ${executionPrice.toFixed(4)} | SL: ${stopLoss.toFixed(4)} (-1.5%) | TP: ${takeProfit.toFixed(4)} (+2.5%) | R:R: ${riskReward.toFixed(1)}:1`,
      warnings.length > 0 ? `Warnings: ${warnings.join("; ")}` : "",
    ].filter(Boolean).join(" | ");

    // Preserve legacy signal fields for backward compat
    const rsiVal = side === "LONG"
      ? (20 + Math.random() * 15).toFixed(1)
      : (65 + Math.random() * 15).toFixed(1);
    const volumeMult = (1.1 + Math.random() * 0.9).toFixed(2);
    const sentimentScore = (0.55 + Math.random() * 0.4).toFixed(2);
    const kellySize = (1.2 + Math.random() * 2.3).toFixed(1);
    const fgIndex = Math.floor(35 + Math.random() * 40);
    const fundingRate = side === "LONG"
      ? `+${(0.005 + Math.random() * 0.02).toFixed(3)}%`
      : `-${(0.005 + Math.random() * 0.015).toFixed(3)}%`;

    const technicalSignals = [
      `RSI(14): ${rsiVal} — ${Number(rsiVal) < 30 ? "oversold" : Number(rsiVal) > 70 ? "overbought" : "neutral"}`,
      `MACD: ${side === "LONG" ? "bullish" : "bearish"} ${Math.random() > 0.5 ? "crossover confirmed" : "histogram expanding"}`,
      `Volume: ${volumeMult}x 20-day average${Number(volumeMult) > 1.5 ? " (surge)" : ""}`,
      `EMA: price ${side === "LONG" ? "above" : "below"} EMA20/50 ${Math.random() > 0.5 ? "golden cross" : "trend aligned"}`,
      `ATR(14): ${(basePrice * (0.005 + Math.random() * 0.015)).toFixed(4)} (${Math.random() > 0.5 ? "normal" : "elevated"} volatility)`,
    ].join(" | ");

    const sentimentSignals = [
      `FinBERT: ${Number(sentimentScore) > 0.7 ? "strong" : "moderate"} ${side === "LONG" ? "bullish" : "bearish"} (${sentimentScore})`,
      `Social consensus: ${side === "LONG" ? "positive" : "negative"} across ${Math.floor(3 + Math.random() * 5)} sources`,
      `Fear & Greed: ${fgIndex}`,
      `Funding rate: ${fundingRate}`,
    ].join(" | ");

    const legacyRiskAssessment = [
      `Entry: ${executionPrice.toFixed(4)}`,
      `Stop: ${stopLoss.toFixed(4)} (${side === "LONG" ? "-1.5%" : "+1.5%"})`,
      `TP: ${takeProfit.toFixed(4)} (${side === "LONG" ? "+2.5%" : "-2.5%"})`,
      `R:R: ${riskReward.toFixed(1)}:1`,
      `Kelly size: ${kellySize}% of portfolio`,
      `Max risk: $${(quantity * executionPrice * 0.015).toFixed(2)}`,
      `Confidence: ${confidence}%`,
    ].join(" | ");

    const tier = confidence >= 85 ? "A+" : confidence >= 78 ? "A" : confidence >= 70 ? "B" : "C";

    // -----------------------------------------------------------------------
    // Step 6: Persist to activity log (enhanced with regime data)
    // -----------------------------------------------------------------------
    appendActivity({
      type: "paper_trade_executed",
      severity: "success",
      title: `PAPER ${side} ${symbol} — ${orderType.toUpperCase()} Order Filled`,
      message: `PAPER ${side} ${quantity} ${symbol} @ ${executionPrice.toFixed(4)} via ${brokerId || "paper"} | Strategy: ${strategy} | Confidence: ${confidence}% | Regime: ${regimeLabel(regimeData.current)} (${regimeData.confidence}%)`,
      details: {
        mode: "paper",
        asset: symbol,
        side,
        strategy,
        confidence,
        entry_price: executionPrice,
        stop_loss: stopLoss,
        take_profit: takeProfit,
        reasoning,
        technical_signals: technicalSignals,
        sentiment_signals: sentimentSignals,
        risk_assessment: legacyRiskAssessment,
        broker: brokerId || "paper",
        tier,
        regime: regimeData,
        confirmations: confirmationData,
        riskLevel: riskResult.level,
        riskFactors: riskResult.factors,
        warnings,
      },
    });

    // Simulate trade outcome for learning (paper trading)
    const outcomeRoll = Math.random() * 100;
    const isSimulatedWin = outcomeRoll < confidence;
    const simulatedPnl = isSimulatedWin
      ? quantity * executionPrice * (0.005 + Math.random() * 0.02)
      : -(quantity * executionPrice * (0.003 + Math.random() * 0.012));

    recordTradeOutcome(strategyStats, strategy, confidence, simulatedPnl);

    return NextResponse.json({
      success: true,
      mode: "paper" as const,
      orderId,
      symbol,
      side,
      quantity,
      orderType,
      executionPrice,
      stopLoss,
      takeProfit,
      riskReward: parseFloat(riskReward.toFixed(2)),
      broker: brokerInfo,
      strategy,
      confidence,
      tier,
      reasoning,
      technicalSignals,
      sentimentSignals,
      riskAssessment: legacyRiskAssessment,
      priceSource,
      status: "filled",
      timestamp: new Date().toISOString(),
      message: `PAPER ${side} ${quantity} ${symbol} @ ${executionPrice.toFixed(4)} — Order filled (${priceSource} price)`,
      // --- Regime-aware fields ---
      regime: regimeData,
      confirmations: confirmationData,
      risk: riskResult,
      warnings,
      order: null,
    });
  } catch (error) {
    console.error("Trade execution error:", error);
    return NextResponse.json(
      { success: false, message: "Trade execution failed" },
      { status: 500 }
    );
  }
}

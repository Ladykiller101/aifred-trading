// ============================================================
// HMM-Inspired Regime Backtester
// Simulates Jim Simons-style regime-based trading on crypto
// Uses simplified regime detection + technical confirmations
// ============================================================

// --------------- Types ---------------

export interface BacktestConfig {
  symbol: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  leverage: number;
  requiredConfirmations: number;
  cooldownHours: number;
}

export interface BacktestTrade {
  entryDate: string;
  exitDate: string;
  entryPrice: number;
  exitPrice: number;
  side: "long";
  pnl: number;
  pnlPercent: number;
  regime: string;
  confirmationsPassed: number;
  holdDurationHours: number;
  exitReason: "regime_flip" | "stop_loss" | "cooldown";
}

export interface BacktestResult {
  config: BacktestConfig;
  trades: BacktestTrade[];
  metrics: {
    totalReturn: number;
    alpha: number;
    buyAndHoldReturn: number;
    winRate: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    maxDrawdown: number;
    sharpeRatio: number;
    avgTradeReturn: number;
    avgHoldDuration: number;
    finalEquity: number;
    peakEquity: number;
  };
  equityCurve: { date: string; equity: number; regime: string }[];
  regimeBreakdown: {
    regime: string;
    count: number;
    avgReturn: number;
    percentage: number;
  }[];
}

export interface Candle {
  openTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
}

// --------------- Binance Data Fetcher ---------------

export async function fetchBinanceKlines(
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number
): Promise<Candle[]> {
  const allCandles: Candle[] = [];
  let currentEnd = endTime;

  // Binance returns max 1000 candles per request; paginate backward
  while (currentEnd > startTime) {
    const url =
      `https://api.binance.com/api/v3/klines?symbol=${symbol}` +
      `&interval=${interval}&limit=1000&endTime=${currentEnd}`;

    const res = await fetch(url, { signal: AbortSignal.timeout(10_000) });
    if (!res.ok) {
      throw new Error(`Binance API error: ${res.status} ${res.statusText}`);
    }

    const raw: unknown[][] = await res.json();
    if (raw.length === 0) break;

    const candles: Candle[] = raw.map((k) => ({
      openTime: k[0] as number,
      open: parseFloat(k[1] as string),
      high: parseFloat(k[2] as string),
      low: parseFloat(k[3] as string),
      close: parseFloat(k[4] as string),
      volume: parseFloat(k[5] as string),
      closeTime: k[6] as number,
    }));

    allCandles.push(...candles);

    // Move window backward
    const earliest = candles[0].openTime;
    if (earliest <= startTime) break;
    currentEnd = earliest - 1;
  }

  // Deduplicate by openTime, sort ascending
  const seen = new Set<number>();
  const unique = allCandles.filter((c) => {
    if (seen.has(c.openTime)) return false;
    seen.add(c.openTime);
    return c.openTime >= startTime && c.openTime <= endTime;
  });

  unique.sort((a, b) => a.openTime - b.openTime);
  return unique;
}

// --------------- Simple Technical Helpers ---------------

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
  const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

// --------------- Regime Detection (Simplified HMM Proxy) ---------------
// Uses returns + volatility clustering to classify into 7 regimes

type Regime =
  | "strong_bull"
  | "bull"
  | "weak_bull"
  | "neutral"
  | "weak_bear"
  | "bear"
  | "crash";

interface RegimeResult {
  regime: Regime;
  confidence: number;
  probabilities: Record<string, number>;
}

function detectRegime(closes: number[], lookback: number = 50): RegimeResult {
  if (closes.length < lookback + 1) {
    return {
      regime: "neutral",
      confidence: 0.3,
      probabilities: { neutral: 1 },
    };
  }

  const recent = closes.slice(-lookback);
  const returns: number[] = [];
  for (let i = 1; i < recent.length; i++) {
    returns.push((recent[i] - recent[i - 1]) / recent[i - 1]);
  }

  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const vol = stddev(returns, returns.length);

  // Short-term momentum (last 10 candles)
  const shortReturns = returns.slice(-10);
  const shortMomentum =
    shortReturns.reduce((a, b) => a + b, 0) / shortReturns.length;

  // SMA crossover signal
  const sma20 = sma(closes, 20);
  const sma50 = sma(closes, 50);
  const smaCross = (sma20 - sma50) / sma50;

  // Composite score: weighted blend
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
    confidence = 0.4 + Math.random() * 0.15;
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

  // Build probability distribution (soft assignment)
  const regimes: Regime[] = [
    "strong_bull",
    "bull",
    "weak_bull",
    "neutral",
    "weak_bear",
    "bear",
    "crash",
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
  // Normalize
  for (const key of Object.keys(probs)) {
    probs[key] = parseFloat((probs[key] / total).toFixed(4));
  }

  return { regime, confidence, probabilities: probs };
}

// --------------- Technical Confirmations (8 signals) ---------------

export interface Confirmation {
  name: string;
  passed: boolean;
  value: number;
  threshold: string;
}

function computeConfirmations(
  closes: number[],
  volumes: number[],
  highs: number[],
  lows: number[]
): Confirmation[] {
  const currentClose = closes[closes.length - 1];

  // 1. RSI not overbought (< 70)
  const rsiVal = rsi(closes, 14);
  const rsiOk = rsiVal < 70 && rsiVal > 25;

  // 2. Price above SMA20
  const sma20 = sma(closes, 20);
  const aboveSma20 = currentClose > sma20;

  // 3. SMA20 above SMA50 (trend confirmation)
  const sma50 = sma(closes, 50);
  const smaCross = sma20 > sma50;

  // 4. Volume above 20-period average
  const avgVol = sma(volumes, 20);
  const currentVol = volumes[volumes.length - 1];
  const volOk = currentVol > avgVol * 0.8;

  // 5. EMA12 above EMA26 (MACD proxy)
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);
  const macdOk = ema12 > ema26;

  // 6. Price above lower Bollinger Band (not oversold crash)
  const sma20bb = sma(closes, 20);
  const std = stddev(closes, 20);
  const lowerBB = sma20bb - 2 * std;
  const aboveLowerBB = currentClose > lowerBB;

  // 7. Average True Range filter (volatility not extreme)
  const atrValues: number[] = [];
  for (
    let i = Math.max(1, closes.length - 14);
    i < closes.length;
    i++
  ) {
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
  const atrOk = atrPct < 5; // Not in extreme volatility

  // 8. Momentum: close > close 10 bars ago
  const close10ago =
    closes.length >= 11 ? closes[closes.length - 11] : closes[0];
  const momentumOk = currentClose > close10ago;

  return [
    {
      name: "RSI (14)",
      passed: rsiOk,
      value: parseFloat(rsiVal.toFixed(2)),
      threshold: "25 < RSI < 70",
    },
    {
      name: "Price > SMA20",
      passed: aboveSma20,
      value: parseFloat(((currentClose / sma20 - 1) * 100).toFixed(2)),
      threshold: "Above SMA20",
    },
    {
      name: "SMA20 > SMA50",
      passed: smaCross,
      value: parseFloat(((sma20 / sma50 - 1) * 100).toFixed(2)),
      threshold: "Golden cross",
    },
    {
      name: "Volume confirmation",
      passed: volOk,
      value: parseFloat((avgVol > 0 ? currentVol / avgVol : 0).toFixed(2)),
      threshold: "> 0.8x avg volume",
    },
    {
      name: "MACD (EMA12 > EMA26)",
      passed: macdOk,
      value: parseFloat((ema12 - ema26).toFixed(4)),
      threshold: "Positive MACD",
    },
    {
      name: "Above lower Bollinger",
      passed: aboveLowerBB,
      value: parseFloat(((currentClose - lowerBB) / currentClose * 100).toFixed(2)),
      threshold: "Price > BB lower",
    },
    {
      name: "ATR volatility filter",
      passed: atrOk,
      value: parseFloat(atrPct.toFixed(2)),
      threshold: "ATR% < 5%",
    },
    {
      name: "10-bar momentum",
      passed: momentumOk,
      value: parseFloat(
        (((currentClose - close10ago) / close10ago) * 100).toFixed(2)
      ),
      threshold: "Close > close[10]",
    },
  ];
}

// --------------- Core Backtester ---------------

function isBullish(regime: Regime): boolean {
  return regime === "strong_bull" || regime === "bull" || regime === "weak_bull";
}

export function runBacktest(
  candles: Candle[],
  config: BacktestConfig
): BacktestResult {
  const {
    initialCapital,
    leverage,
    requiredConfirmations,
    cooldownHours,
  } = config;

  const trades: BacktestTrade[] = [];
  const equityCurve: { date: string; equity: number; regime: string }[] = [];

  let equity = initialCapital;
  let peakEquity = initialCapital;
  let maxDrawdown = 0;

  // Position tracking
  let inPosition = false;
  let entryPrice = 0;
  let entryDate = "";
  let entryRegime = "";
  let entryConfirmations = 0;
  let lastExitTime = 0;

  // Sliding window arrays
  const closes: number[] = [];
  const volumes: number[] = [];
  const highs: number[] = [];
  const lows: number[] = [];

  // Need at least 50 candles for indicators
  const WARMUP = 55;

  // Equity tracking for Sharpe
  const periodicReturns: number[] = [];
  let prevEquity = initialCapital;

  // Stop loss: 3% from entry
  const STOP_LOSS_PCT = 0.03;

  for (let i = 0; i < candles.length; i++) {
    const c = candles[i];
    closes.push(c.close);
    volumes.push(c.volume);
    highs.push(c.high);
    lows.push(c.low);

    // Keep a rolling window of 200 max
    if (closes.length > 200) {
      closes.shift();
      volumes.shift();
      highs.shift();
      lows.shift();
    }

    if (i < WARMUP) continue;

    const regimeResult = detectRegime(closes);
    const regime = regimeResult.regime;
    const dateStr = new Date(c.openTime).toISOString();

    // Check cooldown
    const hoursSinceExit =
      lastExitTime > 0 ? (c.openTime - lastExitTime) / 3_600_000 : Infinity;
    const cooldownActive = hoursSinceExit < cooldownHours;

    if (inPosition) {
      // Check exit conditions
      const currentPnlPct = (c.close - entryPrice) / entryPrice;
      const hitStopLoss = currentPnlPct <= -STOP_LOSS_PCT;
      const regimeFlipped = !isBullish(regime);

      if (hitStopLoss || regimeFlipped) {
        const exitPrice = hitStopLoss
          ? entryPrice * (1 - STOP_LOSS_PCT)
          : c.close;
        const pnlPct = (exitPrice - entryPrice) / entryPrice;
        const pnl = equity * (pnlPct * leverage);

        equity += pnl;
        if (equity > peakEquity) peakEquity = equity;
        const dd = ((peakEquity - equity) / peakEquity) * 100;
        if (dd > maxDrawdown) maxDrawdown = dd;

        const entryTime = new Date(entryDate).getTime();
        const holdHours = (c.openTime - entryTime) / 3_600_000;

        trades.push({
          entryDate,
          exitDate: dateStr,
          entryPrice,
          exitPrice,
          side: "long",
          pnl: parseFloat(pnl.toFixed(2)),
          pnlPercent: parseFloat((pnlPct * 100).toFixed(2)),
          regime: entryRegime,
          confirmationsPassed: entryConfirmations,
          holdDurationHours: parseFloat(holdHours.toFixed(1)),
          exitReason: hitStopLoss ? "stop_loss" : "regime_flip",
        });

        inPosition = false;
        lastExitTime = c.openTime;
      }
    } else {
      // Check entry conditions
      if (!cooldownActive && isBullish(regime)) {
        const confirmations = computeConfirmations(
          closes,
          volumes,
          highs,
          lows
        );
        const passed = confirmations.filter((c) => c.passed).length;

        if (passed >= requiredConfirmations) {
          inPosition = true;
          entryPrice = c.close;
          entryDate = dateStr;
          entryRegime = regime;
          entryConfirmations = passed;
        }
      }
    }

    // Record equity curve (sample every 6 hours to keep payload manageable)
    if (i % 6 === 0) {
      const currentEquity = inPosition
        ? equity +
          equity *
            (((c.close - entryPrice) / entryPrice) * leverage)
        : equity;

      equityCurve.push({
        date: dateStr,
        equity: parseFloat(currentEquity.toFixed(2)),
        regime,
      });

      // Track returns for Sharpe
      if (prevEquity > 0) {
        periodicReturns.push((currentEquity - prevEquity) / prevEquity);
      }
      prevEquity = currentEquity;
    }
  }

  // Close any open position at last candle
  if (inPosition && candles.length > 0) {
    const lastCandle = candles[candles.length - 1];
    const pnlPct = (lastCandle.close - entryPrice) / entryPrice;
    const pnl = equity * (pnlPct * leverage);
    equity += pnl;
    if (equity > peakEquity) peakEquity = equity;

    const entryTime = new Date(entryDate).getTime();
    const holdHours = (lastCandle.openTime - entryTime) / 3_600_000;

    trades.push({
      entryDate,
      exitDate: new Date(lastCandle.openTime).toISOString(),
      exitPrice: lastCandle.close,
      entryPrice,
      side: "long",
      pnl: parseFloat(pnl.toFixed(2)),
      pnlPercent: parseFloat((pnlPct * 100).toFixed(2)),
      regime: entryRegime,
      confirmationsPassed: entryConfirmations,
      holdDurationHours: parseFloat(holdHours.toFixed(1)),
      exitReason: "regime_flip",
    });
  }

  // Compute metrics
  const firstClose = candles.length > WARMUP ? candles[WARMUP].close : candles[0]?.close || 1;
  const lastClose = candles[candles.length - 1]?.close || firstClose;
  const buyAndHoldReturn = ((lastClose - firstClose) / firstClose) * 100;
  const totalReturn = ((equity - initialCapital) / initialCapital) * 100;

  const winningTrades = trades.filter((t) => t.pnl > 0);
  const losingTrades = trades.filter((t) => t.pnl <= 0);

  // Sharpe ratio (annualized, assuming hourly candles sampled every 6h)
  const meanReturn =
    periodicReturns.length > 0
      ? periodicReturns.reduce((a, b) => a + b, 0) / periodicReturns.length
      : 0;
  const retStd = stddev(periodicReturns, periodicReturns.length);
  // ~1460 six-hour periods per year
  const sharpeRatio =
    retStd > 0
      ? parseFloat(((meanReturn / retStd) * Math.sqrt(1460)).toFixed(2))
      : 0;

  const avgTradeReturn =
    trades.length > 0
      ? trades.reduce((s, t) => s + t.pnlPercent, 0) / trades.length
      : 0;
  const avgHoldDuration =
    trades.length > 0
      ? trades.reduce((s, t) => s + t.holdDurationHours, 0) / trades.length
      : 0;

  // Regime breakdown
  const regimeCounts: Record<string, { count: number; totalReturn: number }> =
    {};
  for (const t of trades) {
    if (!regimeCounts[t.regime]) {
      regimeCounts[t.regime] = { count: 0, totalReturn: 0 };
    }
    regimeCounts[t.regime].count++;
    regimeCounts[t.regime].totalReturn += t.pnlPercent;
  }

  const regimeBreakdown = Object.entries(regimeCounts).map(
    ([regime, data]) => ({
      regime,
      count: data.count,
      avgReturn: parseFloat((data.totalReturn / data.count).toFixed(2)),
      percentage: parseFloat(
        ((data.count / Math.max(1, trades.length)) * 100).toFixed(1)
      ),
    })
  );

  return {
    config,
    trades,
    metrics: {
      totalReturn: parseFloat(totalReturn.toFixed(2)),
      alpha: parseFloat((totalReturn - buyAndHoldReturn).toFixed(2)),
      buyAndHoldReturn: parseFloat(buyAndHoldReturn.toFixed(2)),
      winRate:
        trades.length > 0
          ? parseFloat(
              ((winningTrades.length / trades.length) * 100).toFixed(1)
            )
          : 0,
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
      sharpeRatio,
      avgTradeReturn: parseFloat(avgTradeReturn.toFixed(2)),
      avgHoldDuration: parseFloat(avgHoldDuration.toFixed(1)),
      finalEquity: parseFloat(equity.toFixed(2)),
      peakEquity: parseFloat(peakEquity.toFixed(2)),
    },
    equityCurve,
    regimeBreakdown,
  };
}

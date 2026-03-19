// =============================================================================
// Technical Indicators Confirmation System
// 8 confirmation signals required (7/8 minimum) before entering a trade
// All calculations from scratch - no external libraries
// =============================================================================

export interface OHLCVCandle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ConfirmationResult {
  name: string;
  passed: boolean;
  value: number;
  threshold: string;
  description: string;
}

export interface ConfirmationAnalysis {
  confirmations: ConfirmationResult[];
  passed: number;
  required: number;
  total: number;
  overallPass: boolean;
  signalStrength: number;
}

// =============================================================================
// Core Math Helpers
// =============================================================================

export function calculateSMA(data: number[], period: number): number {
  if (data.length < period) return data.length > 0 ? data[data.length - 1] : 0;
  const slice = data.slice(data.length - period);
  return slice.reduce((sum, v) => sum + v, 0) / period;
}

export function calculateEMA(data: number[], period: number): number {
  if (data.length === 0) return 0;
  if (data.length < period) return calculateSMA(data, data.length);

  const k = 2 / (period + 1);
  // Seed EMA with SMA of first `period` values
  let ema = data.slice(0, period).reduce((s, v) => s + v, 0) / period;
  for (let i = period; i < data.length; i++) {
    ema = data[i] * k + ema * (1 - k);
  }
  return ema;
}

/**
 * Calculate full EMA series (needed for MACD signal line which is EMA of MACD line)
 */
function emaArray(data: number[], period: number): number[] {
  if (data.length === 0) return [];
  if (data.length < period) {
    // Return running SMA as fallback
    const result: number[] = [];
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i];
      result.push(sum / (i + 1));
    }
    return result;
  }

  const k = 2 / (period + 1);
  const result: number[] = new Array(data.length);

  // Fill initial values with SMA buildup
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += data[i];
    result[i] = sum / (i + 1);
  }
  // Seed with proper SMA
  result[period - 1] = sum / period;

  for (let i = period; i < data.length; i++) {
    result[i] = data[i] * k + result[i - 1] * (1 - k);
  }
  return result;
}

function standardDeviation(data: number[], period: number): number {
  if (data.length < period) return 0;
  const slice = data.slice(data.length - period);
  const mean = slice.reduce((s, v) => s + v, 0) / period;
  const variance = slice.reduce((s, v) => s + (v - mean) ** 2, 0) / period;
  return Math.sqrt(variance);
}

// =============================================================================
// Indicator Calculations
// =============================================================================

/**
 * RSI - Relative Strength Index
 * Standard Wilder's smoothing method
 */
export function calculateRSI(closes: number[], period: number = 14): number {
  if (closes.length < period + 1) return 50; // neutral default

  // Calculate price changes
  const changes: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1]);
  }

  // First average gain/loss using SMA
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i];
    else avgLoss += Math.abs(changes[i]);
  }
  avgGain /= period;
  avgLoss /= period;

  // Smooth with Wilder's method for remaining periods
  for (let i = period; i < changes.length; i++) {
    const gain = changes[i] > 0 ? changes[i] : 0;
    const loss = changes[i] < 0 ? Math.abs(changes[i]) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

/**
 * MACD - Moving Average Convergence Divergence
 * MACD line: EMA(12) - EMA(26)
 * Signal line: EMA(9) of MACD line
 * Histogram: MACD - Signal
 */
export function calculateMACD(closes: number[]): {
  macd: number;
  signal: number;
  histogram: number;
} {
  if (closes.length < 26) {
    return { macd: 0, signal: 0, histogram: 0 };
  }

  const ema12 = emaArray(closes, 12);
  const ema26 = emaArray(closes, 26);

  // MACD line series (only valid from index 25 onward)
  const macdLine: number[] = [];
  for (let i = 25; i < closes.length; i++) {
    macdLine.push(ema12[i] - ema26[i]);
  }

  // Signal line: EMA(9) of MACD line
  const signalLine = emaArray(macdLine, 9);

  const macd = macdLine[macdLine.length - 1];
  const signal = signalLine[signalLine.length - 1];

  return {
    macd,
    signal,
    histogram: macd - signal,
  };
}

/**
 * ATR - Average True Range
 */
export function calculateATR(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 14
): number {
  const len = highs.length;
  if (len < 2) return 0;

  const trueRanges: number[] = [];
  // First TR is just high - low
  trueRanges.push(highs[0] - lows[0]);

  for (let i = 1; i < len; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trueRanges.push(tr);
  }

  if (trueRanges.length < period) {
    return trueRanges.reduce((s, v) => s + v, 0) / trueRanges.length;
  }

  // Wilder's smoothing
  let atr = trueRanges.slice(0, period).reduce((s, v) => s + v, 0) / period;
  for (let i = period; i < trueRanges.length; i++) {
    atr = (atr * (period - 1) + trueRanges[i]) / period;
  }
  return atr;
}

/**
 * ADX - Average Directional Index
 * Measures trend strength regardless of direction
 */
export function calculateADX(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 14
): number {
  const len = highs.length;
  if (len < period * 2) return 0; // Need enough data

  // Step 1: Calculate +DM and -DM
  const plusDM: number[] = [];
  const minusDM: number[] = [];
  const trueRanges: number[] = [];

  for (let i = 1; i < len; i++) {
    const upMove = highs[i] - highs[i - 1];
    const downMove = lows[i - 1] - lows[i];

    plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);

    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trueRanges.push(tr);
  }

  if (trueRanges.length < period) return 0;

  // Step 2: Smooth with Wilder's method
  let smoothTR = trueRanges.slice(0, period).reduce((s, v) => s + v, 0);
  let smoothPlusDM = plusDM.slice(0, period).reduce((s, v) => s + v, 0);
  let smoothMinusDM = minusDM.slice(0, period).reduce((s, v) => s + v, 0);

  const dxValues: number[] = [];

  // First DX
  const plusDI = smoothTR !== 0 ? (smoothPlusDM / smoothTR) * 100 : 0;
  const minusDI = smoothTR !== 0 ? (smoothMinusDM / smoothTR) * 100 : 0;
  const diSum = plusDI + minusDI;
  if (diSum !== 0) {
    dxValues.push((Math.abs(plusDI - minusDI) / diSum) * 100);
  }

  for (let i = period; i < trueRanges.length; i++) {
    smoothTR = smoothTR - smoothTR / period + trueRanges[i];
    smoothPlusDM = smoothPlusDM - smoothPlusDM / period + plusDM[i];
    smoothMinusDM = smoothMinusDM - smoothMinusDM / period + minusDM[i];

    const pDI = smoothTR !== 0 ? (smoothPlusDM / smoothTR) * 100 : 0;
    const mDI = smoothTR !== 0 ? (smoothMinusDM / smoothTR) * 100 : 0;
    const sum = pDI + mDI;

    if (sum !== 0) {
      dxValues.push((Math.abs(pDI - mDI) / sum) * 100);
    } else {
      dxValues.push(0);
    }
  }

  if (dxValues.length < period) {
    return dxValues.length > 0
      ? dxValues.reduce((s, v) => s + v, 0) / dxValues.length
      : 0;
  }

  // Step 3: ADX = Wilder's smoothed average of DX
  let adx = dxValues.slice(0, period).reduce((s, v) => s + v, 0) / period;
  for (let i = period; i < dxValues.length; i++) {
    adx = (adx * (period - 1) + dxValues[i]) / period;
  }

  return adx;
}

/**
 * Bollinger Bands
 * Middle: SMA(period), Upper/Lower: middle +/- stdDev * std
 */
export function calculateBollingerBands(
  closes: number[],
  period: number = 20,
  stdDevMultiplier: number = 2
): { upper: number; middle: number; lower: number } {
  const middle = calculateSMA(closes, period);
  const std = standardDeviation(closes, period);

  return {
    upper: middle + stdDevMultiplier * std,
    middle,
    lower: middle - stdDevMultiplier * std,
  };
}

// =============================================================================
// Main Confirmation System
// =============================================================================

export function calculateConfirmations(
  ohlcv: OHLCVCandle[]
): ConfirmationAnalysis {
  const REQUIRED = 7;
  const TOTAL = 8;

  if (ohlcv.length < 2) {
    // Not enough data - return all failed
    const empty: ConfirmationResult[] = Array.from({ length: TOTAL }, (_, i) => ({
      name: `Signal ${i + 1}`,
      passed: false,
      value: 0,
      threshold: "N/A",
      description: "Not enough data",
    }));
    return {
      confirmations: empty,
      passed: 0,
      required: REQUIRED,
      total: TOTAL,
      overallPass: false,
      signalStrength: 0,
    };
  }

  const closes = ohlcv.map((c) => c.close);
  const highs = ohlcv.map((c) => c.high);
  const lows = ohlcv.map((c) => c.low);
  const volumes = ohlcv.map((c) => c.volume);
  const lastClose = closes[closes.length - 1];
  const lastVolume = volumes[volumes.length - 1];

  const confirmations: ConfirmationResult[] = [];

  // 1. RSI < 75 (not overbought)
  const rsi = calculateRSI(closes, 14);
  confirmations.push({
    name: "RSI",
    passed: rsi < 75,
    value: Math.round(rsi * 100) / 100,
    threshold: "< 75",
    description: rsi < 75
      ? `RSI at ${rsi.toFixed(1)} - not overbought`
      : `RSI at ${rsi.toFixed(1)} - overbought territory`,
  });

  // 2. MACD Bullish (MACD line above signal line)
  const macdResult = calculateMACD(closes);
  confirmations.push({
    name: "MACD Bullish",
    passed: macdResult.macd > macdResult.signal,
    value: Math.round(macdResult.histogram * 10000) / 10000,
    threshold: "MACD > Signal",
    description:
      macdResult.macd > macdResult.signal
        ? `MACD histogram positive (${macdResult.histogram.toFixed(4)})`
        : `MACD histogram negative (${macdResult.histogram.toFixed(4)})`,
  });

  // 3. Momentum - Price above 20-period SMA
  const sma20 = calculateSMA(closes, 20);
  confirmations.push({
    name: "Momentum",
    passed: lastClose > sma20,
    value: Math.round(lastClose * 100) / 100,
    threshold: `> SMA20 (${sma20.toFixed(2)})`,
    description:
      lastClose > sma20
        ? `Price ${lastClose.toFixed(2)} above SMA20 ${sma20.toFixed(2)}`
        : `Price ${lastClose.toFixed(2)} below SMA20 ${sma20.toFixed(2)}`,
  });

  // 4. Volatility OK - ATR not extreme
  const atr = calculateATR(highs, lows, closes, 14);
  // Calculate average ATR over last 50 periods (or available data)
  const atrLookback = Math.min(ohlcv.length, 50);
  let atrSum = 0;
  let atrCount = 0;
  // Calculate ATR for sub-windows to get an average ATR
  // Simpler approach: use a longer ATR period as baseline
  const atrLong = calculateATR(highs, lows, closes, Math.min(atrLookback, closes.length - 1));
  const avgAtr = atrLong > 0 ? atrLong : atr;
  const volatilityOk = atr < 2 * avgAtr;
  confirmations.push({
    name: "Volatility OK",
    passed: volatilityOk,
    value: Math.round(atr * 10000) / 10000,
    threshold: `ATR < ${(2 * avgAtr).toFixed(4)}`,
    description: volatilityOk
      ? `ATR ${atr.toFixed(4)} within normal range`
      : `ATR ${atr.toFixed(4)} indicates extreme volatility`,
  });

  // 5. Volume Confirmation - Volume above 20-period average
  const avgVolume20 = calculateSMA(volumes, 20);
  confirmations.push({
    name: "Volume Confirmation",
    passed: lastVolume > avgVolume20,
    value: Math.round(lastVolume),
    threshold: `> Avg20 (${Math.round(avgVolume20)})`,
    description:
      lastVolume > avgVolume20
        ? `Volume ${lastVolume.toFixed(0)} above average ${avgVolume20.toFixed(0)}`
        : `Volume ${lastVolume.toFixed(0)} below average ${avgVolume20.toFixed(0)}`,
  });

  // 6. ADX Trend Strength - ADX > 20
  const adx = calculateADX(highs, lows, closes, 14);
  confirmations.push({
    name: "ADX Trend Strength",
    passed: adx > 20,
    value: Math.round(adx * 100) / 100,
    threshold: "> 20",
    description:
      adx > 20
        ? `ADX at ${adx.toFixed(1)} - trending market`
        : `ADX at ${adx.toFixed(1)} - weak/no trend`,
  });

  // 7. Price Above EMA50
  const ema50 = calculateEMA(closes, 50);
  confirmations.push({
    name: "Price Above EMA50",
    passed: lastClose > ema50,
    value: Math.round(lastClose * 100) / 100,
    threshold: `> EMA50 (${ema50.toFixed(2)})`,
    description:
      lastClose > ema50
        ? `Price ${lastClose.toFixed(2)} above EMA50 ${ema50.toFixed(2)}`
        : `Price ${lastClose.toFixed(2)} below EMA50 ${ema50.toFixed(2)}`,
  });

  // 8. Bollinger Band Position - Price not above upper band
  const bb = calculateBollingerBands(closes, 20, 2);
  confirmations.push({
    name: "Bollinger Band Position",
    passed: lastClose < bb.upper,
    value: Math.round(lastClose * 100) / 100,
    threshold: `< Upper BB (${bb.upper.toFixed(2)})`,
    description:
      lastClose < bb.upper
        ? `Price ${lastClose.toFixed(2)} within Bollinger Bands`
        : `Price ${lastClose.toFixed(2)} above upper band ${bb.upper.toFixed(2)}`,
  });

  const passed = confirmations.filter((c) => c.passed).length;

  return {
    confirmations,
    passed,
    required: REQUIRED,
    total: TOTAL,
    overallPass: passed >= REQUIRED,
    signalStrength: Math.round((passed / TOTAL) * 100),
  };
}

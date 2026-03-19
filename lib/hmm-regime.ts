/**
 * HMM-inspired Regime Detection Engine
 *
 * Implements K-means + Gaussian clustering to detect 7 market regimes
 * from OHLCV data, inspired by Hidden Markov Model approaches used
 * by Renaissance Technologies.
 *
 * Pure TypeScript — no Python or external ML libraries.
 */

// ---------------------------------------------------------------------------
// Types & Enums
// ---------------------------------------------------------------------------

export enum MarketRegime {
  STRONG_BULL = "strong_bull",
  BULL = "bull",
  MODERATE_BULL = "moderate_bull",
  SIDEWAYS = "sideways",
  CHOPPY = "choppy",
  BEAR = "bear",
  CRASH = "crash",
}

const REGIME_ORDER: MarketRegime[] = [
  MarketRegime.STRONG_BULL,
  MarketRegime.BULL,
  MarketRegime.MODERATE_BULL,
  MarketRegime.SIDEWAYS,
  MarketRegime.CHOPPY,
  MarketRegime.BEAR,
  MarketRegime.CRASH,
];

export interface OHLCVData {
  timestamp: number; // ms epoch
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RegimeAnalysis {
  currentRegime: MarketRegime;
  regimeConfidence: number; // 0-100
  signal: "LONG_ENTER" | "LONG_HOLD" | "EXIT" | "CASH";
  regimeProbabilities: Record<MarketRegime, number>;
  regimeHistory: { timestamp: string; regime: MarketRegime }[];
  holdingSince: string | null;
  cooldownUntil: string | null;
  metadata: {
    dataPoints: number;
    features: { returns: number; range: number; volumeChange: number };
    regimeTransitions: number;
  };
}

interface FeatureVector {
  returns: number;
  range: number;
  volumeChange: number;
}

interface ClusterParams {
  mean: number[];
  std: number[];
  count: number;
}

interface TrainingResult {
  clusterParams: ClusterParams[];
  regimeMap: MarketRegime[]; // index = sorted cluster index → regime
  assignments: number[];
  features: number[][];
  timestamps: number[];
  trainedAt: number;
}

// ---------------------------------------------------------------------------
// In-memory cache  (per symbol, 1-hour TTL)
// ---------------------------------------------------------------------------

const CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour

const trainingCache = new Map<
  string,
  { result: TrainingResult; cachedAt: number }
>();

// ---------------------------------------------------------------------------
// Position state  (per symbol, in-memory for serverless simplicity)
// ---------------------------------------------------------------------------

interface PositionState {
  holdingSince: string | null;
  cooldownUntil: string | null;
  lastRegime: MarketRegime | null;
}

const positionState = new Map<string, PositionState>();

const COOLDOWN_MS = 48 * 60 * 60 * 1000; // 48 hours

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

function gaussianPdf(x: number, mean: number, std: number): number {
  const s = Math.max(std, 1e-10); // avoid division by zero
  return (
    Math.exp(-0.5 * ((x - mean) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))
  );
}

function euclidean(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

// ---------------------------------------------------------------------------
// K-Means (from scratch)
// ---------------------------------------------------------------------------

function kMeans(
  data: number[][],
  k: number,
  maxIter = 100
): { assignments: number[]; centroids: number[][] } {
  const n = data.length;
  const d = data[0].length;

  // Initialize centroids using k-means++ seeding
  const centroids: number[][] = [];
  const usedIndices = new Set<number>();

  // First centroid: random
  const first = Math.floor(Math.random() * n);
  centroids.push([...data[first]]);
  usedIndices.add(first);

  for (let c = 1; c < k; c++) {
    // Compute distance from each point to nearest existing centroid
    const dists: number[] = [];
    let totalDist = 0;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const cent of centroids) {
        const dist = euclidean(data[i], cent);
        if (dist < minDist) minDist = dist;
      }
      dists.push(minDist * minDist);
      totalDist += minDist * minDist;
    }
    // Weighted random selection
    let r = Math.random() * totalDist;
    for (let i = 0; i < n; i++) {
      r -= dists[i];
      if (r <= 0) {
        centroids.push([...data[i]]);
        break;
      }
    }
    if (centroids.length <= c) {
      // fallback
      centroids.push([...data[Math.floor(Math.random() * n)]]);
    }
  }

  const assignments = new Array<number>(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // Assign step
    for (let i = 0; i < n; i++) {
      let bestCluster = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c++) {
        const dist = euclidean(data[i], centroids[c]);
        if (dist < bestDist) {
          bestDist = dist;
          bestCluster = c;
        }
      }
      if (assignments[i] !== bestCluster) {
        assignments[i] = bestCluster;
        changed = true;
      }
    }

    if (!changed) break;

    // Update step
    const sums: number[][] = Array.from({ length: k }, () =>
      new Array(d).fill(0)
    );
    const counts = new Array(k).fill(0);

    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let j = 0; j < d; j++) {
        sums[c][j] += data[i][j];
      }
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) continue;
      for (let j = 0; j < d; j++) {
        centroids[c][j] = sums[c][j] / counts[c];
      }
    }
  }

  return { assignments, centroids };
}

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

function computeFeatures(candles: OHLCVData[]): {
  features: number[][];
  timestamps: number[];
} {
  const features: number[][] = [];
  const timestamps: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const prev = candles[i - 1];
    const cur = candles[i];

    if (prev.close <= 0 || prev.volume <= 0 || cur.close <= 0) continue;

    const returns = Math.log(cur.close / prev.close);
    const range = (cur.high - cur.low) / cur.close;
    const volumeChange =
      prev.volume > 0 ? Math.log(cur.volume / prev.volume) : 0;

    // Clamp extreme outliers to avoid skewing clustering
    const clamp = (v: number, lo: number, hi: number) =>
      Math.max(lo, Math.min(hi, v));

    features.push([
      clamp(returns, -0.15, 0.15),
      clamp(range, 0, 0.15),
      clamp(volumeChange, -3, 3),
    ]);
    timestamps.push(cur.timestamp);
  }

  return { features, timestamps };
}

// ---------------------------------------------------------------------------
// Normalize features (z-score) for clustering
// ---------------------------------------------------------------------------

function zNormalize(features: number[][]): {
  normalized: number[][];
  means: number[];
  stds: number[];
} {
  const d = features[0].length;
  const n = features.length;
  const means = new Array(d).fill(0);
  const stds = new Array(d).fill(0);

  for (const row of features) {
    for (let j = 0; j < d; j++) means[j] += row[j];
  }
  for (let j = 0; j < d; j++) means[j] /= n;

  for (const row of features) {
    for (let j = 0; j < d; j++) stds[j] += (row[j] - means[j]) ** 2;
  }
  for (let j = 0; j < d; j++) stds[j] = Math.sqrt(stds[j] / n) || 1e-10;

  const normalized = features.map((row) =>
    row.map((v, j) => (v - means[j]) / stds[j])
  );

  return { normalized, means, stds };
}

// ---------------------------------------------------------------------------
// Train: cluster + compute Gaussian params on raw features
// ---------------------------------------------------------------------------

function train(candles: OHLCVData[]): TrainingResult {
  const { features, timestamps } = computeFeatures(candles);

  if (features.length < 50) {
    throw new Error(
      `Insufficient data: ${features.length} data points (need >= 50)`
    );
  }

  const k = 7;

  // Normalize for clustering (so no single feature dominates distance)
  const { normalized } = zNormalize(features);

  // Run k-means multiple times, pick best (lowest inertia)
  let bestAssignments: number[] = [];
  let bestInertia = Infinity;

  const runs = 5;
  for (let r = 0; r < runs; r++) {
    const { assignments, centroids } = kMeans(normalized, k);
    let inertia = 0;
    for (let i = 0; i < assignments.length; i++) {
      inertia += euclidean(normalized[i], centroids[assignments[i]]) ** 2;
    }
    if (inertia < bestInertia) {
      bestInertia = inertia;
      bestAssignments = assignments;
    }
  }

  // Compute Gaussian params on RAW features per cluster
  const clusterParams: ClusterParams[] = Array.from({ length: k }, () => ({
    mean: [0, 0, 0],
    std: [0, 0, 0],
    count: 0,
  }));

  // Accumulate sums
  for (let i = 0; i < bestAssignments.length; i++) {
    const c = bestAssignments[i];
    clusterParams[c].count++;
    for (let j = 0; j < 3; j++) {
      clusterParams[c].mean[j] += features[i][j];
    }
  }

  // Means
  for (let c = 0; c < k; c++) {
    if (clusterParams[c].count === 0) continue;
    for (let j = 0; j < 3; j++) {
      clusterParams[c].mean[j] /= clusterParams[c].count;
    }
  }

  // Stds
  for (let i = 0; i < bestAssignments.length; i++) {
    const c = bestAssignments[i];
    for (let j = 0; j < 3; j++) {
      clusterParams[c].std[j] +=
        (features[i][j] - clusterParams[c].mean[j]) ** 2;
    }
  }
  for (let c = 0; c < k; c++) {
    if (clusterParams[c].count <= 1) continue;
    for (let j = 0; j < 3; j++) {
      clusterParams[c].std[j] = Math.sqrt(
        clusterParams[c].std[j] / clusterParams[c].count
      );
      if (clusterParams[c].std[j] < 1e-10) clusterParams[c].std[j] = 1e-10;
    }
  }

  // Sort clusters by mean return (descending) and map to regimes
  const sortedIndices = Array.from({ length: k }, (_, i) => i).sort(
    (a, b) => clusterParams[b].mean[0] - clusterParams[a].mean[0]
  );

  const regimeMap: MarketRegime[] = new Array(k);
  for (let rank = 0; rank < k; rank++) {
    regimeMap[sortedIndices[rank]] = REGIME_ORDER[rank];
  }

  return {
    clusterParams,
    regimeMap,
    assignments: bestAssignments,
    features,
    timestamps,
    trainedAt: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Classify a single feature vector against trained clusters
// ---------------------------------------------------------------------------

function classify(
  fv: number[],
  params: ClusterParams[],
  regimeMap: MarketRegime[]
): { regime: MarketRegime; confidence: number; probabilities: Record<MarketRegime, number> } {
  const k = params.length;
  const logLikelihoods: number[] = [];

  for (let c = 0; c < k; c++) {
    let logP = 0;
    for (let j = 0; j < 3; j++) {
      const p = gaussianPdf(fv[j], params[c].mean[j], params[c].std[j]);
      logP += Math.log(Math.max(p, 1e-300));
    }
    // Weight by cluster size (prior)
    logP += Math.log(Math.max(params[c].count, 1));
    logLikelihoods.push(logP);
  }

  // Convert to probabilities via log-sum-exp
  const maxLL = Math.max(...logLikelihoods);
  const expShifted = logLikelihoods.map((ll) => Math.exp(ll - maxLL));
  const sumExp = expShifted.reduce((a, b) => a + b, 0);
  const probs = expShifted.map((e) => e / sumExp);

  // Build regime probabilities
  const probabilities = {} as Record<MarketRegime, number>;
  for (const r of Object.values(MarketRegime)) {
    probabilities[r] = 0;
  }
  let bestCluster = 0;
  let bestProb = 0;
  for (let c = 0; c < k; c++) {
    const regime = regimeMap[c];
    probabilities[regime] = probs[c];
    if (probs[c] > bestProb) {
      bestProb = probs[c];
      bestCluster = c;
    }
  }

  return {
    regime: regimeMap[bestCluster],
    confidence: Math.round(bestProb * 100),
    probabilities,
  };
}

// ---------------------------------------------------------------------------
// Binance data fetcher
// ---------------------------------------------------------------------------

async function fetchBinanceKlines(
  symbol: string,
  interval = "1h",
  limit = 1000
): Promise<OHLCVData[]> {
  const url = `https://api.binance.com/api/v3/klines?symbol=${encodeURIComponent(
    symbol
  )}&interval=${interval}&limit=${limit}`;

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Binance API error: ${res.status} ${res.statusText}`
    );
  }

  const data: unknown[][] = await res.json();

  return data.map((k) => ({
    timestamp: k[0] as number,
    open: parseFloat(k[1] as string),
    high: parseFloat(k[2] as string),
    low: parseFloat(k[3] as string),
    close: parseFloat(k[4] as string),
    volume: parseFloat(k[5] as string),
  }));
}

// ---------------------------------------------------------------------------
// Signal generation
// ---------------------------------------------------------------------------

const BULLISH_REGIMES = new Set<MarketRegime>([
  MarketRegime.STRONG_BULL,
  MarketRegime.BULL,
  MarketRegime.MODERATE_BULL,
]);

const BEARISH_REGIMES = new Set<MarketRegime>([
  MarketRegime.BEAR,
  MarketRegime.CRASH,
]);

function generateSignal(
  regime: MarketRegime,
  state: PositionState,
  now: Date
): "LONG_ENTER" | "LONG_HOLD" | "EXIT" | "CASH" {
  const inPosition = state.holdingSince !== null;

  // Check cooldown
  if (state.cooldownUntil) {
    const cooldownEnd = new Date(state.cooldownUntil).getTime();
    if (now.getTime() < cooldownEnd) {
      return "CASH";
    }
    // Cooldown expired, clear it
    state.cooldownUntil = null;
  }

  // If in position
  if (inPosition) {
    if (BEARISH_REGIMES.has(regime) || regime === MarketRegime.CHOPPY) {
      // Exit and start cooldown
      state.holdingSince = null;
      state.cooldownUntil = new Date(now.getTime() + COOLDOWN_MS).toISOString();
      return "EXIT";
    }
    return "LONG_HOLD";
  }

  // Not in position
  if (BULLISH_REGIMES.has(regime)) {
    state.holdingSince = now.toISOString();
    return "LONG_ENTER";
  }

  return "CASH";
}

// ---------------------------------------------------------------------------
// Regime history from training assignments
// ---------------------------------------------------------------------------

function buildRegimeHistory(
  training: TrainingResult,
  recentCount = 48
): { timestamp: string; regime: MarketRegime }[] {
  const start = Math.max(0, training.assignments.length - recentCount);
  const history: { timestamp: string; regime: MarketRegime }[] = [];

  for (let i = start; i < training.assignments.length; i++) {
    history.push({
      timestamp: new Date(training.timestamps[i]).toISOString(),
      regime: training.regimeMap[training.assignments[i]],
    });
  }

  return history;
}

function countTransitions(assignments: number[]): number {
  let transitions = 0;
  for (let i = 1; i < assignments.length; i++) {
    if (assignments[i] !== assignments[i - 1]) transitions++;
  }
  return transitions;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Detect the current market regime for a given symbol.
 *
 * @param symbol  Binance symbol, e.g. "BTCUSDT"
 * @param historicalData  Optional pre-fetched OHLCV data (skips Binance fetch)
 */
export async function detectRegime(
  symbol: string,
  historicalData?: OHLCVData[]
): Promise<RegimeAnalysis> {
  const upperSymbol = symbol.toUpperCase();

  // Check cache
  const cached = trainingCache.get(upperSymbol);
  let training: TrainingResult;

  if (cached && Date.now() - cached.cachedAt < CACHE_TTL_MS) {
    training = cached.result;
  } else {
    // Fetch data if not provided
    const candles =
      historicalData ??
      (await fetchBinanceKlines(upperSymbol, "1h", 1000));

    training = train(candles);
    trainingCache.set(upperSymbol, {
      result: training,
      cachedAt: Date.now(),
    });
  }

  // Current feature vector = last computed feature
  const lastIdx = training.features.length - 1;
  const currentFeature = training.features[lastIdx];

  // Classify
  const { regime, confidence, probabilities } = classify(
    currentFeature,
    training.clusterParams,
    training.regimeMap
  );

  // Position state
  if (!positionState.has(upperSymbol)) {
    positionState.set(upperSymbol, {
      holdingSince: null,
      cooldownUntil: null,
      lastRegime: null,
    });
  }
  const state = positionState.get(upperSymbol)!;
  const now = new Date();

  const signal = generateSignal(regime, state, now);
  state.lastRegime = regime;

  // Regime history (last 48 candles)
  const regimeHistory = buildRegimeHistory(training, 48);

  return {
    currentRegime: regime,
    regimeConfidence: confidence,
    signal,
    regimeProbabilities: probabilities,
    regimeHistory,
    holdingSince: state.holdingSince,
    cooldownUntil: state.cooldownUntil,
    metadata: {
      dataPoints: training.features.length,
      features: {
        returns: currentFeature[0],
        range: currentFeature[1],
        volumeChange: currentFeature[2],
      },
      regimeTransitions: countTransitions(training.assignments),
    },
  };
}

/**
 * Get recommended trading action for a given regime.
 */
export function getRegimeAction(regime: MarketRegime): {
  action: string;
  leverage: number;
  description: string;
} {
  switch (regime) {
    case MarketRegime.STRONG_BULL:
      return {
        action: "AGGRESSIVE_BUY",
        leverage: 3,
        description:
          "Strong bullish trend detected. Max position size with leveraged longs.",
      };
    case MarketRegime.BULL:
      return {
        action: "BUY",
        leverage: 2,
        description:
          "Bullish regime. Standard long positions with moderate leverage.",
      };
    case MarketRegime.MODERATE_BULL:
      return {
        action: "LIGHT_BUY",
        leverage: 1.5,
        description:
          "Moderate uptrend. Smaller position sizes, scale in gradually.",
      };
    case MarketRegime.SIDEWAYS:
      return {
        action: "HOLD",
        leverage: 1,
        description:
          "Range-bound market. Hold existing positions, avoid new entries.",
      };
    case MarketRegime.CHOPPY:
      return {
        action: "AVOID",
        leverage: 0,
        description:
          "High volatility with no direction. Stay in cash, no new trades.",
      };
    case MarketRegime.BEAR:
      return {
        action: "DEFENSIVE",
        leverage: 0,
        description:
          "Bearish regime. Reduce exposure, consider hedging or exit.",
      };
    case MarketRegime.CRASH:
      return {
        action: "EXIT_ALL",
        leverage: 0,
        description:
          "Severe downturn detected. Liquidate all positions immediately.",
      };
  }
}

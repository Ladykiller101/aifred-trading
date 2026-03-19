import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

// In-memory cache with 30-second TTL
const priceCache = new Map<string, { price: number; timestamp: number }>();
const CACHE_TTL = 30_000; // 30 seconds

// Symbol mapping: our format → Binance format
const CRYPTO_MAP: Record<string, string> = {
  "BTC/USDT": "BTCUSDT",
  "ETH/USDT": "ETHUSDT",
  "SOL/USDT": "SOLUSDT",
  "BNB/USDT": "BNBUSDT",
  "XRP/USDT": "XRPUSDT",
  "ADA/USDT": "ADAUSDT",
  "DOGE/USDT": "DOGEUSDT",
  "AVAX/USDT": "AVAXUSDT",
  "DOT/USDT": "DOTUSDT",
  "MATIC/USDT": "MATICUSDT",
};

// Fallback prices for forex & stocks (updated periodically)
const FALLBACK_PRICES: Record<string, number> = {
  "EUR/USD": 1.0842, "GBP/USD": 1.2735, "USD/JPY": 149.85,
  "AUD/USD": 0.6521, "USD/CAD": 1.3612, "NZD/USD": 0.6043,
  "EUR/GBP": 0.8512, "EUR/JPY": 162.48, "GBP/JPY": 190.75,
  "USD/CHF": 0.8923,
  "AAPL": 189.45, "MSFT": 378.2, "TSLA": 245.6, "NVDA": 875.3,
  "GOOGL": 168.9, "AMZN": 185.7, "META": 495.3, "SPY": 521.4, "QQQ": 447.8,
};

function getCached(symbol: string): number | null {
  const cached = priceCache.get(symbol);
  if (!cached) return null;
  if (Date.now() - cached.timestamp > CACHE_TTL) {
    priceCache.delete(symbol);
    return null;
  }
  return cached.price;
}

function setCache(symbol: string, price: number) {
  priceCache.set(symbol, { price, timestamp: Date.now() });
}

async function fetchCryptoPrices(symbols: string[]): Promise<Record<string, number>> {
  const result: Record<string, number> = {};
  const toFetch: string[] = [];

  // Check cache first
  for (const sym of symbols) {
    const cached = getCached(sym);
    if (cached !== null) {
      result[sym] = cached;
    } else {
      toFetch.push(sym);
    }
  }

  if (toFetch.length === 0) return result;

  try {
    // Binance batch price endpoint (no API key needed)
    const binanceSymbols = toFetch
      .map(s => CRYPTO_MAP[s])
      .filter(Boolean);

    if (binanceSymbols.length > 0) {
      const url = `https://api.binance.us/api/v3/ticker/price?symbols=${JSON.stringify(binanceSymbols)}`;
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) });

      if (res.ok) {
        const data = await res.json();
        for (const item of data) {
          // Reverse lookup: BTCUSDT → BTC/USDT
          const original = Object.entries(CRYPTO_MAP).find(([, v]) => v === item.symbol)?.[0];
          if (original) {
            const price = parseFloat(item.price);
            result[original] = price;
            setCache(original, price);
          }
        }
      }
    }
  } catch (err) {
    console.error("Binance price fetch error:", err);
  }

  // Fill any missing with fallback
  for (const sym of toFetch) {
    if (!result[sym]) {
      result[sym] = FALLBACK_PRICES[sym] || 100;
    }
  }

  return result;
}

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const symbolsParam = url.searchParams.get("symbols");

    if (!symbolsParam) {
      // Return all available prices
      const cryptoSymbols = Object.keys(CRYPTO_MAP);
      const cryptoPrices = await fetchCryptoPrices(cryptoSymbols);

      const prices: Record<string, number> = { ...FALLBACK_PRICES, ...cryptoPrices };

      return NextResponse.json({
        success: true,
        prices,
        source: "binance+fallback",
        cached: priceCache.size > 0,
        timestamp: new Date().toISOString(),
      });
    }

    const symbols = symbolsParam.split(",").map(s => s.trim());
    const cryptoSymbols = symbols.filter(s => CRYPTO_MAP[s]);
    const otherSymbols = symbols.filter(s => !CRYPTO_MAP[s]);

    const cryptoPrices = await fetchCryptoPrices(cryptoSymbols);
    const otherPrices: Record<string, number> = {};
    for (const s of otherSymbols) {
      otherPrices[s] = FALLBACK_PRICES[s] || 100;
    }

    return NextResponse.json({
      success: true,
      prices: { ...otherPrices, ...cryptoPrices },
      source: cryptoSymbols.length > 0 ? "binance+fallback" : "fallback",
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Price API error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch prices", prices: FALLBACK_PRICES },
      { status: 500 }
    );
  }
}

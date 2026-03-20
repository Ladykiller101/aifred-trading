import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

// Use dynamic require to avoid Next.js bundling issues with ccxt
let ccxt: any;
try {
  ccxt = require("ccxt");
} catch {
  ccxt = null;
}

// Map broker IDs to ccxt exchange class names (lowercase)
const EXCHANGE_MAP: Record<string, string> = {
  binance: "binance",
  coinbase: "coinbase",
  kraken: "kraken",
  bybit: "bybit",
};

// Fallback balances for brokers not supported by ccxt
const MOCK_BALANCES: Record<string, Record<string, number>> = {
  alpaca: { USD: 25000.0, AAPL: 10, MSFT: 5 },
  oanda: { USD: 50000.0, EUR: 5000.0 },
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { brokerId, credentials } = body as {
      brokerId: string;
      credentials: Record<string, string>;
    };

    if (!brokerId || !credentials) {
      return NextResponse.json(
        { success: false, error: "Missing brokerId or credentials" },
        { status: 400 },
      );
    }

    // Check if ccxt is available and broker is supported
    const exchangeName = EXCHANGE_MAP[brokerId];

    if (ccxt && exchangeName && ccxt[exchangeName]) {
      // ── REAL validation via ccxt ──────────────────────────────────────
      const start = Date.now();

      try {
        const ExchangeClass = ccxt[exchangeName];

        // Normalize PEM format for CDP EC private keys
        let secret = credentials.api_secret || credentials.secret || "";
        if (secret.includes("BEGIN EC PRIVATE KEY")) {
          const pemBody = secret
            .replace(/-----BEGIN EC PRIVATE KEY-----/g, "")
            .replace(/-----END EC PRIVATE KEY-----/g, "")
            .replace(/[\s\r\n]+/g, "");
          const lines: string[] = [];
          for (let i = 0; i < pemBody.length; i += 64) {
            lines.push(pemBody.slice(i, i + 64));
          }
          secret = `-----BEGIN EC PRIVATE KEY-----\n${lines.join("\n")}\n-----END EC PRIVATE KEY-----\n`;
        }

        const exchange = new ExchangeClass({
          apiKey: credentials.api_key || credentials.apiKey,
          secret: secret,
          password: credentials.passphrase || credentials.password,
          enableRateLimit: true,
          timeout: 15000,
        });

        const balance = await exchange.fetchBalance();
        const latency = Date.now() - start;

        // Extract non-zero balances
        const nonZero: Record<string, number> = {};
        if (balance.total) {
          for (const [currency, amount] of Object.entries(balance.total)) {
            if (typeof amount === "number" && amount > 0) {
              nonZero[currency] = amount;
            }
          }
        }

        return NextResponse.json({
          success: true,
          latency_ms: latency,
          account_id: `${brokerId}_${Date.now().toString(36)}`,
          balance: Object.keys(nonZero).length > 0 ? nonZero : { USD: 0 },
          message: `Connected to ${brokerId} — ${Object.keys(nonZero).length} assets found`,
          source: "live",
        });
      } catch (err: any) {
        const latency = Date.now() - start;

        // Classify ccxt errors
        const errorName = err?.constructor?.name || "";
        let userMessage = "Connection test failed";
        let status = 500;

        if (
          errorName === "AuthenticationError" ||
          errorName === "InvalidCredentials"
        ) {
          userMessage =
            "Invalid API credentials — check your API key and secret";
          status = 401;
        } else if (errorName === "PermissionDenied") {
          userMessage =
            "API key lacks required permissions (need read access)";
          status = 403;
        } else if (
          errorName === "DDoSProtection" ||
          errorName === "RateLimitExceeded"
        ) {
          userMessage = "Exchange rate limited — try again in 30 seconds";
          status = 429;
        } else if (
          errorName === "NetworkError" ||
          errorName === "RequestTimeout"
        ) {
          userMessage = "Network timeout — check your connection";
          status = 504;
        } else if (errorName === "ExchangeNotAvailable") {
          userMessage = "Exchange temporarily unavailable";
          status = 503;
        }

        return NextResponse.json(
          {
            success: false,
            latency_ms: latency,
            error: userMessage,
            details: err?.message?.slice(0, 200),
            source: "live",
          },
          { status },
        );
      }
    }

    // ── Fallback: mock validation for unsupported brokers (alpaca, oanda) ──
    const mockDelay = 50 + Math.floor(Math.random() * 200);
    await new Promise((r) => setTimeout(r, mockDelay));

    // Basic credential validation
    const requiredFields = Object.values(credentials).filter(
      (v) => typeof v === "string" && v.trim().length > 0,
    );
    if (requiredFields.length < 2) {
      return NextResponse.json(
        { success: false, error: "Missing required credentials" },
        { status: 400 },
      );
    }

    return NextResponse.json({
      success: true,
      latency_ms: mockDelay,
      account_id: `${brokerId}_${Date.now().toString(36)}`,
      balance: MOCK_BALANCES[brokerId] ?? { USD: 0 },
      message: `Connected to ${brokerId} (simulated — real API integration coming soon)`,
      source: "mock",
    });
  } catch (error) {
    console.error("Broker test error:", error);
    return NextResponse.json(
      { success: false, error: "Unexpected error" },
      { status: 500 },
    );
  }
}

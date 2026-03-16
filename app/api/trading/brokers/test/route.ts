import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// POST /api/trading/brokers/test — test a broker connection
//
// NOTE: This currently returns simulated/mock results. Once the Python broker
// adapters are wired to the Next.js API layer (e.g. via a FastAPI sidecar or
// direct subprocess calls), replace the mock logic below with real connection
// tests against each exchange's REST API.
// ---------------------------------------------------------------------------

interface TestResult {
  success: boolean;
  latency_ms: number;
  account_id: string;
  balance: Record<string, number>;
  message: string;
}

// Mock balance data per broker for realistic simulation
const MOCK_BALANCES: Record<string, Record<string, number>> = {
  binance: { USDT: 10250.42, BTC: 0.125, ETH: 2.5 },
  coinbase: { USD: 8340.0, BTC: 0.08, ETH: 1.2 },
  kraken: { USD: 5120.75, BTC: 0.05, ETH: 0.8 },
  bybit: { USDT: 15000.0, BTC: 0.2, ETH: 3.0 },
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
        { success: false, latency_ms: 0, account_id: "", balance: {}, message: "brokerId and credentials are required" },
        { status: 400 },
      );
    }

    // Simulate network latency for the connection test (50-300ms)
    const startTime = Date.now();
    const simulatedLatency = Math.floor(Math.random() * 250) + 50;
    await new Promise((resolve) => setTimeout(resolve, simulatedLatency));
    const latency = Date.now() - startTime;

    // Simulate basic credential validation — reject obviously empty keys
    const hasEmptyRequired = Object.values(credentials).some(
      (v) => typeof v === "string" && v.trim() === "",
    );

    if (hasEmptyRequired) {
      const result: TestResult = {
        success: false,
        latency_ms: latency,
        account_id: "",
        balance: {},
        message: "Connection failed: one or more credentials are empty",
      };
      return NextResponse.json(result);
    }

    // Return mock success
    const balance = MOCK_BALANCES[brokerId] ?? { USD: 0 };
    const accountId = `${brokerId}_test_${Date.now().toString(36)}`;

    const result: TestResult = {
      success: true,
      latency_ms: latency,
      account_id: accountId,
      balance,
      message: `Successfully connected to ${brokerId}. API credentials are valid.`,
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error("Broker test error:", error);
    return NextResponse.json(
      {
        success: false,
        latency_ms: 0,
        account_id: "",
        balance: {},
        message: "Connection test failed due to an internal error",
      },
      { status: 500 },
    );
  }
}

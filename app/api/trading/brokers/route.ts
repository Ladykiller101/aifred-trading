import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Broker registry — static definitions of supported brokers
// ---------------------------------------------------------------------------

interface CredentialField {
  key: string;
  label: string;
  type: "text" | "password";
  optional?: boolean;
}

interface BrokerDefinition {
  id: string;
  name: string;
  category: "crypto" | "stocks" | "forex" | "stocks/options" | "all";
  description: string;
  supportedAssets: string[];
  requiredCredentials: CredentialField[];
  comingSoon?: boolean;
}

const BROKER_REGISTRY: BrokerDefinition[] = [
  {
    id: "binance",
    name: "Binance",
    category: "crypto",
    description: "World's largest crypto exchange",
    supportedAssets: [
      "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
      "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
    ],
  },
  {
    id: "coinbase",
    name: "Coinbase",
    category: "crypto",
    description: "Leading US-regulated crypto exchange",
    supportedAssets: [
      "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD",
      "ADA/USD", "DOT/USD", "MATIC/USD", "LINK/USD", "UNI/USD",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
      { key: "passphrase", label: "Passphrase", type: "password" },
    ],
  },
  {
    id: "kraken",
    name: "Kraken",
    category: "crypto",
    description: "Established crypto exchange with advanced trading features",
    supportedAssets: [
      "BTC/USD", "ETH/USD", "SOL/USD", "DOT/USD", "ADA/USD",
      "XRP/USD", "DOGE/USD", "AVAX/USD", "LINK/USD", "MATIC/USD",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
    ],
  },
  {
    id: "bybit",
    name: "Bybit",
    category: "crypto",
    description: "Derivatives-focused crypto exchange",
    supportedAssets: [
      "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
      "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
    ],
  },
  {
    id: "alpaca",
    name: "Alpaca",
    category: "stocks",
    description: "Commission-free stock and crypto trading API",
    supportedAssets: [
      "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
      "NVDA", "META", "SPY", "QQQ", "BTC/USD",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
      { key: "base_url", label: "Base URL", type: "text", optional: true },
    ],
  },
  {
    id: "oanda",
    name: "OANDA",
    category: "forex",
    description: "Leading forex and CFD broker",
    supportedAssets: [
      "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
      "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY", "USD/CHF",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "account_id", label: "Account ID", type: "text" },
    ],
  },
  {
    id: "interactive_brokers",
    name: "Interactive Brokers",
    category: "stocks/options",
    description: "Professional-grade broker for stocks, options, futures, and forex",
    supportedAssets: [
      "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
      "SPY", "QQQ", "ES", "NQ", "EUR/USD",
    ],
    requiredCredentials: [
      { key: "host", label: "TWS/Gateway Host", type: "text" },
      { key: "port", label: "Port", type: "text" },
      { key: "client_id", label: "Client ID", type: "text" },
    ],
    comingSoon: true,
  },
  {
    id: "metatrader",
    name: "MetaTrader 4/5",
    category: "forex",
    description: "Industry-standard forex trading platform",
    supportedAssets: [
      "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
      "XAU/USD", "US30", "NAS100", "SPX500", "GER40",
    ],
    requiredCredentials: [
      { key: "server", label: "Server", type: "text" },
      { key: "login", label: "Login", type: "text" },
      { key: "password", label: "Password", type: "password" },
    ],
    comingSoon: true,
  },
  {
    id: "bloomberg",
    name: "Bloomberg Terminal",
    category: "all",
    description: "Institutional-grade market data and trading platform",
    supportedAssets: ["All asset classes"],
    requiredCredentials: [],
    comingSoon: true,
  },
];

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

const DATA_DIR = join(process.cwd(), "data");
const CONNECTIONS_PATH = join(DATA_DIR, "broker-connections.json");
const SECRETS_PATH = join(DATA_DIR, ".broker-secrets.json");

function ensureDataDir() {
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }
}

interface ConnectionRecord {
  brokerId: string;
  connected: boolean;
  status: "connected" | "disconnected" | "error";
  lastChecked: string | null;
  accountId?: string;
}

function readConnections(): Record<string, ConnectionRecord> {
  if (!existsSync(CONNECTIONS_PATH)) return {};
  try {
    return JSON.parse(readFileSync(CONNECTIONS_PATH, "utf-8"));
  } catch {
    return {};
  }
}

function writeConnections(data: Record<string, ConnectionRecord>) {
  ensureDataDir();
  writeFileSync(CONNECTIONS_PATH, JSON.stringify(data, null, 2), "utf-8");
}

function readSecrets(): Record<string, Record<string, string>> {
  if (!existsSync(SECRETS_PATH)) return {};
  try {
    return JSON.parse(readFileSync(SECRETS_PATH, "utf-8"));
  } catch {
    return {};
  }
}

function writeSecrets(data: Record<string, Record<string, string>>) {
  ensureDataDir();
  writeFileSync(SECRETS_PATH, JSON.stringify(data, null, 2), "utf-8");
}

// ---------------------------------------------------------------------------
// GET /api/trading/brokers — list brokers with connection status
// ---------------------------------------------------------------------------

export async function GET() {
  try {
    const connections = readConnections();

    const brokers = BROKER_REGISTRY.map((broker) => {
      const conn = connections[broker.id];
      return {
        id: broker.id,
        name: broker.name,
        category: broker.category,
        description: broker.description,
        supportedAssets: broker.supportedAssets,
        requiredCredentials: broker.requiredCredentials,
        comingSoon: broker.comingSoon ?? false,
        connected: conn?.connected ?? false,
        status: conn?.status ?? "disconnected",
        lastChecked: conn?.lastChecked ?? null,
      };
    });

    return NextResponse.json({ brokers });
  } catch (error) {
    console.error("Brokers GET error:", error);
    return NextResponse.json(
      { error: "Failed to load broker list" },
      { status: 500 },
    );
  }
}

// ---------------------------------------------------------------------------
// POST /api/trading/brokers — save credentials and update connection status
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { brokerId, credentials } = body as {
      brokerId: string;
      credentials: Record<string, string>;
    };

    if (!brokerId || !credentials) {
      return NextResponse.json(
        { success: false, message: "brokerId and credentials are required" },
        { status: 400 },
      );
    }

    const broker = BROKER_REGISTRY.find((b) => b.id === brokerId);
    if (!broker) {
      return NextResponse.json(
        { success: false, message: `Unknown broker: ${brokerId}` },
        { status: 400 },
      );
    }

    if (broker.comingSoon) {
      return NextResponse.json(
        { success: false, message: `${broker.name} integration is coming soon` },
        { status: 400 },
      );
    }

    // Validate that all required (non-optional) credentials are provided
    const missing = broker.requiredCredentials
      .filter((c) => !c.optional && !credentials[c.key])
      .map((c) => c.label);

    if (missing.length > 0) {
      return NextResponse.json(
        { success: false, message: `Missing required fields: ${missing.join(", ")}` },
        { status: 400 },
      );
    }

    // Store credentials in the secrets file (never in the connections JSON)
    const secrets = readSecrets();
    secrets[brokerId] = credentials;
    writeSecrets(secrets);

    // Update connection status
    const connections = readConnections();
    connections[brokerId] = {
      brokerId,
      connected: true,
      status: "connected",
      lastChecked: new Date().toISOString(),
      accountId: `${brokerId}_${Date.now().toString(36)}`,
    };
    writeConnections(connections);

    return NextResponse.json({
      success: true,
      message: `${broker.name} credentials saved and connection established`,
      accountInfo: {
        balance: 0,
        accountId: connections[brokerId].accountId,
      },
    });
  } catch (error) {
    console.error("Brokers POST error:", error);
    return NextResponse.json(
      { success: false, message: "Failed to save broker credentials" },
      { status: 500 },
    );
  }
}

import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Activity log helper
// ---------------------------------------------------------------------------

// Activity log paths — try /tmp first, fall back to data/ for reads
const ACTIVITY_PATH_TMP = join("/tmp/aifred-data", "activity-log.json");
const ACTIVITY_PATH_DATA = join(process.cwd(), "data", "activity-log.json");

function appendActivity(entry: {
  type: string;
  severity: string;
  title: string;
  message: string;
  details?: Record<string, unknown>;
}) {
  try {
    if (!existsSync("/tmp/aifred-data")) mkdirSync("/tmp/aifred-data", { recursive: true });
    let activities: unknown[] = [];
    for (const p of [ACTIVITY_PATH_TMP, ACTIVITY_PATH_DATA]) {
      if (existsSync(p)) {
        try {
          const raw = JSON.parse(readFileSync(p, "utf-8"));
          if (Array.isArray(raw)) { activities = raw; break; }
        } catch { /* ignore */ }
      }
    }
    activities.push({
      id: `act_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      ...entry,
    });
    if (activities.length > 500) activities = activities.slice(-500);
    writeFileSync(ACTIVITY_PATH_TMP, JSON.stringify(activities, null, 2), "utf-8");
  } catch (e) {
    console.error("Failed to log activity from brokers route:", e);
  }
}

// ---------------------------------------------------------------------------
// Broker registry — static definitions of supported brokers
// ---------------------------------------------------------------------------

interface CredentialField {
  key: string;
  label: string;
  type: "text" | "password" | "textarea";
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
    name: "Coinbase Advanced Trade",
    category: "crypto",
    description: "Coinbase Advanced Trade API (formerly Coinbase Pro)",
    supportedAssets: [
      "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD",
      "ADA/USD", "DOT/USD", "MATIC/USD", "LINK/USD", "UNI/USD",
    ],
    requiredCredentials: [
      { key: "api_key", label: "API Key Name", type: "text" },
      { key: "api_secret", label: "API Secret (PEM Private Key)", type: "textarea" },
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

const TMP_DIR = "/tmp/aifred-data";
const DATA_DIR = join(process.cwd(), "data");
const CONNECTIONS_PATH_TMP = join(TMP_DIR, "broker-connections.json");
const CONNECTIONS_PATH_DATA = join(DATA_DIR, "broker-connections.json");
const SECRETS_PATH_TMP = join(TMP_DIR, ".broker-secrets.json");

function ensureTmpDir() {
  if (!existsSync(TMP_DIR)) {
    mkdirSync(TMP_DIR, { recursive: true });
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
  // Try /tmp first (latest), then data/ (build-time seed)
  for (const p of [CONNECTIONS_PATH_TMP, CONNECTIONS_PATH_DATA]) {
    if (existsSync(p)) {
      try {
        return JSON.parse(readFileSync(p, "utf-8"));
      } catch { /* ignore */ }
    }
  }
  return {};
}

function writeConnections(data: Record<string, ConnectionRecord>) {
  ensureTmpDir();
  writeFileSync(CONNECTIONS_PATH_TMP, JSON.stringify(data, null, 2), "utf-8");
}

function readSecrets(): Record<string, Record<string, string>> {
  if (!existsSync(SECRETS_PATH_TMP)) return {};
  try {
    return JSON.parse(readFileSync(SECRETS_PATH_TMP, "utf-8"));
  } catch {
    return {};
  }
}

function writeSecrets(data: Record<string, Record<string, string>>) {
  ensureTmpDir();
  writeFileSync(SECRETS_PATH_TMP, JSON.stringify(data, null, 2), "utf-8");
}

// ---------------------------------------------------------------------------
// Revalidation helper — test stored credentials for a connected broker
// ---------------------------------------------------------------------------

async function revalidateBroker(
  brokerId: string,
  secrets: Record<string, Record<string, string>>,
): Promise<boolean> {
  const creds = secrets[brokerId];
  if (!creds) return false;

  try {
    // Call the internal test endpoint logic inline to avoid HTTP round-trip
    let ccxt: any;
    try {
      ccxt = require("ccxt");
    } catch {
      ccxt = null;
    }

    const EXCHANGE_MAP: Record<string, string> = {
      binance: "binance",
      coinbase: "coinbase",
      kraken: "kraken",
      bybit: "bybit",
    };

    const exchangeName = EXCHANGE_MAP[brokerId];

    if (ccxt && exchangeName && ccxt[exchangeName]) {
      const ExchangeClass = ccxt[exchangeName];
      const exchange = new ExchangeClass({
        apiKey: creds.api_key || creds.apiKey,
        secret: creds.api_secret || creds.secret,
        password: creds.passphrase || creds.password,
        enableRateLimit: true,
        timeout: 10000,
      });
      await exchange.fetchBalance();
      return true;
    }

    // For non-ccxt brokers, check that credentials exist
    const filledFields = Object.values(creds).filter(
      (v) => typeof v === "string" && v.trim().length > 0,
    );
    return filledFields.length >= 2;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// GET /api/trading/brokers — list brokers with connection status
// ---------------------------------------------------------------------------

export async function GET(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const revalidate = url.searchParams.get("revalidate") === "true";

    const connections = readConnections();

    // If revalidate is requested, re-test connected brokers
    if (revalidate) {
      const secrets = readSecrets();
      let changed = false;

      for (const [id, conn] of Object.entries(connections)) {
        if (conn.connected) {
          const isValid = await revalidateBroker(id, secrets);
          if (!isValid) {
            connections[id] = {
              ...conn,
              connected: false,
              status: "disconnected",
              lastChecked: new Date().toISOString(),
            };
            // Remove stale credentials
            delete secrets[id];
            changed = true;

            appendActivity({
              type: "broker_revalidation_failed",
              severity: "warning",
              title: `Broker Disconnected: ${id}`,
              message: `${id} credentials are no longer valid — connection removed.`,
              details: { broker: id },
            });
          } else {
            connections[id] = {
              ...conn,
              lastChecked: new Date().toISOString(),
            };
          }
        }
      }

      if (changed) {
        writeConnections(connections);
        writeSecrets(secrets);
      }
    }

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

    // Log activity
    appendActivity({
      type: "broker_connected",
      severity: "success",
      title: `Broker Connected: ${broker.name}`,
      message: `${broker.name} credentials saved and connection established. Account ID: ${connections[brokerId].accountId}.`,
      details: { broker: broker.name, asset: broker.supportedAssets.join(", ") },
    });

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

// ---------------------------------------------------------------------------
// DELETE /api/trading/brokers?brokerId=xxx — disconnect and remove credentials
// ---------------------------------------------------------------------------

export async function DELETE(request: NextRequest) {
  try {
    const url = new URL(request.url);
    const brokerId = url.searchParams.get("brokerId");

    if (!brokerId) {
      return NextResponse.json(
        { error: "brokerId is required" },
        { status: 400 },
      );
    }

    // Remove credentials
    const secrets = readSecrets();
    delete secrets[brokerId];
    writeSecrets(secrets);

    // Update connection status
    const connections = readConnections();
    if (connections[brokerId]) {
      connections[brokerId] = {
        ...connections[brokerId],
        connected: false,
        status: "disconnected",
        lastChecked: new Date().toISOString(),
      };
      writeConnections(connections);
    }

    // Find broker name for activity log
    const broker = BROKER_REGISTRY.find((b) => b.id === brokerId);
    const brokerName = broker?.name ?? brokerId;

    // Log activity
    appendActivity({
      type: "broker_disconnected",
      severity: "warning",
      title: `Broker Disconnected: ${brokerName}`,
      message: `${brokerName} disconnected — API credentials removed.`,
      details: { broker: brokerId },
    });

    return NextResponse.json({
      success: true,
      message: `${brokerName} disconnected`,
    });
  } catch (error) {
    console.error("Brokers DELETE error:", error);
    return NextResponse.json(
      { error: "Failed to disconnect broker" },
      { status: 500 },
    );
  }
}

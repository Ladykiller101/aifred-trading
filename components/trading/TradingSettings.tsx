"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  Coins,
  Building2,
  Globe,
  Monitor,
  Shield,
  Zap,
  Bell,
  Key,
  X,
  Check,
  AlertTriangle,
  Loader2,
  Play,
  Square,
  Settings,
  Wifi,
  WifiOff,
  Clock,
  Send,
  Brain,
} from "lucide-react";
import { useRouter } from "next/navigation";

// ─── Types ────────────────────────────────────────────────────
interface Broker {
  id: string;
  name: string;
  category: "crypto" | "stocks" | "forex";
  description: string;
  icon: "coins" | "building" | "globe" | "monitor";
  status: "connected" | "disconnected" | "error";
  comingSoon?: boolean;
  requiredCredentials: { key: string; label: string; type: string }[];
  accountInfo?: { label: string; value: string }[];
}

interface TradingControls {
  mode: "paper" | "live";
  scanInterval: number;
  isRunning: boolean;
  assets: { symbol: string; category: string; enabled: boolean }[];
}

// ─── Default data ─────────────────────────────────────────────
const DEFAULT_BROKERS: Broker[] = [
  {
    id: "alpaca",
    name: "Alpaca",
    category: "stocks",
    description: "Commission-free stock & crypto trading API",
    icon: "building",
    status: "disconnected",
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
      { key: "base_url", label: "Base URL", type: "text" },
    ],
  },
  {
    id: "binance",
    name: "Binance",
    category: "crypto",
    description: "World's largest cryptocurrency exchange",
    icon: "coins",
    status: "disconnected",
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
    ],
  },
  {
    id: "coinbase",
    name: "Coinbase Pro",
    category: "crypto",
    description: "Advanced crypto trading platform",
    icon: "coins",
    status: "disconnected",
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "api_secret", label: "API Secret", type: "password" },
      { key: "passphrase", label: "Passphrase", type: "password" },
    ],
  },
  {
    id: "oanda",
    name: "OANDA",
    category: "forex",
    description: "Forex & CFD trading platform",
    icon: "globe",
    status: "disconnected",
    requiredCredentials: [
      { key: "api_key", label: "API Key", type: "text" },
      { key: "account_id", label: "Account ID", type: "text" },
    ],
  },
  {
    id: "interactive_brokers",
    name: "Interactive Brokers",
    category: "stocks",
    description: "Professional-grade multi-asset broker",
    icon: "building",
    status: "disconnected",
    comingSoon: true,
    requiredCredentials: [],
  },
  {
    id: "metatrader",
    name: "MetaTrader 5",
    category: "forex",
    description: "Industry-standard forex trading terminal",
    icon: "globe",
    status: "disconnected",
    comingSoon: true,
    requiredCredentials: [],
  },
  {
    id: "bloomberg",
    name: "Bloomberg Terminal",
    category: "stocks",
    description: "Institutional-grade market data & execution",
    icon: "monitor",
    status: "disconnected",
    comingSoon: true,
    requiredCredentials: [],
  },
];

const DEFAULT_CONTROLS: TradingControls = {
  mode: "paper",
  scanInterval: 30,
  isRunning: false,
  assets: [
    { symbol: "BTC/USDT", category: "crypto", enabled: true },
    { symbol: "ETH/USDT", category: "crypto", enabled: true },
    { symbol: "SOL/USDT", category: "crypto", enabled: true },
    { symbol: "AAPL", category: "stocks", enabled: true },
    { symbol: "TSLA", category: "stocks", enabled: true },
    { symbol: "NVDA", category: "stocks", enabled: true },
    { symbol: "MSFT", category: "stocks", enabled: false },
    { symbol: "EUR/USD", category: "forex", enabled: false },
    { symbol: "GBP/USD", category: "forex", enabled: false },
  ],
};

const RISK_PARAMS = [
  { label: "Max Position Size", value: "3%", icon: Shield },
  { label: "Max Risk Per Trade", value: "1.5%", icon: AlertTriangle },
  { label: "Max Concurrent Positions", value: "10", icon: Zap },
  { label: "Daily Drawdown Limit", value: "5%", icon: Shield },
  { label: "Confidence Threshold", value: "78%", icon: Brain },
];

// ─── Injected styles ──────────────────────────────────────────
const SETTINGS_STYLES = `
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

@keyframes pulse-glow {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}
@keyframes pulse-ring {
  0% { transform: scale(1); opacity: 0.6; }
  100% { transform: scale(2.5); opacity: 0; }
}
.glow-green { text-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
.glow-red { text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
.card-glass {
  background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.06);
}
.card-glass:hover {
  border-color: rgba(255,255,255,0.12);
}
.noise-bg {
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
}
.toggle-track {
  transition: background-color 0.3s ease;
}
.toggle-thumb {
  transition: transform 0.3s ease;
}
`;

// ─── Broker icon map ──────────────────────────────────────────
const BROKER_ICONS: Record<string, React.ReactNode> = {
  coins: <Coins className="w-5 h-5" />,
  building: <Building2 className="w-5 h-5" />,
  globe: <Globe className="w-5 h-5" />,
  monitor: <Monitor className="w-5 h-5" />,
};

const CATEGORY_COLORS: Record<string, string> = {
  crypto: "text-amber-400 bg-amber-400/10",
  stocks: "text-indigo-400 bg-indigo-400/10",
  forex: "text-emerald-400 bg-emerald-400/10",
};

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
export default function TradingSettings() {
  const router = useRouter();
  const [brokers, setBrokers] = useState<Broker[]>(DEFAULT_BROKERS);
  const [controls, setControls] = useState<TradingControls>(DEFAULT_CONTROLS);
  const [selectedBroker, setSelectedBroker] = useState<Broker | null>(null);
  const [credentials, setCredentials] = useState<Record<string, string>>({});
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    latency?: number;
    accountInfo?: { label: string; value: string }[];
    error?: string;
  } | null>(null);
  const [saving, setSaving] = useState(false);
  const [showLiveWarning, setShowLiveWarning] = useState(false);
  const [notifications, setNotifications] = useState({
    telegramBotToken: "",
    telegramChatId: "",
    anthropicApiKey: "",
    newsApiKey: "",
  });

  // Inject styles
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = SETTINGS_STYLES;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Load brokers — merge API data with localStorage (localStorage wins for connection status)
  useEffect(() => {
    const categoryToIcon: Record<string, Broker["icon"]> = {
      crypto: "coins",
      stocks: "building",
      "stocks/options": "building",
      forex: "globe",
      all: "monitor",
    };

    // Read persisted connections from localStorage
    let savedConnections: Record<string, string> = {};
    try {
      const raw = localStorage.getItem("aifred_broker_connections");
      if (raw) savedConnections = JSON.parse(raw);
    } catch { /* ignore */ }

    fetch("/api/trading/brokers")
      .then((r) => r.json())
      .then((data) => {
        if (data && Array.isArray(data.brokers)) {
          const mapped: Broker[] = data.brokers.map(
            (b: Record<string, unknown>) => ({
              id: b.id as string,
              name: b.name as string,
              category: (
                b.category === "stocks/options" || b.category === "all"
                  ? "stocks"
                  : b.category
              ) as Broker["category"],
              description: b.description as string,
              icon: categoryToIcon[(b.category as string) || ""] || "monitor",
              // localStorage status takes precedence (survives page refresh)
              status: (savedConnections[b.id as string] as Broker["status"]) ||
                (b.status as Broker["status"]) || "disconnected",
              comingSoon: (b.comingSoon as boolean) || false,
              requiredCredentials:
                (b.requiredCredentials as Broker["requiredCredentials"]) || [],
            })
          );
          setBrokers(mapped);
        }
      })
      .catch(() => {
        // Use defaults but still apply localStorage connections
        if (Object.keys(savedConnections).length > 0) {
          setBrokers((prev) =>
            prev.map((b) => ({
              ...b,
              status: (savedConnections[b.id] as Broker["status"]) || b.status,
            }))
          );
        }
      });
  }, []);

  // Load controls — localStorage wins for persistence across refreshes
  useEffect(() => {
    // Always start with DEFAULT_CONTROLS as safe base, then overlay localStorage, then don't fetch API
    let savedFromStorage: Partial<TradingControls> | null = null;
    try {
      const raw = localStorage.getItem("aifred_trading_controls");
      if (raw) {
        const parsed = JSON.parse(raw);
        // Validate assets is a proper array of objects (guard against corruption)
        const assets = Array.isArray(parsed.assets) && parsed.assets.length > 0 && typeof parsed.assets[0] === "object"
          ? parsed.assets as TradingControls["assets"]
          : DEFAULT_CONTROLS.assets;
        savedFromStorage = { ...parsed, assets };
      }
    } catch { /* ignore */ }

    if (savedFromStorage) {
      // Merge with DEFAULT_CONTROLS so all fields are always defined
      setControls({
        ...DEFAULT_CONTROLS,
        ...savedFromStorage,
      });
      return; // localStorage is the source of truth — skip API to avoid overwrite
    }

    // No localStorage — fetch from API to get server-side state
    fetch("/api/trading/controls")
      .then((r) => r.json())
      .then((data) => {
        if (data && data.mode) {
          // API returns { mode, running, scanInterval, assets: string[] }
          // Reconstruct assets as objects using DEFAULT_CONTROLS as base
          const assetSymbols: string[] = Array.isArray(data.assets) ? data.assets : [];
          const apiSet = new Set(assetSymbols);
          const mergedAssets = DEFAULT_CONTROLS.assets.map((a) => ({
            ...a,
            enabled: apiSet.has(a.symbol) ? true : a.enabled,
          }));
          for (const sym of assetSymbols) {
            if (!mergedAssets.find((a) => a.symbol === sym)) {
              const isForex = sym.includes("/") && !sym.includes("USDT") &&
                !["BTC","ETH","SOL","DOGE","ADA","XRP","DOT","AVAX","MATIC","LINK","BNB"].some((c) => sym.startsWith(c));
              const isCrypto = sym.includes("USDT") || sym.includes("BTC") || sym.includes("ETH");
              mergedAssets.push({ symbol: sym, category: isForex ? "forex" : isCrypto ? "crypto" : "stocks", enabled: true });
            }
          }
          setControls({
            mode: data.mode,
            // Normalize scanInterval: API uses 300 (5 min) as default, frontend uses 30s
            scanInterval: data.scanInterval === 300 ? DEFAULT_CONTROLS.scanInterval : (data.scanInterval ?? DEFAULT_CONTROLS.scanInterval),
            isRunning: data.running ?? false,
            assets: mergedAssets,
          });
        }
      })
      .catch(() => {
        // Use DEFAULT_CONTROLS already set as initial state
      });
  }, []);

  const handleTestConnection = useCallback(async () => {
    if (!selectedBroker) return;
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/trading/brokers/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brokerId: selectedBroker.id,
          credentials,
        }),
      });
      const data = await res.json();
      setTestResult(data);
    } catch {
      setTestResult({ success: false, error: "Network error — API unreachable" });
    } finally {
      setTesting(false);
    }
  }, [selectedBroker, credentials]);

  const handleSaveConnection = useCallback(async () => {
    if (!selectedBroker) return;
    setSaving(true);
    try {
      const res = await fetch("/api/trading/brokers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          brokerId: selectedBroker.id,
          credentials,
        }),
      });
      const data = await res.json();
      if (data.success) {
        // Update local broker status immediately
        setBrokers((prev) =>
          prev.map((b) =>
            b.id === selectedBroker.id ? { ...b, status: "connected" as const } : b
          )
        );
        // Persist to localStorage so connection survives page refresh
        try {
          const raw = localStorage.getItem("aifred_broker_connections");
          const saved: Record<string, string> = raw ? JSON.parse(raw) : {};
          saved[selectedBroker.id] = "connected";
          localStorage.setItem("aifred_broker_connections", JSON.stringify(saved));
        } catch { /* ignore */ }
      }
      setSelectedBroker(null);
      setCredentials({});
      setTestResult(null);
    } catch {
      // Silently handle
    } finally {
      setSaving(false);
    }
  }, [selectedBroker, credentials]);

  const updateControls = useCallback(
    async (updates: Partial<TradingControls>) => {
      const newControls = { ...controls, ...updates };
      setControls(newControls);

      // Determine the correct API action
      let action: "start" | "stop" | "toggle_mode" | null = null;
      let extraPayload: Record<string, unknown> = {};

      if (updates.isRunning !== undefined && updates.isRunning !== controls.isRunning) {
        action = updates.isRunning ? "start" : "stop";
      } else if (updates.mode !== undefined && updates.mode !== controls.mode) {
        action = "toggle_mode";
        extraPayload.mode = updates.mode;
      }

      // For asset or scanInterval changes without a start/stop/mode action,
      // we still need to call the API. Use a start/stop to persist.
      if (!action && (updates.assets !== undefined || updates.scanInterval !== undefined)) {
        // Persist via a toggle that matches current state
        action = newControls.isRunning ? "start" : "stop";
      }

      if (!action) return;

      try {
        const enabledSymbols = newControls.assets
          .filter((a) => a.enabled)
          .map((a) => a.symbol);

        const res = await fetch("/api/trading/controls", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action,
            scanInterval: newControls.scanInterval,
            assets: enabledSymbols,
            ...extraPayload,
          }),
        });
        const data = await res.json();
        if (data.currentState) {
          // Update local state from API response to stay in sync
          setControls((prev) => ({
            ...prev,
            mode: data.currentState.mode ?? prev.mode,
            isRunning: data.currentState.running ?? prev.isRunning,
            scanInterval: data.currentState.scanInterval ?? prev.scanInterval,
          }));
        }
      } catch {
        // API failed — that's ok, state is already saved locally
      }
      // Always persist to localStorage (survives page refresh regardless of API)
      try {
        localStorage.setItem("aifred_trading_controls", JSON.stringify({
          mode: newControls.mode,
          isRunning: newControls.isRunning,
          scanInterval: newControls.scanInterval,
          assets: newControls.assets,
        }));
      } catch { /* ignore */ }
    },
    [controls]
  );

  const toggleAsset = useCallback(
    (symbol: string) => {
      const newAssets = controls.assets.map((a) =>
        a.symbol === symbol ? { ...a, enabled: !a.enabled } : a
      );
      updateControls({ assets: newAssets });
    },
    [controls.assets, updateControls]
  );

  const handleModeSwitch = useCallback(
    (mode: "paper" | "live") => {
      if (mode === "live" && controls.mode === "paper") {
        setShowLiveWarning(true);
      } else {
        updateControls({ mode });
      }
    },
    [controls.mode, updateControls]
  );

  const groupedAssets = controls.assets.reduce(
    (acc, asset) => {
      if (!acc[asset.category]) acc[asset.category] = [];
      acc[asset.category].push(asset);
      return acc;
    },
    {} as Record<string, typeof controls.assets>
  );

  return (
    <div
      className="min-h-screen bg-[#06060a] text-white noise-bg relative overflow-hidden"
      style={{ fontFamily: "Outfit, sans-serif" }}
    >
      {/* Ambient glow */}
      <div className="fixed inset-0 pointer-events-none">
        <div
          className="absolute top-0 left-1/4 w-[600px] h-[600px] rounded-full opacity-[0.03]"
          style={{
            background:
              "radial-gradient(circle, rgba(16,185,129,1) 0%, transparent 70%)",
          }}
        />
        <div
          className="absolute bottom-0 right-1/4 w-[400px] h-[400px] rounded-full opacity-[0.02]"
          style={{
            background:
              "radial-gradient(circle, rgba(99,102,241,1) 0%, transparent 70%)",
          }}
        />
      </div>

      {/* ─── Header ─────────────────────────────────────────── */}
      <header className="relative z-10 border-b border-white/[0.06] px-6 py-4">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => router.push("/trading")}
              className="w-9 h-9 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center hover:bg-white/[0.08] transition-colors"
            >
              <ArrowLeft className="w-4 h-4 text-zinc-400" />
            </button>
            <div>
              <h1
                className="text-lg font-bold tracking-tight"
                style={{ fontFamily: "Outfit, sans-serif" }}
              >
                Settings
              </h1>
              <p
                className="text-[11px] text-zinc-500 tracking-[0.2em] uppercase"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                BROKER CONNECTIONS & CONFIGURATION
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full bg-emerald-400"
              style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
            />
            <span
              className="text-xs text-emerald-400/80 tracking-wider"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              SYSTEM ONLINE
            </span>
          </div>
        </div>
      </header>

      {/* ─── Content ────────────────────────────────────────── */}
      <main className="relative z-10 max-w-[1600px] mx-auto px-6 py-6 space-y-6 pb-16">
        {/* Section A: Broker Connections */}
        <SectionHeader
          icon={<Wifi className="w-4 h-4" />}
          title="Broker Connections"
          subtitle="MANAGE EXCHANGE & BROKER INTEGRATIONS"
          delay={0}
        />
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
        >
          {brokers.map((broker, i) => (
            <BrokerCard
              key={broker.id}
              broker={broker}
              onConnect={() => {
                setSelectedBroker(broker);
                setCredentials({});
                setTestResult(null);
              }}
              delay={i * 0.04}
            />
          ))}
        </motion.div>

        {/* Section B: Trading Controls */}
        <SectionHeader
          icon={<Zap className="w-4 h-4" />}
          title="Trading Controls"
          subtitle="MODE, SCAN INTERVAL & ASSET SELECTION"
          delay={0.2}
        />
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="grid lg:grid-cols-2 gap-4"
        >
          {/* Mode & Interval */}
          <div className="card-glass rounded-2xl p-6 space-y-6">
            {/* Mode Toggle */}
            <div>
              <label
                className="text-[10px] text-zinc-500 uppercase tracking-[0.15em] block mb-3"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                Trading Mode
              </label>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => handleModeSwitch("paper")}
                  className={`flex-1 py-3 rounded-xl text-sm font-medium transition-all ${
                    controls.mode === "paper"
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                      : "bg-white/[0.03] text-zinc-500 border border-white/[0.06] hover:text-zinc-300"
                  }`}
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  PAPER
                </button>
                <button
                  onClick={() => handleModeSwitch("live")}
                  className={`flex-1 py-3 rounded-xl text-sm font-medium transition-all ${
                    controls.mode === "live"
                      ? "bg-red-500/20 text-red-400 border border-red-500/30"
                      : "bg-white/[0.03] text-zinc-500 border border-white/[0.06] hover:text-zinc-300"
                  }`}
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  LIVE
                </button>
              </div>
              {controls.mode === "live" && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="mt-2 flex items-center gap-2 text-[11px] text-amber-400"
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  <AlertTriangle className="w-3 h-3" />
                  <span>REAL MONEY MODE ACTIVE</span>
                </motion.div>
              )}
            </div>

            {/* Scan Interval */}
            <div>
              <label
                className="text-[10px] text-zinc-500 uppercase tracking-[0.15em] block mb-3"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                Scan Interval
              </label>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 flex-1 bg-white/[0.03] border border-white/[0.06] rounded-xl px-4 py-2.5">
                  <Clock className="w-4 h-4 text-zinc-500" />
                  <input
                    type="number"
                    min={5}
                    max={300}
                    value={controls.scanInterval}
                    onChange={(e) =>
                      updateControls({
                        scanInterval: parseInt(e.target.value) || 30,
                      })
                    }
                    className="bg-transparent text-sm text-zinc-200 outline-none w-full"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  />
                </div>
                <span
                  className="text-xs text-zinc-600"
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  seconds
                </span>
              </div>
            </div>

            {/* Start/Stop */}
            <div>
              <label
                className="text-[10px] text-zinc-500 uppercase tracking-[0.15em] block mb-3"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                Engine Status
              </label>
              <button
                onClick={() =>
                  updateControls({ isRunning: !controls.isRunning })
                }
                className={`w-full py-3.5 rounded-xl text-sm font-semibold tracking-wider transition-all flex items-center justify-center gap-2 ${
                  controls.isRunning
                    ? "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30"
                    : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30"
                }`}
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {controls.isRunning ? (
                  <>
                    <Square className="w-4 h-4" />
                    STOP TRADING
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    START TRADING
                  </>
                )}
                {controls.isRunning && (
                  <span className="relative flex h-2 w-2 ml-1">
                    <span
                      className="absolute inline-flex h-full w-full rounded-full bg-red-400"
                      style={{
                        animation: "pulse-ring 1.5s cubic-bezier(0, 0, 0.2, 1) infinite",
                      }}
                    />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-red-400" />
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Asset Selection */}
          <div className="card-glass rounded-2xl p-6">
            <label
              className="text-[10px] text-zinc-500 uppercase tracking-[0.15em] block mb-4"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              Asset Selection
            </label>
            <div className="space-y-5">
              {Object.entries(groupedAssets).map(([category, assets]) => (
                <div key={category}>
                  <span
                    className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded ${CATEGORY_COLORS[category] || "text-zinc-400 bg-zinc-400/10"}`}
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {category}
                  </span>
                  <div className="mt-2.5 space-y-1.5">
                    {assets.map((asset) => (
                      <button
                        key={asset.symbol}
                        onClick={() => toggleAsset(asset.symbol)}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
                          asset.enabled
                            ? "bg-emerald-500/[0.06] border border-emerald-500/20"
                            : "bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.04]"
                        }`}
                      >
                        <span
                          className={`text-xs font-medium ${
                            asset.enabled ? "text-zinc-200" : "text-zinc-500"
                          }`}
                          style={{ fontFamily: "JetBrains Mono, monospace" }}
                        >
                          {asset.symbol}
                        </span>
                        <div
                          className={`w-8 h-4.5 rounded-full relative transition-colors ${
                            asset.enabled ? "bg-emerald-500" : "bg-zinc-700"
                          }`}
                          style={{ height: "18px" }}
                        >
                          <div
                            className="absolute top-0.5 w-3.5 h-3.5 rounded-full bg-white shadow-sm transition-transform"
                            style={{
                              height: "14px",
                              width: "14px",
                              transform: asset.enabled
                                ? "translateX(16px)"
                                : "translateX(2px)",
                            }}
                          />
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Section C: Risk Parameters */}
        <SectionHeader
          icon={<Shield className="w-4 h-4" />}
          title="Risk Parameters"
          subtitle="CURRENT RISK MANAGEMENT CONFIGURATION"
          delay={0.35}
        />
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card-glass rounded-2xl p-6"
        >
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {RISK_PARAMS.map((param) => {
              const Icon = param.icon;
              return (
                <div
                  key={param.label}
                  className="text-center p-4 bg-white/[0.02] rounded-xl border border-white/[0.04]"
                >
                  <Icon className="w-4 h-4 text-zinc-500 mx-auto mb-2" />
                  <div
                    className="text-lg font-bold text-emerald-400 glow-green"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {param.value}
                  </div>
                  <div
                    className="text-[10px] text-zinc-600 mt-1 uppercase tracking-wider"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {param.label}
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* Section D: API & Notifications */}
        <SectionHeader
          icon={<Key className="w-4 h-4" />}
          title="API & Notifications"
          subtitle="EXTERNAL SERVICE INTEGRATIONS"
          delay={0.45}
        />
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="grid lg:grid-cols-2 gap-4"
        >
          {/* Telegram */}
          <div className="card-glass rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2 mb-1">
              <Send className="w-4 h-4 text-zinc-500" />
              <span className="text-sm font-semibold text-zinc-300">
                Telegram Notifications
              </span>
            </div>
            <ApiInput
              label="Bot Token"
              value={notifications.telegramBotToken}
              onChange={(v) =>
                setNotifications((p) => ({ ...p, telegramBotToken: v }))
              }
              placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u..."
              type="password"
            />
            <ApiInput
              label="Chat ID"
              value={notifications.telegramChatId}
              onChange={(v) =>
                setNotifications((p) => ({ ...p, telegramChatId: v }))
              }
              placeholder="-1001234567890"
              type="text"
            />
          </div>

          {/* API Keys */}
          <div className="card-glass rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-2 mb-1">
              <Brain className="w-4 h-4 text-zinc-500" />
              <span className="text-sm font-semibold text-zinc-300">
                API Keys
              </span>
            </div>
            <ApiInput
              label="Anthropic API Key"
              value={notifications.anthropicApiKey}
              onChange={(v) =>
                setNotifications((p) => ({ ...p, anthropicApiKey: v }))
              }
              placeholder="sk-ant-api03-..."
              type="password"
            />
            <ApiInput
              label="NewsAPI Key"
              value={notifications.newsApiKey}
              onChange={(v) =>
                setNotifications((p) => ({ ...p, newsApiKey: v }))
              }
              placeholder="your-newsapi-key"
              type="password"
            />
          </div>
        </motion.div>
      </main>

      {/* ─── Connection Modal ───────────────────────────────── */}
      <AnimatePresence>
        {selectedBroker && (
          <ConnectionModal
            broker={selectedBroker}
            credentials={credentials}
            setCredentials={setCredentials}
            testing={testing}
            testResult={testResult}
            saving={saving}
            onTest={handleTestConnection}
            onSave={handleSaveConnection}
            onClose={() => {
              setSelectedBroker(null);
              setCredentials({});
              setTestResult(null);
            }}
          />
        )}
      </AnimatePresence>

      {/* ─── Live Mode Warning Modal ────────────────────────── */}
      <AnimatePresence>
        {showLiveWarning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            onClick={() => setShowLiveWarning(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="card-glass rounded-2xl p-6 w-full max-w-md mx-4 border-red-500/20"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-zinc-200">
                    Switch to Live Trading?
                  </h3>
                  <p
                    className="text-[11px] text-zinc-500"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    THIS ACTION USES REAL FUNDS
                  </p>
                </div>
              </div>
              <p className="text-xs text-zinc-400 leading-relaxed mb-6">
                You are about to switch to live trading mode. All trades will be
                executed with real money on connected brokers. Ensure your risk
                parameters are properly configured before proceeding.
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() => setShowLiveWarning(false)}
                  className="flex-1 py-2.5 rounded-xl text-xs font-medium text-zinc-400 bg-white/[0.04] border border-white/[0.06] hover:bg-white/[0.08] transition-colors"
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  CANCEL
                </button>
                <button
                  onClick={() => {
                    updateControls({ mode: "live" });
                    setShowLiveWarning(false);
                  }}
                  className="flex-1 py-2.5 rounded-xl text-xs font-medium text-red-400 bg-red-500/20 border border-red-500/30 hover:bg-red-500/30 transition-colors"
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  CONFIRM LIVE MODE
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════

function SectionHeader({
  icon,
  title,
  subtitle,
  delay,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="flex items-center gap-3 pt-2"
    >
      <span className="text-zinc-500">{icon}</span>
      <div>
        <h2 className="text-sm font-semibold text-zinc-300">{title}</h2>
        <p
          className="text-[11px] text-zinc-600 mt-0.5"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {subtitle}
        </p>
      </div>
    </motion.div>
  );
}

function BrokerCard({
  broker,
  onConnect,
  delay,
}: {
  broker: Broker;
  onConnect: () => void;
  delay: number;
}) {
  const statusColor =
    broker.status === "connected"
      ? "bg-emerald-400"
      : broker.status === "error"
      ? "bg-red-400"
      : "bg-zinc-600";

  const statusText =
    broker.status === "connected"
      ? "Connected"
      : broker.status === "error"
      ? "Error"
      : "Disconnected";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card-glass rounded-2xl p-5 group relative"
    >
      {broker.comingSoon && (
        <div className="absolute top-3 right-3">
          <span
            className="text-[9px] text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded-full uppercase tracking-wider"
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            Coming Soon
          </span>
        </div>
      )}

      <div className="flex items-start gap-3 mb-3">
        <div className="w-10 h-10 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-zinc-400">
          {BROKER_ICONS[broker.icon]}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-zinc-200">
            {broker.name}
          </h3>
          <span
            className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded ${CATEGORY_COLORS[broker.category]}`}
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            {broker.category}
          </span>
        </div>
      </div>

      <p className="text-[11px] text-zinc-500 mb-4 leading-relaxed">
        {broker.description}
      </p>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <div
            className={`w-1.5 h-1.5 rounded-full ${statusColor}`}
            style={
              broker.status === "connected"
                ? { animation: "pulse-glow 2s ease-in-out infinite" }
                : undefined
            }
          />
          <span
            className={`text-[10px] tracking-wider ${
              broker.status === "connected"
                ? "text-emerald-400/70"
                : broker.status === "error"
                ? "text-red-400/70"
                : "text-zinc-600"
            }`}
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            {statusText}
          </span>
        </div>

        {!broker.comingSoon && (
          <button
            onClick={onConnect}
            className={`text-[11px] font-medium px-3 py-1.5 rounded-lg transition-all ${
              broker.status === "connected"
                ? "text-zinc-400 bg-white/[0.04] hover:bg-white/[0.08]"
                : "text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20"
            }`}
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            {broker.status === "connected" ? "CONFIGURE" : "CONNECT"}
          </button>
        )}
      </div>
    </motion.div>
  );
}

function ConnectionModal({
  broker,
  credentials,
  setCredentials,
  testing,
  testResult,
  saving,
  onTest,
  onSave,
  onClose,
}: {
  broker: Broker;
  credentials: Record<string, string>;
  setCredentials: (c: Record<string, string>) => void;
  testing: boolean;
  testResult: {
    success: boolean;
    latency?: number;
    accountInfo?: { label: string; value: string }[];
    error?: string;
  } | null;
  saving: boolean;
  onTest: () => void;
  onSave: () => void;
  onClose: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0, y: 20 }}
        animate={{ scale: 1, opacity: 1, y: 0 }}
        exit={{ scale: 0.95, opacity: 0, y: 20 }}
        transition={{ duration: 0.2 }}
        onClick={(e) => e.stopPropagation()}
        className="card-glass rounded-2xl p-6 w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-zinc-400">
              {BROKER_ICONS[broker.icon]}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-zinc-200">
                Connect to {broker.name}
              </h3>
              <p
                className="text-[11px] text-zinc-500 mt-0.5"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {broker.description}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Credential Fields */}
        <div className="space-y-4 mb-6">
          {broker.requiredCredentials.map((field) => (
            <div key={field.key}>
              <label
                className="text-[10px] text-zinc-500 uppercase tracking-[0.15em] block mb-1.5"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {field.label}
              </label>
              <input
                type={field.type}
                value={credentials[field.key] || ""}
                onChange={(e) =>
                  setCredentials({
                    ...credentials,
                    [field.key]: e.target.value,
                  })
                }
                className="w-full bg-white/[0.03] border border-white/[0.06] rounded-xl px-4 py-2.5 text-sm text-zinc-200 outline-none focus:border-emerald-500/30 transition-colors placeholder:text-zinc-700"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
                placeholder={`Enter ${field.label.toLowerCase()}`}
              />
            </div>
          ))}
        </div>

        {/* Test Result */}
        <AnimatePresence>
          {testResult && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-4"
            >
              <div
                className={`rounded-xl p-4 border ${
                  testResult.success
                    ? "bg-emerald-500/[0.06] border-emerald-500/20"
                    : "bg-red-500/[0.06] border-red-500/20"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  {testResult.success ? (
                    <Check className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <X className="w-4 h-4 text-red-400" />
                  )}
                  <span
                    className={`text-xs font-medium ${
                      testResult.success ? "text-emerald-400" : "text-red-400"
                    }`}
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {testResult.success
                      ? "CONNECTION SUCCESSFUL"
                      : "CONNECTION FAILED"}
                  </span>
                </div>
                {testResult.latency && (
                  <p
                    className="text-[11px] text-zinc-500"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    Latency: {testResult.latency}ms
                  </p>
                )}
                {testResult.accountInfo && (
                  <div className="mt-2 space-y-1">
                    {testResult.accountInfo.map((info) => (
                      <div
                        key={info.label}
                        className="flex justify-between text-[11px]"
                        style={{ fontFamily: "JetBrains Mono, monospace" }}
                      >
                        <span className="text-zinc-500">{info.label}</span>
                        <span className="text-zinc-300">{info.value}</span>
                      </div>
                    ))}
                  </div>
                )}
                {testResult.error && (
                  <p
                    className="text-[11px] text-red-400/80 mt-1"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {testResult.error}
                  </p>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onTest}
            disabled={
              testing ||
              broker.requiredCredentials.some((f) => !credentials[f.key])
            }
            className="flex-1 py-2.5 rounded-xl text-xs font-medium transition-all flex items-center justify-center gap-2 text-zinc-300 bg-white/[0.04] border border-white/[0.06] hover:bg-white/[0.08] disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            {testing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Wifi className="w-3.5 h-3.5" />
            )}
            {testing ? "TESTING..." : "TEST CONNECTION"}
          </button>
          <button
            onClick={onSave}
            disabled={
              saving ||
              !testResult?.success ||
              broker.requiredCredentials.some((f) => !credentials[f.key])
            }
            className="flex-1 py-2.5 rounded-xl text-xs font-medium transition-all flex items-center justify-center gap-2 text-emerald-400 bg-emerald-500/20 border border-emerald-500/30 hover:bg-emerald-500/30 disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            {saving ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Check className="w-3.5 h-3.5" />
            )}
            {saving ? "SAVING..." : "SAVE & CONNECT"}
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}

function ApiInput({
  label,
  value,
  onChange,
  placeholder,
  type,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  type: string;
}) {
  return (
    <div>
      <label
        className="text-[10px] text-zinc-500 uppercase tracking-[0.15em] block mb-1.5"
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-white/[0.03] border border-white/[0.06] rounded-xl px-4 py-2.5 text-sm text-zinc-200 outline-none focus:border-emerald-500/30 transition-colors placeholder:text-zinc-700"
        style={{ fontFamily: "JetBrains Mono, monospace" }}
        placeholder={placeholder}
      />
    </div>
  );
}

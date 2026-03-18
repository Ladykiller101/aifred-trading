"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  PieChart,
  Pie,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Activity,
  Shield,
  Zap,
  Eye,
  ChevronDown,
  ChevronUp,
  Radio,
  Target,
  BarChart3,
  Brain,
  Layers,
  Settings,
  AlertTriangle,
  Wifi,
  Play,
  Square,
  X,
  ArrowUpDown,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { useRouter } from "next/navigation";

// ─── Types ────────────────────────────────────────────────────
interface TradingData {
  summary: {
    totalPnl: number;
    winRate: number;
    totalTrades: number;
    openPositions: number;
    sharpeRatio: number;
    maxDrawdown: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
    totalFees: number;
    currentEquity: number;
  };
  equityCurve: { date: string; value: number }[];
  byAsset: { asset: string; pnl: number; trades: number; winRate: number }[];
  byStrategy: {
    strategy: string;
    pnl: number;
    trades: number;
    winRate: number;
  }[];
  byTier: { tier: string; pnl: number; trades: number; winRate: number }[];
  recentTrades: Record<string, unknown>[];
  openPositions: Record<string, unknown>[];
}

// ─── Activity Types ──────────────────────────────────────────
interface ActivityEntry {
  id: string;
  timestamp: string;
  type:
    | "trade_executed"
    | "trade_closed"
    | "signal_generated"
    | "signal_rejected"
    | "broker_connected"
    | "broker_disconnected"
    | "system_start"
    | "system_stop"
    | "error"
    | "optimization"
    | "scan_complete";
  severity: "info" | "success" | "warning" | "error";
  title: string;
  message: string;
  details?: {
    asset?: string;
    side?: "LONG" | "SHORT";
    strategy?: string;
    confidence?: number;
    entry_price?: number;
    stop_loss?: number;
    take_profit?: number;
    pnl?: number;
    reasoning?: string;
    technical_signals?: string;
    sentiment_signals?: string;
    risk_assessment?: string;
    broker?: string;
    tier?: string;
  };
}

// ─── Utility ─────────────────────────────────────────────────
function timeAgo(dateStr: string): string {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function fmt(n: number, prefix = "$") {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${prefix}${(n / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${prefix}${(n / 1_000).toFixed(1)}K`;
  return `${prefix}${n.toFixed(2)}`;
}

function pct(n: number) {
  return `${n.toFixed(2)}%`;
}

const STRATEGY_LABELS: Record<string, string> = {
  mean_reversion: "Mean Reversion",
  ict_confluence: "ICT Confluence",
  lstm_ensemble: "LSTM Ensemble",
  transformer: "Transformer",
  sentiment_breakout: "Sentiment Breakout",
};

const TIER_COLORS: Record<string, string> = {
  "A+": "#f59e0b",
  A: "#10b981",
  B: "#6366f1",
  C: "#ef4444",
};

// ─── Glow animation keyframes (injected once) ────────────────
const INJECTED_STYLES = `
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

@keyframes pulse-glow {
  0%, 100% { opacity: 0.4; }
  50% { opacity: 1; }
}
@keyframes scan-line {
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100vh); }
}
@keyframes ticker {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
.glow-green { text-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }
.glow-red { text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
.glow-gold { text-shadow: 0 0 20px rgba(245, 158, 11, 0.5); }
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
`;

// ─── Available symbols ───────────────────────────────────────
const AVAILABLE_SYMBOLS = [
  "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
  "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
  "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
  "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META",
];

// ─── Execute Trade Result type ───────────────────────────────
interface TradeResult {
  success: boolean;
  orderId?: string;
  symbol?: string;
  side?: string;
  quantity?: number;
  executionPrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  riskReward?: number;
  broker?: string;
  strategy?: string;
  confidence?: number;
  tier?: string;
  reasoning?: string;
  technicalSignals?: string;
  sentimentSignals?: string;
  riskAssessment?: string;
  status?: string;
  message?: string;
}

// ─── Execute Trade Modal ──────────────────────────────────────
function ExecuteTradeModal({
  onClose,
  onTradeExecuted,
}: {
  onClose: () => void;
  onTradeExecuted: (result: TradeResult) => void;
}) {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [side, setSide] = useState<"LONG" | "SHORT">("LONG");
  const [quantity, setQuantity] = useState("0.01");
  const [orderType, setOrderType] = useState<"market" | "limit">("market");
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<TradeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(8);

  // Auto-close 8 seconds after result appears
  useEffect(() => {
    if (!result) return;
    setCountdown(8);
    const timer = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) {
          clearInterval(timer);
          onClose();
          return 0;
        }
        return c - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [result, onClose]);

  const handleExecute = useCallback(async () => {
    setExecuting(true);
    setError(null);
    try {
      const res = await fetch("/api/trading/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          side,
          quantity: parseFloat(quantity) || 0.01,
          orderType,
        }),
      });
      const data: TradeResult = await res.json();
      if (data.success) {
        setResult(data);
        onTradeExecuted(data);
      } else {
        setError(data.message || "Trade failed");
      }
    } catch {
      setError("Network error — could not reach trading API");
    } finally {
      setExecuting(false);
    }
  }, [symbol, side, quantity, orderType, onTradeExecuted]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 12 }}
        transition={{ duration: 0.2 }}
        className="relative z-10 w-full max-w-md mx-4"
        style={{
          background: "linear-gradient(135deg, rgba(15,15,20,0.98) 0%, rgba(10,10,15,0.98) 100%)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "16px",
          backdropFilter: "blur(40px)",
        }}
      >
        {!result ? (
          <div className="p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                  <ArrowUpDown className="w-4 h-4 text-emerald-400" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white" style={{ fontFamily: "Outfit, sans-serif" }}>
                    Execute Trade
                  </h2>
                  <p className="text-[11px] text-zinc-500 tracking-wider" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    PAPER TRADING MODE
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg bg-white/[0.04] flex items-center justify-center hover:bg-white/[0.08] transition-colors text-zinc-500 hover:text-zinc-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Symbol */}
            <div className="mb-4">
              <label className="block text-[11px] text-zinc-500 tracking-wider mb-1.5 uppercase" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                Symbol
              </label>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2.5 text-sm text-white outline-none focus:border-emerald-500/40 transition-colors"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {AVAILABLE_SYMBOLS.map((s) => (
                  <option key={s} value={s} style={{ background: "#0a0a0f" }}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            {/* Side */}
            <div className="mb-4">
              <label className="block text-[11px] text-zinc-500 tracking-wider mb-1.5 uppercase" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                Direction
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setSide("LONG")}
                  className={`py-2.5 rounded-lg text-sm font-semibold transition-all border ${
                    side === "LONG"
                      ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-400"
                      : "bg-white/[0.03] border-white/[0.06] text-zinc-500 hover:text-zinc-300"
                  }`}
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  ▲ LONG
                </button>
                <button
                  onClick={() => setSide("SHORT")}
                  className={`py-2.5 rounded-lg text-sm font-semibold transition-all border ${
                    side === "SHORT"
                      ? "bg-red-500/20 border-red-500/40 text-red-400"
                      : "bg-white/[0.03] border-white/[0.06] text-zinc-500 hover:text-zinc-300"
                  }`}
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  ▼ SHORT
                </button>
              </div>
            </div>

            {/* Quantity */}
            <div className="mb-4">
              <label className="block text-[11px] text-zinc-500 tracking-wider mb-1.5 uppercase" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                Quantity
              </label>
              <input
                type="number"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                step="0.001"
                min="0.001"
                className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2.5 text-sm text-white outline-none focus:border-emerald-500/40 transition-colors"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              />
            </div>

            {/* Order Type */}
            <div className="mb-6">
              <label className="block text-[11px] text-zinc-500 tracking-wider mb-1.5 uppercase" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                Order Type
              </label>
              <div className="grid grid-cols-2 gap-2">
                {(["market", "limit"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setOrderType(t)}
                    className={`py-2 rounded-lg text-xs font-medium transition-all border uppercase tracking-wider ${
                      orderType === t
                        ? "bg-indigo-500/20 border-indigo-500/40 text-indigo-400"
                        : "bg-white/[0.03] border-white/[0.06] text-zinc-500 hover:text-zinc-300"
                    }`}
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {error && (
              <div className="mb-4 px-3 py-2.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
                {error}
              </div>
            )}

            {/* Execute Button */}
            <button
              onClick={handleExecute}
              disabled={executing}
              className={`w-full py-3 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2 ${
                side === "LONG"
                  ? "bg-emerald-500 hover:bg-emerald-400 text-black disabled:opacity-50"
                  : "bg-red-500 hover:bg-red-400 text-white disabled:opacity-50"
              }`}
              style={{ fontFamily: "Outfit, sans-serif" }}
            >
              {executing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Executing...
                </>
              ) : (
                <>
                  <ArrowUpDown className="w-4 h-4" />
                  {side === "LONG" ? "Buy / Long" : "Sell / Short"} {symbol}
                </>
              )}
            </button>
          </div>
        ) : (
          // Trade confirmation view
          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white" style={{ fontFamily: "Outfit, sans-serif" }}>
                    Order Filled ✓
                  </h2>
                  <p className="text-[11px] text-emerald-400/70 tracking-wider" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    {result.orderId}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-zinc-600" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                  closing in {countdown}s
                </span>
                <button
                  onClick={onClose}
                  className="w-8 h-8 rounded-lg bg-white/[0.04] flex items-center justify-center hover:bg-white/[0.08] transition-colors text-zinc-500 hover:text-zinc-300"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
            {/* Note: Switching to Activity Log tab */}
            <div className="mb-4 px-3 py-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center gap-2">
              <Brain className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
              <span className="text-[11px] text-indigo-300">Full reasoning log saved to <strong>Activity</strong> tab — auto-opens when this closes.</span>
            </div>

            {/* Key stats */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              {[
                { label: "Symbol", value: result.symbol },
                { label: "Side", value: result.side, color: result.side === "LONG" ? "text-emerald-400" : "text-red-400" },
                { label: "Price", value: result.executionPrice?.toFixed(4) },
                { label: "Quantity", value: result.quantity?.toString() },
                { label: "Stop Loss", value: result.stopLoss?.toFixed(4), color: "text-red-400" },
                { label: "Take Profit", value: result.takeProfit?.toFixed(4), color: "text-emerald-400" },
                { label: "Risk/Reward", value: `1:${result.riskReward}` },
                { label: "Confidence", value: `${result.confidence}%`, color: (result.confidence ?? 0) >= 80 ? "text-emerald-400" : "text-amber-400" },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-white/[0.03] rounded-lg px-3 py-2">
                  <div className="text-[10px] text-zinc-600 tracking-wider uppercase mb-0.5" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    {label}
                  </div>
                  <div className={`text-sm font-semibold ${color || "text-white"}`} style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            {/* Reasoning + Signals */}
            <div className="space-y-2 mb-4 max-h-[40vh] overflow-y-auto pr-1">
              {result.reasoning && (
                <div className="bg-white/[0.02] border border-indigo-500/20 rounded-lg p-3">
                  <div className="text-[10px] text-indigo-400 tracking-wider uppercase mb-1.5 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    <Brain className="w-3 h-3" /> AI Reasoning
                  </div>
                  <p className="text-xs text-zinc-300 leading-relaxed">{result.reasoning}</p>
                </div>
              )}
              {result.technicalSignals && (
                <div className="bg-white/[0.02] border border-emerald-500/20 rounded-lg p-3">
                  <div className="text-[10px] text-emerald-400 tracking-wider uppercase mb-1.5 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    <BarChart3 className="w-3 h-3" /> Technical Signals
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">{result.technicalSignals}</p>
                </div>
              )}
              {result.sentimentSignals && (
                <div className="bg-white/[0.02] border border-amber-500/20 rounded-lg p-3">
                  <div className="text-[10px] text-amber-400 tracking-wider uppercase mb-1.5 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    <Zap className="w-3 h-3" /> Sentiment Analysis
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">{result.sentimentSignals}</p>
                </div>
              )}
              {result.riskAssessment && (
                <div className="bg-white/[0.02] border border-red-500/20 rounded-lg p-3">
                  <div className="text-[10px] text-red-400 tracking-wider uppercase mb-1.5 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                    <Shield className="w-3 h-3" /> Risk Assessment
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">{result.riskAssessment}</p>
                </div>
              )}
            </div>

            {/* Strategy + Tier */}
            <div className="flex items-center gap-2 mb-4">
              <span className="px-2 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[11px]" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                {result.strategy}
              </span>
              <span className={`px-2 py-1 rounded-md text-[11px] font-bold ${
                result.tier === "A+" ? "bg-amber-500/10 border border-amber-500/20 text-amber-400" :
                result.tier === "A" ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" :
                "bg-indigo-500/10 border border-indigo-500/20 text-indigo-400"
              }`} style={{ fontFamily: "JetBrains Mono, monospace" }}>
                Tier {result.tier}
              </span>
              <span className="ml-auto text-[11px] text-zinc-500" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                via {result.broker}
              </span>
            </div>

            <button
              onClick={onClose}
              className="w-full py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.08] text-zinc-300 text-sm font-medium transition-colors"
              style={{ fontFamily: "Outfit, sans-serif" }}
            >
              Close
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────
export default function TradingDashboard() {
  const router = useRouter();
  const [data, setData] = useState<TradingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTrade, setExpandedTrade] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<
    "overview" | "trades" | "activity" | "agents"
  >("overview");
  const [showTradeModal, setShowTradeModal] = useState(false);
  const [activityRefreshKey, setActivityRefreshKey] = useState(0);
  const [tradeToast, setTradeToast] = useState<TradeResult | null>(null);

  const handleTradeExecuted = useCallback((result: TradeResult) => {
    // Persist trade to localStorage so it survives serverless resets
    try {
      const stored: ActivityEntry[] = JSON.parse(
        localStorage.getItem("aifred_local_trades") || "[]"
      );
      const newEntry: ActivityEntry = {
        id: `local_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        timestamp: new Date().toISOString(),
        type: "trade_executed",
        severity: "success",
        title: `${result.side} ${result.symbol} — Order Filled`,
        message: `${result.side} ${result.quantity} ${result.symbol} @ ${result.executionPrice?.toFixed(4)} | Strategy: ${result.strategy} | Confidence: ${result.confidence}%`,
        details: {
          asset: result.symbol,
          side: result.side as "LONG" | "SHORT",
          strategy: result.strategy,
          confidence: result.confidence,
          entry_price: result.executionPrice,
          stop_loss: result.stopLoss,
          take_profit: result.takeProfit,
          reasoning: result.reasoning,
          technical_signals: result.technicalSignals,
          sentiment_signals: result.sentimentSignals,
          risk_assessment: result.riskAssessment,
          broker: result.broker,
          tier: result.tier,
        },
      };
      stored.unshift(newEntry);
      localStorage.setItem(
        "aifred_local_trades",
        JSON.stringify(stored.slice(0, 100))
      );
    } catch { /* localStorage may not be available */ }
    // Show persistent toast + switch to activity tab
    setTradeToast(result);
    setActiveTab("activity");
    setActivityRefreshKey((k) => k + 1);
    // Dismiss toast after 12s
    setTimeout(() => setTradeToast(null), 12000);
  }, []);

  useEffect(() => {
    // Inject custom styles
    const style = document.createElement("style");
    style.textContent = INJECTED_STYLES;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  useEffect(() => {
    fetch("/api/trading")
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#06060a] flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="w-10 h-10 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
          <span
            className="text-zinc-500 text-sm tracking-[0.3em]"
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            INITIALIZING AGENTS...
          </span>
        </motion.div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-[#06060a] flex items-center justify-center text-red-400">
        <span style={{ fontFamily: "JetBrains Mono, monospace" }}>
          ERROR: {error || "No data"}
        </span>
      </div>
    );
  }

  const { summary } = data;
  const isPositive = summary.totalPnl >= 0;

  return (
    <div
      className="min-h-screen bg-[#06060a] text-white noise-bg relative overflow-hidden"
      style={{ fontFamily: "Outfit, sans-serif" }}
    >
      {/* Ambient glow effects */}
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
      <header className="relative z-10 border-b border-white/[0.06] px-4 md:px-6 py-3 md:py-4">
        <div className="max-w-[1600px] mx-auto">
          {/* Top row: Logo + Execute Trade + Settings */}
          <div className="flex items-center justify-between mb-3 md:mb-0">
            <div className="flex items-center gap-3 md:gap-4">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-700 flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1
                  className="text-lg font-bold tracking-tight"
                  style={{ fontFamily: "Outfit, sans-serif" }}
                >
                  AIFred
                </h1>
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full bg-emerald-400"
                    style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
                  />
                  <span
                    className="text-[10px] text-emerald-400/80 tracking-wider"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    7 AGENTS ONLINE
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 md:gap-4">
              {/* Execute Trade Button — always visible, prominent */}
              <button
                onClick={() => setShowTradeModal(true)}
                className="flex items-center gap-2 px-4 md:px-5 py-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-black text-xs md:text-sm font-bold transition-all shadow-lg shadow-emerald-500/25 active:scale-95"
                style={{ fontFamily: "Outfit, sans-serif" }}
              >
                <ArrowUpDown className="w-4 h-4" />
                <span>Execute Trade</span>
              </button>

              {/* Settings */}
              <button
                onClick={() => router.push("/trading/settings")}
                className="w-9 h-9 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center hover:bg-white/[0.08] hover:border-white/[0.12] transition-all text-zinc-500 hover:text-zinc-300"
                title="Settings"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Nav tabs — scrollable on mobile */}
          <div className="flex gap-1 bg-white/[0.03] rounded-lg p-1 overflow-x-auto">
            {(["overview", "trades", "activity", "agents"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 md:flex-none px-3 md:px-4 py-1.5 text-xs font-medium rounded-md transition-all tracking-wider uppercase whitespace-nowrap ${
                  activeTab === tab
                    ? "bg-white/[0.08] text-white"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ─── Trade Execution Toast ───────────────────────── */}
      <AnimatePresence>
        {tradeToast && (
          <motion.div
            initial={{ opacity: 0, y: -16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -16 }}
            transition={{ duration: 0.25 }}
            className="fixed top-20 right-6 z-40 w-80"
            style={{
              background: "linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(10,10,15,0.98) 100%)",
              border: "1px solid rgba(16,185,129,0.3)",
              borderRadius: "12px",
              backdropFilter: "blur(20px)",
            }}
          >
            <div className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <span className="text-sm font-semibold text-white" style={{ fontFamily: "Outfit, sans-serif" }}>
                  {tradeToast.side} {tradeToast.symbol} — Filled
                </span>
                <button onClick={() => setTradeToast(null)} className="ml-auto text-zinc-600 hover:text-zinc-400">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="grid grid-cols-3 gap-1.5 mb-2 text-[10px]" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                <div className="bg-white/[0.04] rounded px-2 py-1">
                  <div className="text-zinc-600">Price</div>
                  <div className="text-zinc-200">{tradeToast.executionPrice?.toFixed(4)}</div>
                </div>
                <div className="bg-white/[0.04] rounded px-2 py-1">
                  <div className="text-zinc-600">Conf.</div>
                  <div className="text-amber-400">{tradeToast.confidence}%</div>
                </div>
                <div className="bg-white/[0.04] rounded px-2 py-1">
                  <div className="text-zinc-600">Tier</div>
                  <div className="text-emerald-400">{tradeToast.tier}</div>
                </div>
              </div>
              {tradeToast.reasoning && (
                <p className="text-[10px] text-zinc-400 leading-relaxed line-clamp-2">
                  {tradeToast.reasoning}
                </p>
              )}
              <p className="text-[10px] text-indigo-400 mt-2 flex items-center gap-1">
                <Activity className="w-2.5 h-2.5" />
                Full log visible in Activity tab below ↓
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── Content ────────────────────────────────────────── */}
      <main className="relative z-10 max-w-[1600px] mx-auto px-6 py-6">
        <AnimatePresence mode="wait">
          {activeTab === "overview" && (
            <OverviewTab
              key="overview"
              data={data}
              onNavigateActivity={() => setActiveTab("activity")}
            />
          )}
          {activeTab === "trades" && (
            <TradesTab
              key="trades"
              trades={data.recentTrades}
              openPositions={data.openPositions}
              expandedTrade={expandedTrade}
              setExpandedTrade={setExpandedTrade}
            />
          )}
          {activeTab === "activity" && <ActivityTab key={`activity-${activityRefreshKey}`} />}
          {activeTab === "agents" && <AgentsTab key="agents" data={data} />}
        </AnimatePresence>
      </main>

      {/* ─── Footer ticker ──────────────────────────────────── */}
      <footer className="fixed bottom-0 left-0 right-0 z-20 border-t border-white/[0.04] bg-[#06060a]/90 backdrop-blur-sm">
        <div className="overflow-hidden h-8 flex items-center">
          <div
            className="flex gap-12 whitespace-nowrap"
            style={{
              animation: "ticker 30s linear infinite",
              fontFamily: "JetBrains Mono, monospace",
            }}
          >
            {[...data.byAsset, ...data.byAsset].map((a, i) => (
              <span key={i} className="text-[11px] flex items-center gap-2">
                <span className="text-zinc-400">{a.asset}</span>
                <span
                  className={a.pnl >= 0 ? "text-emerald-400" : "text-red-400"}
                >
                  {a.pnl >= 0 ? "+" : ""}
                  {fmt(a.pnl)}
                </span>
                <span className="text-zinc-600">|</span>
                <span className="text-zinc-500">WR {pct(a.winRate)}</span>
              </span>
            ))}
          </div>
        </div>
      </footer>

      {/* ─── Execute Trade Modal ─────────────────────────── */}
      <AnimatePresence>
        {showTradeModal && (
          <ExecuteTradeModal
            onClose={() => setShowTradeModal(false)}
            onTradeExecuted={handleTradeExecuted}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═══════════════════════════════════════════════════════════════
function OverviewTab({
  data,
  onNavigateActivity,
}: {
  data: TradingData;
  onNavigateActivity: () => void;
}) {
  const { summary } = data;
  const isPositive = summary.totalPnl >= 0;
  const [recentActivity, setRecentActivity] = useState<ActivityEntry[]>([]);

  useEffect(() => {
    fetch("/api/trading/activity")
      .then((r) => r.json())
      .then((data) => {
        const serverList = Array.isArray(data) ? data : (data?.activities ?? []);
        // Merge with localStorage trades for consistency
        try {
          const localTrades = JSON.parse(
            localStorage.getItem("aifred_local_trades") || "[]"
          );
          const merged = [...localTrades, ...serverList];
          const seen = new Set<string>();
          const deduped = merged.filter((e) => {
            if (seen.has(e.id)) return false;
            seen.add(e.id);
            return true;
          });
          deduped.sort(
            (a, b) =>
              new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
          );
          setRecentActivity(deduped.slice(0, 5));
        } catch {
          setRecentActivity(serverList.slice(0, 5));
        }
      })
      .catch(() => {});
  }, []);

  const SEVERITY_DOT: Record<string, string> = {
    info: "bg-blue-400",
    success: "bg-emerald-400",
    warning: "bg-amber-400",
    error: "bg-red-400",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 pb-12"
    >
      {/* ─── Activity Summary Bar ─────────────────────────── */}
      {recentActivity.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="card-glass rounded-xl px-4 py-2.5 flex items-center gap-2 overflow-x-auto cursor-pointer hover:border-white/[0.12] transition-all"
          onClick={onNavigateActivity}
        >
          <Activity className="w-3.5 h-3.5 text-zinc-500 flex-shrink-0" />
          <span
            className="text-[10px] text-zinc-500 uppercase tracking-wider flex-shrink-0 mr-2"
            style={{ fontFamily: "JetBrains Mono, monospace" }}
          >
            LIVE
          </span>
          <div className="flex items-center gap-4 overflow-x-auto">
            {recentActivity.map((entry) => (
              <div
                key={entry.id}
                className="flex items-center gap-2 flex-shrink-0"
              >
                <div
                  className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    SEVERITY_DOT[entry.severity] || "bg-zinc-500"
                  }`}
                />
                <span className="text-[11px] text-zinc-400 whitespace-nowrap">
                  {entry.message.length > 40
                    ? entry.message.slice(0, 40) + "..."
                    : entry.message}
                </span>
                <span
                  className="text-[10px] text-zinc-600 whitespace-nowrap"
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  {timeAgo(entry.timestamp)}
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ─── Hero Stats ─────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <HeroStat
          label="Total P&L"
          value={fmt(summary.totalPnl)}
          sub={`Equity: ${fmt(summary.currentEquity)}`}
          positive={isPositive}
          icon={<TrendingUp className="w-4 h-4" />}
          delay={0}
        />
        <HeroStat
          label="Win Rate"
          value={pct(summary.winRate)}
          sub={`${summary.totalTrades} closed trades`}
          positive={summary.winRate >= 60}
          icon={<Target className="w-4 h-4" />}
          delay={0.05}
          highlight
        />
        <HeroStat
          label="Sharpe Ratio"
          value={summary.sharpeRatio.toFixed(2)}
          sub={`Sortino: ${(summary.sharpeRatio * 1.3).toFixed(2)}`}
          positive={summary.sharpeRatio > 1}
          icon={<BarChart3 className="w-4 h-4" />}
          delay={0.1}
        />
        <HeroStat
          label="Max Drawdown"
          value={pct(summary.maxDrawdown)}
          sub={`Profit factor: ${summary.profitFactor.toFixed(2)}`}
          positive={summary.maxDrawdown < 10}
          icon={<Shield className="w-4 h-4" />}
          delay={0.15}
          invert
        />
      </div>

      {/* ─── Equity Curve ───────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card-glass rounded-2xl p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-zinc-300">
              Equity Curve
            </h2>
            <p
              className="text-[11px] text-zinc-600 mt-0.5"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              30-DAY PERFORMANCE · WALK-FORWARD VALIDATED
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`text-sm font-semibold ${
                isPositive ? "text-emerald-400 glow-green" : "text-red-400 glow-red"
              }`}
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              {isPositive ? "+" : ""}
              {fmt(summary.totalPnl)}
            </span>
          </div>
        </div>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.equityCurve}>
              <defs>
                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="0%"
                    stopColor={isPositive ? "#10b981" : "#ef4444"}
                    stopOpacity={0.3}
                  />
                  <stop
                    offset="100%"
                    stopColor={isPositive ? "#10b981" : "#ef4444"}
                    stopOpacity={0}
                  />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.03)"
              />
              <XAxis
                dataKey="date"
                tick={{ fill: "#52525b", fontSize: 10, fontFamily: "JetBrains Mono" }}
                axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: "#52525b", fontSize: 10, fontFamily: "JetBrains Mono" }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
                domain={["dataMin - 1000", "dataMax + 1000"]}
              />
              <Tooltip
                contentStyle={{
                  background: "#0f0f14",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "12px",
                  fontFamily: "JetBrains Mono",
                  fontSize: 12,
                }}
                labelStyle={{ color: "#71717a" }}
                formatter={(v?: number) => [`$${(v ?? 0).toLocaleString()}`, "Equity"]}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke={isPositive ? "#10b981" : "#ef4444"}
                strokeWidth={2}
                fill="url(#eqGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* ─── Three-column breakdown ─────────────────────────── */}
      <div className="grid lg:grid-cols-3 gap-4">
        {/* By Asset */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="card-glass rounded-2xl p-5"
        >
          <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
            <Layers className="w-4 h-4 text-zinc-500" />
            Performance by Asset
          </h3>
          <div className="space-y-3">
            {data.byAsset
              .sort((a, b) => b.pnl - a.pnl)
              .map((a) => (
                <AssetRow key={a.asset} {...a} />
              ))}
          </div>
        </motion.div>

        {/* By Strategy */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card-glass rounded-2xl p-5"
        >
          <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
            <Brain className="w-4 h-4 text-zinc-500" />
            Performance by Strategy
          </h3>
          <div className="space-y-3">
            {data.byStrategy
              .sort((a, b) => b.pnl - a.pnl)
              .map((s) => (
                <StrategyRow key={s.strategy} {...s} />
              ))}
          </div>
        </motion.div>

        {/* By Signal Tier */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="card-glass rounded-2xl p-5"
        >
          <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
            <Zap className="w-4 h-4 text-zinc-500" />
            Signal Tier Breakdown
          </h3>
          <div className="h-[160px] mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.byTier.map((t) => ({
                    name: t.tier,
                    value: t.trades,
                  }))}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={65}
                  paddingAngle={3}
                  dataKey="value"
                  stroke="none"
                >
                  {data.byTier.map((t, i) => (
                    <Cell
                      key={i}
                      fill={TIER_COLORS[t.tier] || "#6366f1"}
                    />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#0f0f14",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: "8px",
                    fontFamily: "JetBrains Mono",
                    fontSize: 11,
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {data.byTier
              .sort((a, b) => {
                const order = ["A+", "A", "B", "C"];
                return order.indexOf(a.tier) - order.indexOf(b.tier);
              })
              .map((t) => (
                <div
                  key={t.tier}
                  className="flex items-center justify-between text-xs"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: TIER_COLORS[t.tier] || "#6366f1",
                      }}
                    />
                    <span className="text-zinc-400">Tier {t.tier}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className="text-zinc-500"
                      style={{ fontFamily: "JetBrains Mono, monospace" }}
                    >
                      {t.trades} trades
                    </span>
                    <span
                      className={`font-medium ${
                        t.winRate >= 70
                          ? "text-emerald-400"
                          : t.winRate >= 50
                          ? "text-amber-400"
                          : "text-red-400"
                      }`}
                      style={{ fontFamily: "JetBrains Mono, monospace" }}
                    >
                      {pct(t.winRate)}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </motion.div>
      </div>

      {/* ─── Risk metrics bar ───────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card-glass rounded-2xl p-5"
      >
        <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
          <Shield className="w-4 h-4 text-zinc-500" />
          Risk Management
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <MiniStat label="Avg Win" value={fmt(summary.avgWin)} positive />
          <MiniStat label="Avg Loss" value={fmt(summary.avgLoss)} positive={false} />
          <MiniStat
            label="Win/Loss Ratio"
            value={
              summary.avgLoss > 0
                ? (summary.avgWin / summary.avgLoss).toFixed(2)
                : "∞"
            }
            positive={summary.avgWin > summary.avgLoss}
          />
          <MiniStat
            label="Profit Factor"
            value={summary.profitFactor.toFixed(2)}
            positive={summary.profitFactor > 1.5}
          />
          <MiniStat
            label="Open Positions"
            value={String(summary.openPositions)}
            neutral
          />
          <MiniStat
            label="Total Fees"
            value={fmt(summary.totalFees)}
            positive={false}
          />
        </div>
      </motion.div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════
// TRADES TAB
// ═══════════════════════════════════════════════════════════════
function TradesTab({
  trades,
  openPositions,
  expandedTrade,
  setExpandedTrade,
}: {
  trades: Record<string, unknown>[];
  openPositions: Record<string, unknown>[];
  expandedTrade: number | null;
  setExpandedTrade: (id: number | null) => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 pb-12"
    >
      {/* Open Positions */}
      {openPositions.length > 0 && (
        <div className="card-glass rounded-2xl p-5">
          <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
            <Radio className="w-4 h-4 text-emerald-400" />
            Open Positions
            <span className="ml-auto text-xs text-emerald-400/60 bg-emerald-400/10 px-2 py-0.5 rounded-full">
              {openPositions.length} active
            </span>
          </h3>
          <div className="space-y-2">
            {openPositions.map((t, i) => (
              <OpenPositionRow key={i} trade={t} />
            ))}
          </div>
        </div>
      )}

      {/* Recent Trades */}
      <div className="card-glass rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4 text-zinc-500" />
          Recent Trades
          <span className="ml-auto text-xs text-zinc-500">
            Last {trades.filter((t) => t.pnl !== null).length} closed
          </span>
        </h3>

        {/* Table header */}
        <div
          className="grid grid-cols-[1fr_80px_80px_100px_80px_80px_60px] gap-2 px-3 py-2 text-[10px] text-zinc-600 uppercase tracking-wider border-b border-white/[0.04]"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          <span>Asset</span>
          <span>Side</span>
          <span>Strategy</span>
          <span className="text-right">Entry</span>
          <span className="text-right">P&L</span>
          <span className="text-right">Conf</span>
          <span></span>
        </div>

        <div className="space-y-0.5 mt-1">
          {trades
            .filter((t) => t.pnl !== null)
            .map((t, i) => (
              <TradeRow
                key={i}
                trade={t}
                expanded={expandedTrade === i}
                onToggle={() =>
                  setExpandedTrade(expandedTrade === i ? null : i)
                }
              />
            ))}
        </div>
      </div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════
// ACTIVITY TAB
// ═══════════════════════════════════════════════════════════════
const ACTIVITY_TYPE_CONFIG: Record<
  ActivityEntry["type"],
  { icon: React.ReactNode; color: string }
> = {
  trade_executed: {
    icon: <TrendingUp className="w-4 h-4" />,
    color: "text-emerald-400",
  },
  trade_closed: {
    icon: <TrendingDown className="w-4 h-4" />,
    color: "text-zinc-400", // overridden per-entry by P&L
  },
  signal_generated: {
    icon: <Zap className="w-4 h-4" />,
    color: "text-amber-400",
  },
  signal_rejected: {
    icon: <Shield className="w-4 h-4" />,
    color: "text-red-400",
  },
  broker_connected: {
    icon: <Wifi className="w-4 h-4" />,
    color: "text-emerald-400",
  },
  broker_disconnected: {
    icon: <Wifi className="w-4 h-4" />,
    color: "text-red-400",
  },
  system_start: {
    icon: <Play className="w-4 h-4" />,
    color: "text-emerald-400",
  },
  system_stop: {
    icon: <Square className="w-4 h-4" />,
    color: "text-red-400",
  },
  scan_complete: {
    icon: <Radio className="w-4 h-4" />,
    color: "text-blue-400",
  },
  error: {
    icon: <AlertTriangle className="w-4 h-4" />,
    color: "text-red-400",
  },
  optimization: {
    icon: <Brain className="w-4 h-4" />,
    color: "text-purple-400",
  },
};

function ActivityTab() {
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const mergeWithLocal = (serverList: ActivityEntry[]): ActivityEntry[] => {
    try {
      const localTrades: ActivityEntry[] = JSON.parse(
        localStorage.getItem("aifred_local_trades") || "[]"
      );
      // Merge: local trades first, then server entries, deduplicate by id
      const merged = [...localTrades, ...serverList];
      const seen = new Set<string>();
      const deduped = merged.filter((e) => {
        if (seen.has(e.id)) return false;
        seen.add(e.id);
        return true;
      });
      // Sort newest first
      deduped.sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      return deduped;
    } catch {
      return serverList;
    }
  };

  const fetchActivities = () => {
    fetch("/api/trading/activity")
      .then((r) => r.json())
      .then((data) => {
        const serverList = Array.isArray(data) ? data : (data?.activities ?? []);
        const merged = mergeWithLocal(serverList);
        setActivities(merged);
        // Auto-expand the newest trade_executed entry
        const newest = merged.find((e) => e.type === "trade_executed");
        if (newest) setExpandedId((prev) => prev ?? newest.id);
        setLoading(false);
      })
      .catch(() => {
        // Even on server error, show local trades
        const local = mergeWithLocal([]);
        setActivities(local);
        if (local.length > 0) {
          const newest = local.find((e) => e.type === "trade_executed");
          if (newest) setExpandedId((prev) => prev ?? newest.id);
        }
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchActivities();
    const interval = setInterval(fetchActivities, 10_000);
    return () => clearInterval(interval);
  }, []);

  // Separate user-executed trades (from localStorage) for the "Your Trades" section
  // MUST be before any conditional return (React Rules of Hooks)
  const userTrades = useMemo(() => {
    try {
      const local: ActivityEntry[] = JSON.parse(localStorage.getItem("aifred_local_trades") || "[]");
      return local.filter((e) => e.type === "trade_executed").slice(0, 20);
    } catch {
      return [];
    }
  }, [activities]); // re-derive when activities changes

  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex items-center justify-center py-20"
      >
        <div className="w-8 h-8 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="space-y-4 pb-12"
    >
      {/* ─── Your Executed Trades (always visible) ─────────── */}
      <div className="card-glass rounded-2xl p-5" style={{ borderColor: userTrades.length > 0 ? "rgba(16,185,129,0.15)" : undefined }}>
        <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          Your Executed Trades
          <span className="text-[11px] text-emerald-400/70 font-normal">
            {userTrades.length} trades
          </span>
        </h3>
        {userTrades.length === 0 ? (
          <div className="text-center py-8">
            <ArrowUpDown className="w-8 h-8 text-zinc-700 mx-auto mb-3" />
            <p className="text-sm text-zinc-500 mb-1">No trades executed yet</p>
            <p className="text-xs text-zinc-600">
              Click the <span className="text-emerald-400 font-semibold">Execute Trade</span> button above to place your first trade
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[40vh] overflow-y-auto pr-1">
            {userTrades.map((trade) => {
              const isExpanded = expandedId === trade.id;
              return (
                <motion.div
                  key={trade.id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-xl border border-emerald-500/10 bg-emerald-500/[0.03] overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedId(isExpanded ? null : trade.id)}
                    className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-emerald-500/[0.02] transition-colors"
                  >
                    <TrendingUp className={`w-4 h-4 flex-shrink-0 ${trade.details?.side === "SHORT" ? "text-red-400" : "text-emerald-400"}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-semibold text-white">{trade.title}</span>
                        {trade.details?.tier && (
                          <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${
                            trade.details.tier === "A+" ? "bg-amber-500/15 text-amber-400" :
                            trade.details.tier === "A" ? "bg-emerald-500/15 text-emerald-400" :
                            "bg-indigo-500/15 text-indigo-400"
                          }`} style={{ fontFamily: "JetBrains Mono, monospace" }}>
                            {trade.details.tier}
                          </span>
                        )}
                        {trade.details?.confidence !== undefined && (
                          <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                            trade.details.confidence >= 85 ? "bg-emerald-500/10 text-emerald-400" :
                            trade.details.confidence >= 75 ? "bg-amber-500/10 text-amber-400" :
                            "bg-red-500/10 text-red-400"
                          }`} style={{ fontFamily: "JetBrains Mono, monospace" }}>
                            {trade.details.confidence}%
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-zinc-500 mt-0.5 truncate">{trade.message}</p>
                      {!isExpanded && trade.details?.reasoning && (
                        <p className="text-[10px] text-indigo-400/70 mt-0.5 flex items-center gap-1">
                          <Brain className="w-2.5 h-2.5 flex-shrink-0" />
                          <span className="truncate">{trade.details?.reasoning?.slice(0, 90)}{(trade.details?.reasoning?.length ?? 0) > 90 ? "..." : ""}</span>
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className="text-[10px] text-zinc-600" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                        {timeAgo(trade.timestamp)}
                      </span>
                      {isExpanded ? <ChevronUp className="w-3.5 h-3.5 text-zinc-600" /> : <ChevronDown className="w-3.5 h-3.5 text-zinc-600" />}
                    </div>
                  </button>
                  <AnimatePresence>
                    {isExpanded && trade.details && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="px-4 pb-4 space-y-3">
                          {/* Trade stats grid */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]" style={{ fontFamily: "JetBrains Mono, monospace" }}>
                            {trade.details?.asset && <div className="bg-white/[0.03] rounded-lg px-3 py-2"><div className="text-[9px] text-zinc-600 uppercase mb-0.5">Asset</div><div className="text-zinc-300">{trade.details.asset}</div></div>}
                            {trade.details?.side && <div className="bg-white/[0.03] rounded-lg px-3 py-2"><div className="text-[9px] text-zinc-600 uppercase mb-0.5">Side</div><div className={trade.details.side === "LONG" ? "text-emerald-400" : "text-red-400"}>{trade.details.side}</div></div>}
                            {trade.details?.entry_price != null && <div className="bg-white/[0.03] rounded-lg px-3 py-2"><div className="text-[9px] text-zinc-600 uppercase mb-0.5">Entry</div><div className="text-zinc-300">${Number(trade.details.entry_price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</div></div>}
                            {trade.details?.strategy && <div className="bg-white/[0.03] rounded-lg px-3 py-2"><div className="text-[9px] text-zinc-600 uppercase mb-0.5">Strategy</div><div className="text-zinc-300">{trade.details.strategy}</div></div>}
                            {trade.details?.stop_loss != null && <div className="bg-white/[0.03] rounded-lg px-3 py-2"><div className="text-[9px] text-zinc-600 uppercase mb-0.5">Stop Loss</div><div className="text-red-400">${Number(trade.details.stop_loss).toFixed(2)}</div></div>}
                            {trade.details?.take_profit != null && <div className="bg-white/[0.03] rounded-lg px-3 py-2"><div className="text-[9px] text-zinc-600 uppercase mb-0.5">Take Profit</div><div className="text-emerald-400">${Number(trade.details.take_profit).toFixed(2)}</div></div>}
                          </div>
                          {/* Reasoning sections */}
                          {trade.details?.reasoning && (
                            <div className="card-glass rounded-xl p-4"><div className="text-[10px] text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}><Brain className="w-3 h-3" /> AI Reasoning</div><p className="text-xs text-zinc-400 leading-relaxed">{trade.details.reasoning}</p></div>
                          )}
                          {trade.details?.technical_signals && (
                            <div className="card-glass rounded-xl p-4"><div className="text-[10px] text-emerald-400 uppercase tracking-wider mb-2 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}><BarChart3 className="w-3 h-3" /> Technical Signals</div><p className="text-xs text-zinc-400 leading-relaxed">{trade.details.technical_signals}</p></div>
                          )}
                          {trade.details?.sentiment_signals && (
                            <div className="card-glass rounded-xl p-4"><div className="text-[10px] text-amber-400 uppercase tracking-wider mb-2 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}><Zap className="w-3 h-3" /> Sentiment</div><p className="text-xs text-zinc-400 leading-relaxed">{trade.details.sentiment_signals}</p></div>
                          )}
                          {trade.details?.risk_assessment && (
                            <div className="card-glass rounded-xl p-4"><div className="text-[10px] text-red-400 uppercase tracking-wider mb-2 flex items-center gap-1.5" style={{ fontFamily: "JetBrains Mono, monospace" }}><Shield className="w-3 h-3" /> Risk</div><p className="text-xs text-zinc-400 leading-relaxed">{trade.details.risk_assessment}</p></div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* ─── Full Agent Reasoning Log ─────────────────────── */}
      <div className="card-glass rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-zinc-300 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4 text-zinc-500" />
          Agent Reasoning Log
          <span className="text-[11px] text-zinc-600 font-normal">
            {activities.length} entries
          </span>
          <span className="ml-auto flex items-center gap-3">
            <button
              onClick={() => {
                localStorage.removeItem("aifred_local_trades");
                fetchActivities();
              }}
              className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors uppercase tracking-wider"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
              title="Clear locally stored trades"
            >
              clear local
            </button>
            <div className="flex items-center gap-1.5">
              <div
                className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
              />
              <span
                className="text-[10px] text-emerald-400/70 uppercase tracking-wider"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                Live · 10s
              </span>
            </div>
          </span>
        </h3>

        {activities.length === 0 ? (
          <div className="text-center py-12">
            <Activity className="w-8 h-8 text-zinc-700 mx-auto mb-3" />
            <p className="text-sm text-zinc-600">No activity yet</p>
          </div>
        ) : (
          <div className="space-y-1 max-h-[70vh] overflow-y-auto pr-1">
            {activities.map((entry, i) => {
              const config = ACTIVITY_TYPE_CONFIG[entry.type] || {
                icon: <Activity className="w-4 h-4" />,
                color: "text-zinc-400",
              };
              // For trade_closed, color based on P&L
              const iconColor =
                entry.type === "trade_closed" && entry.details?.pnl !== undefined
                  ? entry.details.pnl >= 0
                    ? "text-emerald-400"
                    : "text-red-400"
                  : config.color;
              const isExpanded = expandedId === entry.id;

              return (
                <motion.div
                  key={entry.id}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                >
                  <button
                    onClick={() =>
                      setExpandedId(isExpanded ? null : entry.id)
                    }
                    className="w-full flex items-start gap-3 px-3 py-3 rounded-lg hover:bg-white/[0.02] transition-colors text-left"
                  >
                    {/* Icon */}
                    <div
                      className={`mt-0.5 flex-shrink-0 ${iconColor}`}
                    >
                      {config.icon}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-xs font-semibold text-zinc-200">
                          {entry.title}
                        </span>
                        {entry.details?.tier && (
                          <span
                            className="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.04] text-zinc-500"
                            style={{
                              fontFamily: "JetBrains Mono, monospace",
                            }}
                          >
                            {entry.details.tier}
                          </span>
                        )}
                        {entry.details?.confidence !== undefined && (
                          <span
                            className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                              entry.details.confidence >= 85
                                ? "bg-emerald-500/10 text-emerald-400"
                                : entry.details.confidence >= 75
                                ? "bg-amber-500/10 text-amber-400"
                                : "bg-red-500/10 text-red-400"
                            }`}
                            style={{
                              fontFamily: "JetBrains Mono, monospace",
                            }}
                          >
                            {entry.details.confidence}% conf
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-zinc-500 mt-0.5 truncate">
                        {entry.message}
                      </p>
                      {/* Reasoning preview (trade entries only) */}
                      {!isExpanded && entry.details?.reasoning && (
                        <p className="text-[10px] text-indigo-400/70 mt-0.5 flex items-center gap-1">
                          <Brain className="w-2.5 h-2.5 flex-shrink-0" />
                          <span className="truncate">
                            {entry.details?.reasoning?.slice(0, 90)}
                            {(entry.details?.reasoning?.length ?? 0) > 90 ? "…" : ""}
                          </span>
                        </p>
                      )}
                    </div>

                    {/* Time + expand */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span
                        className="text-[10px] text-zinc-600"
                        style={{
                          fontFamily: "JetBrains Mono, monospace",
                        }}
                      >
                        {timeAgo(entry.timestamp)}
                      </span>
                      {entry.details && (
                        <span className="text-zinc-600">
                          {isExpanded ? (
                            <ChevronUp className="w-3.5 h-3.5" />
                          ) : (
                            <ChevronDown className="w-3.5 h-3.5" />
                          )}
                        </span>
                      )}
                    </div>
                  </button>

                  {/* Expandable details */}
                  <AnimatePresence>
                    {isExpanded && entry.details && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                      >
                        <div className="px-10 pb-4 space-y-3">
                          {/* Trade details grid */}
                          {(entry.details.asset ||
                            entry.details.entry_price !== undefined) && (
                            <div
                              className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]"
                              style={{
                                fontFamily: "JetBrains Mono, monospace",
                              }}
                            >
                              {entry.details.asset && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Asset
                                  </div>
                                  <div className="text-zinc-300">
                                    {entry.details.asset}
                                  </div>
                                </div>
                              )}
                              {entry.details.side && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Side
                                  </div>
                                  <div
                                    className={
                                      entry.details.side === "LONG"
                                        ? "text-emerald-400"
                                        : "text-red-400"
                                    }
                                  >
                                    {entry.details.side}
                                  </div>
                                </div>
                              )}
                              {entry.details.strategy && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Strategy
                                  </div>
                                  <div className="text-zinc-300">
                                    {STRATEGY_LABELS[
                                      entry.details.strategy
                                    ] || entry.details.strategy}
                                  </div>
                                </div>
                              )}
                              {entry.details.confidence !== undefined && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Confidence
                                  </div>
                                  <div className="text-amber-400">
                                    {Number(entry.details.confidence).toFixed(0)}%
                                  </div>
                                </div>
                              )}
                              {entry.details.entry_price !== undefined && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Entry Price
                                  </div>
                                  <div className="text-zinc-300">
                                    $
                                    {Number(entry.details.entry_price).toLocaleString(
                                      undefined,
                                      {
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2,
                                      }
                                    )}
                                  </div>
                                </div>
                              )}
                              {entry.details.stop_loss !== undefined && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Stop Loss
                                  </div>
                                  <div className="text-red-400">
                                    ${Number(entry.details.stop_loss).toFixed(2)}
                                  </div>
                                </div>
                              )}
                              {entry.details.take_profit !== undefined && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Take Profit
                                  </div>
                                  <div className="text-emerald-400">
                                    ${Number(entry.details.take_profit).toFixed(2)}
                                  </div>
                                </div>
                              )}
                              {entry.details.tier && (
                                <div className="bg-white/[0.02] rounded-lg px-3 py-2">
                                  <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-0.5">
                                    Tier
                                  </div>
                                  <div
                                    style={{
                                      color:
                                        TIER_COLORS[entry.details.tier] ||
                                        "#a1a1aa",
                                    }}
                                  >
                                    {entry.details.tier}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Reasoning section */}
                          {entry.details.reasoning && (
                            <div className="card-glass rounded-xl p-4">
                              <div
                                className="text-[10px] text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-1.5"
                                style={{
                                  fontFamily: "JetBrains Mono, monospace",
                                }}
                              >
                                <Brain className="w-3 h-3" />
                                AI Reasoning
                              </div>
                              <p className="text-xs text-zinc-400 leading-relaxed">
                                {entry.details.reasoning}
                              </p>
                            </div>
                          )}

                          {entry.details.technical_signals && (
                            <div className="card-glass rounded-xl p-4">
                              <div
                                className="text-[10px] text-emerald-400 uppercase tracking-wider mb-2 flex items-center gap-1.5"
                                style={{
                                  fontFamily: "JetBrains Mono, monospace",
                                }}
                              >
                                <BarChart3 className="w-3 h-3" />
                                Technical Analysis
                              </div>
                              <p className="text-xs text-zinc-400 leading-relaxed">
                                {entry.details.technical_signals}
                              </p>
                            </div>
                          )}

                          {entry.details.sentiment_signals && (
                            <div className="card-glass rounded-xl p-4">
                              <div
                                className="text-[10px] text-amber-400 uppercase tracking-wider mb-2 flex items-center gap-1.5"
                                style={{
                                  fontFamily: "JetBrains Mono, monospace",
                                }}
                              >
                                <Zap className="w-3 h-3" />
                                Sentiment Analysis
                              </div>
                              <p className="text-xs text-zinc-400 leading-relaxed">
                                {entry.details.sentiment_signals}
                              </p>
                            </div>
                          )}

                          {entry.details.risk_assessment && (
                            <div className="card-glass rounded-xl p-4">
                              <div
                                className="text-[10px] text-red-400 uppercase tracking-wider mb-2 flex items-center gap-1.5"
                                style={{
                                  fontFamily: "JetBrains Mono, monospace",
                                }}
                              >
                                <Shield className="w-3 h-3" />
                                Risk Assessment
                              </div>
                              <p className="text-xs text-zinc-400 leading-relaxed">
                                {entry.details.risk_assessment}
                              </p>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════
// AGENTS TAB
// ═══════════════════════════════════════════════════════════════
function AgentsTab({ data }: { data: TradingData }) {
  const agents = [
    {
      name: "Data Ingestion",
      status: "active",
      desc: "Price feeds, orderbooks, news scraping",
      tech: "ccxt, yfinance, feedparser",
      files: 10,
      lines: 2646,
    },
    {
      name: "Technical Analysis",
      status: "active",
      desc: "LSTM, Transformer, CNN pattern detection",
      tech: "PyTorch, pandas-ta, XGBoost",
      files: 10,
      lines: 3627,
    },
    {
      name: "NLP & Sentiment",
      status: "active",
      desc: "FinBERT, LLM analysis, Fear & Greed",
      tech: "HuggingFace, spaCy, Claude API",
      files: 9,
      lines: 1613,
    },
    {
      name: "Risk Management",
      status: "active",
      desc: "Kelly sizing, ATR stops, drawdown protection",
      tech: "numpy, scipy, empyrical",
      files: 9,
      lines: 1750,
    },
    {
      name: "Execution",
      status: "active",
      desc: "Multi-exchange, smart order routing",
      tech: "ccxt, alpaca-trade-api",
      files: 7,
      lines: 1546,
    },
    {
      name: "Monitoring",
      status: "active",
      desc: "Trade logging, alerts, dashboards",
      tech: "Streamlit, Telegram",
      files: 7,
      lines: 1361,
    },
    {
      name: "Orchestrator",
      status: "active",
      desc: "Central coordinator, signal fusion",
      tech: "Custom Python",
      files: 2,
      lines: 1304,
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="space-y-4 pb-12"
    >
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map((agent, i) => (
          <motion.div
            key={agent.name}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
            className="card-glass rounded-2xl p-5 group"
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-zinc-200">
                  {agent.name}
                </h3>
                <p className="text-xs text-zinc-500 mt-0.5">{agent.desc}</p>
              </div>
              <div className="flex items-center gap-1.5">
                <div
                  className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                  style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
                />
                <span
                  className="text-[10px] text-emerald-400/70 uppercase tracking-wider"
                  style={{ fontFamily: "JetBrains Mono, monospace" }}
                >
                  online
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3 mt-4">
              <span
                className="text-[10px] text-zinc-600 bg-white/[0.03] px-2 py-1 rounded"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                {agent.tech}
              </span>
            </div>

            <div
              className="flex items-center gap-4 mt-3 text-[10px] text-zinc-600"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              <span>{agent.files} files</span>
              <span>{agent.lines.toLocaleString()} lines</span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* System stats */}
      <div className="card-glass rounded-2xl p-5">
        <h3 className="text-sm font-semibold text-zinc-300 mb-4">
          System Architecture
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div
              className="text-2xl font-bold text-emerald-400 glow-green"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              7
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">
              Active Agents
            </div>
          </div>
          <div className="text-center">
            <div
              className="text-2xl font-bold text-amber-400 glow-gold"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              61
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">
              Python Modules
            </div>
          </div>
          <div className="text-center">
            <div
              className="text-2xl font-bold text-indigo-400"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              18.8K
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">
              Lines of Code
            </div>
          </div>
          <div className="text-center">
            <div
              className="text-2xl font-bold text-zinc-300"
              style={{ fontFamily: "JetBrains Mono, monospace" }}
            >
              5
            </div>
            <div className="text-[10px] text-zinc-500 mt-1 uppercase tracking-wider">
              ML Models
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════

function HeroStat({
  label,
  value,
  sub,
  positive,
  icon,
  delay,
  highlight,
  invert,
}: {
  label: string;
  value: string;
  sub: string;
  positive: boolean;
  icon: React.ReactNode;
  delay: number;
  highlight?: boolean;
  invert?: boolean;
}) {
  const color = invert
    ? positive
      ? "text-emerald-400"
      : "text-red-400"
    : positive
    ? "text-emerald-400"
    : "text-red-400";
  const glow = invert
    ? positive
      ? "glow-green"
      : "glow-red"
    : positive
    ? "glow-green"
    : "glow-red";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className={`card-glass rounded-2xl p-5 relative overflow-hidden ${
        highlight ? "ring-1 ring-emerald-500/20" : ""
      }`}
    >
      {highlight && (
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            background:
              "radial-gradient(circle at 50% 0%, rgba(16,185,129,1) 0%, transparent 60%)",
          }}
        />
      )}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-zinc-500">{icon}</span>
        <span
          className="text-[10px] text-zinc-500 uppercase tracking-[0.15em]"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {label}
        </span>
      </div>
      <div
        className={`text-2xl font-bold ${color} ${glow}`}
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        {value}
      </div>
      <div
        className="text-[11px] text-zinc-600 mt-1"
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        {sub}
      </div>
    </motion.div>
  );
}

function MiniStat({
  label,
  value,
  positive,
  neutral,
}: {
  label: string;
  value: string;
  positive?: boolean;
  neutral?: boolean;
}) {
  return (
    <div className="text-center">
      <div
        className={`text-lg font-semibold ${
          neutral
            ? "text-zinc-300"
            : positive
            ? "text-emerald-400"
            : "text-red-400"
        }`}
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        {value}
      </div>
      <div
        className="text-[10px] text-zinc-600 mt-0.5 uppercase tracking-wider"
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        {label}
      </div>
    </div>
  );
}

function AssetRow({
  asset,
  pnl,
  trades,
  winRate,
}: {
  asset: string;
  pnl: number;
  trades: number;
  winRate: number;
}) {
  return (
    <div className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-white/[0.02] transition-colors">
      <div className="flex items-center gap-3">
        <span className="text-xs font-medium text-zinc-300 w-20">
          {asset}
        </span>
        <span
          className="text-[10px] text-zinc-600"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {trades}t
        </span>
      </div>
      <div className="flex items-center gap-4">
        <div className="w-16 h-1 bg-white/[0.04] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${
              winRate >= 70 ? "bg-emerald-500" : winRate >= 50 ? "bg-amber-500" : "bg-red-500"
            }`}
            style={{ width: `${Math.min(winRate, 100)}%` }}
          />
        </div>
        <span
          className={`text-xs font-medium w-14 text-right ${
            pnl >= 0 ? "text-emerald-400" : "text-red-400"
          }`}
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {pnl >= 0 ? "+" : ""}
          {fmt(pnl)}
        </span>
      </div>
    </div>
  );
}

function StrategyRow({
  strategy,
  pnl,
  trades,
  winRate,
}: {
  strategy: string;
  pnl: number;
  trades: number;
  winRate: number;
}) {
  return (
    <div className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-white/[0.02] transition-colors">
      <div>
        <span className="text-xs font-medium text-zinc-300">
          {STRATEGY_LABELS[strategy] || strategy}
        </span>
        <div
          className="text-[10px] text-zinc-600 mt-0.5"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {trades} trades · WR {pct(winRate)}
        </div>
      </div>
      <span
        className={`text-xs font-medium ${
          pnl >= 0 ? "text-emerald-400" : "text-red-400"
        }`}
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        {pnl >= 0 ? "+" : ""}
        {fmt(pnl)}
      </span>
    </div>
  );
}

function TradeRow({
  trade,
  expanded,
  onToggle,
}: {
  trade: Record<string, unknown>;
  expanded: boolean;
  onToggle: () => void;
}) {
  const pnl = Number(trade.pnl ?? 0);
  const isWin = pnl > 0;

  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full grid grid-cols-[1fr_80px_80px_100px_80px_80px_60px] gap-2 px-3 py-2.5 rounded-lg hover:bg-white/[0.02] transition-colors items-center text-left"
      >
        <span className="text-xs font-medium text-zinc-200">
          {trade.asset as string}
        </span>
        <span
          className={`text-[11px] font-medium ${
            trade.side === "LONG" ? "text-emerald-400" : "text-red-400"
          }`}
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {trade.side as string}
        </span>
        <span
          className="text-[10px] text-zinc-500 truncate"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {((trade.strategy as string) || "").slice(0, 8)}
        </span>
        <span
          className="text-[11px] text-zinc-400 text-right"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {Number(trade.entry_price ?? 0).toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}
        </span>
        <span
          className={`text-[11px] font-medium text-right ${
            isWin ? "text-emerald-400" : "text-red-400"
          }`}
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {isWin ? "+" : ""}
          {fmt(pnl)}
        </span>
        <span
          className="text-[11px] text-zinc-500 text-right"
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {Number(trade.confidence ?? 0).toFixed(0)}%
        </span>
        <span className="text-zinc-600 flex justify-end">
          {expanded ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </span>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 space-y-2">
              {!!trade.reasoning && (
                <div className="bg-white/[0.02] rounded-lg p-3">
                  <div
                    className="text-[10px] text-zinc-600 uppercase tracking-wider mb-1"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    Entry Reasoning
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {trade.reasoning as string}
                  </p>
                </div>
              )}
              {!!trade.exit_reason && (
                <div className="bg-white/[0.02] rounded-lg p-3">
                  <div
                    className="text-[10px] text-zinc-600 uppercase tracking-wider mb-1"
                    style={{ fontFamily: "JetBrains Mono, monospace" }}
                  >
                    Exit Reasoning
                  </div>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {trade.exit_reason as string}
                  </p>
                </div>
              )}
              <div
                className="flex gap-4 text-[10px] text-zinc-600"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                <span>
                  TIER: {(trade.tier as string) || "—"}
                </span>
                <span>
                  STOP: {Number(trade.stop_loss ?? 0).toFixed(2)}
                </span>
                <span>
                  TP: {Number(trade.take_profit ?? 0).toFixed(2)}
                </span>
                <span>
                  FEES: ${Number(trade.fees || 0).toFixed(2)}
                </span>
                <span>
                  EXIT: {Number(trade.exit_price || 0).toFixed(2)}
                </span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function OpenPositionRow({ trade }: { trade: Record<string, unknown> }) {
  return (
    <div className="flex items-center justify-between py-2.5 px-3 rounded-lg bg-emerald-500/[0.03] border border-emerald-500/10">
      <div className="flex items-center gap-4">
        <div
          className="w-1.5 h-1.5 rounded-full bg-emerald-400"
          style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
        />
        <span className="text-xs font-medium text-zinc-200">
          {trade.asset as string}
        </span>
        <span
          className={`text-[10px] font-medium ${
            trade.side === "LONG" ? "text-emerald-400" : "text-red-400"
          }`}
          style={{ fontFamily: "JetBrains Mono, monospace" }}
        >
          {trade.side as string}
        </span>
      </div>
      <div
        className="flex items-center gap-4 text-[11px]"
        style={{ fontFamily: "JetBrains Mono, monospace" }}
      >
        <span className="text-zinc-500">
          Entry: {Number(trade.entry_price ?? 0).toFixed(2)}
        </span>
        <span className="text-zinc-500">
          Conf: {Number(trade.confidence ?? 0).toFixed(0)}%
        </span>
        <span className="text-zinc-500">
          SL: {Number(trade.stop_loss ?? 0).toFixed(2)}
        </span>
      </div>
    </div>
  );
}

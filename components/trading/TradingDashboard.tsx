"use client";

import { useEffect, useState, useMemo } from "react";
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

// ─── Utility ─────────────────────────────────────────────────
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

// ─── Component ───────────────────────────────────────────────
export default function TradingDashboard() {
  const router = useRouter();
  const [data, setData] = useState<TradingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTrade, setExpandedTrade] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "trades" | "agents">(
    "overview"
  );

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
      <header className="relative z-10 border-b border-white/[0.06] px-6 py-4">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
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
              <p
                className="text-[11px] text-zinc-500 tracking-[0.2em] uppercase"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                Multi-Agent Trading Intelligence
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            {/* System Status */}
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full bg-emerald-400"
                style={{ animation: "pulse-glow 2s ease-in-out infinite" }}
              />
              <span
                className="text-xs text-emerald-400/80 tracking-wider"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              >
                7 AGENTS ONLINE
              </span>
            </div>

            {/* Nav tabs */}
            <div className="flex gap-1 bg-white/[0.03] rounded-lg p-1">
              {(["overview", "trades", "agents"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-1.5 text-xs font-medium rounded-md transition-all tracking-wider uppercase ${
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
      </header>

      {/* ─── Content ────────────────────────────────────────── */}
      <main className="relative z-10 max-w-[1600px] mx-auto px-6 py-6">
        <AnimatePresence mode="wait">
          {activeTab === "overview" && (
            <OverviewTab key="overview" data={data} />
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
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// OVERVIEW TAB
// ═══════════════════════════════════════════════════════════════
function OverviewTab({ data }: { data: TradingData }) {
  const { summary } = data;
  const isPositive = summary.totalPnl >= 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 pb-12"
    >
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
  const pnl = trade.pnl as number;
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
          {Number(trade.entry_price).toLocaleString(undefined, {
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
          {Number(trade.confidence).toFixed(0)}%
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
                  STOP: {Number(trade.stop_loss).toFixed(2)}
                </span>
                <span>
                  TP: {Number(trade.take_profit).toFixed(2)}
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
          Entry: {Number(trade.entry_price).toFixed(2)}
        </span>
        <span className="text-zinc-500">
          Conf: {Number(trade.confidence).toFixed(0)}%
        </span>
        <span className="text-zinc-500">
          SL: {Number(trade.stop_loss).toFixed(2)}
        </span>
      </div>
    </div>
  );
}

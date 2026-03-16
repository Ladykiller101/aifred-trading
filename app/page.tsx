"use client";

import dynamic from "next/dynamic";

const TradingDashboard = dynamic(
  () => import("@/components/trading/TradingDashboard"),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-[#06060a] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
          <span className="text-zinc-500 text-sm font-mono tracking-wider">
            LOADING SYSTEMS...
          </span>
        </div>
      </div>
    ),
  }
);

export default function HomePage() {
  return <TradingDashboard />;
}

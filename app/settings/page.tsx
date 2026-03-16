"use client";

import dynamic from "next/dynamic";

const TradingSettings = dynamic(
  () => import("@/components/trading/TradingSettings"),
  { ssr: false }
);

export default function SettingsPage() {
  return <TradingSettings />;
}

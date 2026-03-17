import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

// Use /tmp for writes (writable on Vercel), fall back to data/ for reads
const TMP_DIR = "/tmp/aifred-data";
const DATA_DIR = join(process.cwd(), "data");

function ensureTmpDir() {
  if (!existsSync(TMP_DIR)) mkdirSync(TMP_DIR, { recursive: true });
}

// ---------------------------------------------------------------------------
// Activity log helper
// ---------------------------------------------------------------------------

function appendActivity(entry: {
  type: string;
  severity: string;
  title: string;
  message: string;
  details?: Record<string, unknown>;
}) {
  try {
    ensureTmpDir();
    const paths = [join(TMP_DIR, "activity-log.json"), join(DATA_DIR, "activity-log.json")];
    let activities: unknown[] = [];
    for (const p of paths) {
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
    writeFileSync(join(TMP_DIR, "activity-log.json"), JSON.stringify(activities, null, 2), "utf-8");
  } catch (e) {
    console.error("Failed to log activity from controls route:", e);
  }
}

// ---------------------------------------------------------------------------
// Trading controls state
// ---------------------------------------------------------------------------

interface TradingControlsState {
  mode: "paper" | "live";
  running: boolean;
  scanInterval: number; // seconds
  assets: string[];
  lastScan: string | null;
}

const CONTROLS_PATH_TMP = join(TMP_DIR, "trading-controls.json");
const CONTROLS_PATH_DATA = join(DATA_DIR, "trading-controls.json");

const DEFAULT_STATE: TradingControlsState = {
  mode: "paper",
  running: false,
  scanInterval: 300, // 5 minutes
  assets: ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  lastScan: null,
};

function readState(): TradingControlsState {
  // Try /tmp first (latest writes), then data/ (seeded defaults)
  for (const p of [CONTROLS_PATH_TMP, CONTROLS_PATH_DATA]) {
    if (existsSync(p)) {
      try {
        return { ...DEFAULT_STATE, ...JSON.parse(readFileSync(p, "utf-8")) };
      } catch { /* ignore */ }
    }
  }
  return { ...DEFAULT_STATE };
}

function writeState(state: TradingControlsState) {
  ensureTmpDir();
  writeFileSync(CONTROLS_PATH_TMP, JSON.stringify(state, null, 2), "utf-8");
}

// ---------------------------------------------------------------------------
// GET /api/trading/controls — current trading system status
// ---------------------------------------------------------------------------

export async function GET() {
  try {
    const state = readState();
    return NextResponse.json(state);
  } catch (error) {
    console.error("Trading controls GET error:", error);
    return NextResponse.json(
      { error: "Failed to read trading controls" },
      { status: 500 },
    );
  }
}

// ---------------------------------------------------------------------------
// POST /api/trading/controls — update trading system settings
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, mode, scanInterval, assets } = body as {
      action: "start" | "stop" | "toggle_mode";
      mode?: "paper" | "live";
      scanInterval?: number;
      assets?: string[];
    };

    if (!action) {
      return NextResponse.json(
        { success: false, message: "action is required (start | stop | toggle_mode)" },
        { status: 400 },
      );
    }

    const state = readState();

    switch (action) {
      case "start":
        state.running = true;
        state.lastScan = new Date().toISOString();
        break;

      case "stop":
        state.running = false;
        break;

      case "toggle_mode":
        if (mode && (mode === "paper" || mode === "live")) {
          state.mode = mode;
        } else {
          state.mode = state.mode === "paper" ? "live" : "paper";
        }
        // Automatically stop when switching to live for safety
        if (state.mode === "live") {
          state.running = false;
        }
        break;

      default:
        return NextResponse.json(
          { success: false, message: `Unknown action: ${action}` },
          { status: 400 },
        );
    }

    // Apply optional overrides
    if (scanInterval !== undefined && scanInterval > 0) {
      state.scanInterval = scanInterval;
    }
    if (assets !== undefined && Array.isArray(assets)) {
      state.assets = assets;
    }

    writeState(state);

    // Log activity for state changes
    if (action === "start") {
      appendActivity({
        type: "system_start",
        severity: "success",
        title: "Trading System Started",
        message: `Trading engine started in ${state.mode} mode. Monitoring ${state.assets.length} assets.`,
        details: { tier: state.mode },
      });
    } else if (action === "stop") {
      appendActivity({
        type: "system_stop",
        severity: "warning",
        title: "Trading System Stopped",
        message: `Trading engine stopped. Mode was: ${state.mode}.`,
        details: { tier: state.mode },
      });
    } else if (action === "toggle_mode") {
      appendActivity({
        type: state.mode === "live" ? "system_stop" : "optimization",
        severity: state.mode === "live" ? "warning" : "info",
        title: `Mode Changed to ${state.mode.toUpperCase()}`,
        message: `Trading mode switched to ${state.mode}.${state.mode === "live" ? " CAUTION: Real money mode. Trading auto-stopped for safety." : " Paper trading active."}`,
        details: { tier: state.mode },
      });
    }

    return NextResponse.json({
      success: true,
      message: `Trading system: action="${action}" applied`,
      currentState: state,
    });
  } catch (error) {
    console.error("Trading controls POST error:", error);
    return NextResponse.json(
      { success: false, message: "Failed to update trading controls" },
      { status: 500 },
    );
  }
}

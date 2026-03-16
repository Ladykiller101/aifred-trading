import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

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

const DATA_DIR = join(process.cwd(), "data");
const CONTROLS_PATH = join(DATA_DIR, "trading-controls.json");

const DEFAULT_STATE: TradingControlsState = {
  mode: "paper",
  running: false,
  scanInterval: 300, // 5 minutes
  assets: ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  lastScan: null,
};

function ensureDataDir() {
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }
}

function readState(): TradingControlsState {
  if (!existsSync(CONTROLS_PATH)) return { ...DEFAULT_STATE };
  try {
    return { ...DEFAULT_STATE, ...JSON.parse(readFileSync(CONTROLS_PATH, "utf-8")) };
  } catch {
    return { ...DEFAULT_STATE };
  }
}

function writeState(state: TradingControlsState) {
  ensureDataDir();
  writeFileSync(CONTROLS_PATH, JSON.stringify(state, null, 2), "utf-8");
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

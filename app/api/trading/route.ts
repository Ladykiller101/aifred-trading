import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const jsonPath = join(process.cwd(), "data", "trading-data.json");

    if (!existsSync(jsonPath)) {
      return NextResponse.json(
        { error: "Trading data not found. Run: python scripts/export_trading_data.py" },
        { status: 404 }
      );
    }

    const raw = readFileSync(jsonPath, "utf-8");
    const data = JSON.parse(raw);

    return NextResponse.json(data);
  } catch (error) {
    console.error("Trading API error:", error);
    return NextResponse.json(
      { error: "Failed to load trading data" },
      { status: 500 }
    );
  }
}

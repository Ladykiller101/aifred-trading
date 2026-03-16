"""Seed the trading database with optimized demo data including reasoning logs."""

import sqlite3
import random
import os
from datetime import datetime, timedelta, timezone

DB_PATH = "data/trading.db"

# Reasoning templates per strategy
REASONING_TEMPLATES = {
    "mean_reversion": {
        "LONG": [
            "RSI({rsi}) oversold + BB lower band touch at {price:.2f}. MACD histogram turning positive. {tf} mean reversion setup with {conf:.0f}% ensemble confidence. Kelly size: {size_pct:.1f}% of portfolio.",
            "Price 2.1 std below 20-period mean at {price:.2f}. Stochastic K({stoch}) in oversold zone with bullish divergence. Volume spike confirms selling exhaustion. {n_confluences}/6 confluences active.",
            "BB squeeze breakout pending. RSI({rsi}) bouncing from 28 with bullish divergence on {tf}. Order block support at {support:.2f}. ICT discount zone entry. Tier: {tier}.",
            "Mean reversion trigger: price at -2.3σ from VWAP. CCI({cci}) extreme oversold. ATR-based stop at {stop:.2f} (1.8 ATR). R:R = {rr:.1f}. Sentiment neutral — pure technical play.",
        ],
        "SHORT": [
            "RSI({rsi}) overbought + BB upper band rejection at {price:.2f}. MACD bearish crossover confirmed. {tf} mean reversion short with {n_confluences}/6 confluences.",
            "Price 2.4 std above 20-period mean. Stochastic K({stoch}) overbought with bearish divergence. Volume declining on rally — distribution pattern. Tier: {tier}.",
            "Extreme overbought: RSI({rsi}), CCI({cci}). Supply zone rejection at {price:.2f}. FVG above acting as resistance. BB-middle target at {target:.2f}.",
        ],
    },
    "ict_confluence": {
        "LONG": [
            "ICT bullish order block at {support:.2f} with FVG fill. Liquidity sweep below {sweep:.2f} low confirmed. London kill zone entry (08:15 UTC). Displacement candle +{disp:.1f} ATR.",
            "Smart money long: sweep of Asian session low at {sweep:.2f} → immediate displacement up. Bullish OB at {support:.2f} holding. Institutional buying footprint in order flow. {n_confluences} ICT confluences.",
            "Optimal trade entry in discount zone (below 50% fib). Breaker block support at {support:.2f}. NY kill zone overlap. Liquidity target: {target:.2f} (previous high).",
        ],
        "SHORT": [
            "ICT bearish order block rejection at {price:.2f}. Buyside liquidity swept at {sweep:.2f}. Bearish FVG at {price:.2f} acting as supply. NY session reversal setup.",
            "Smart money short: sweep of London high at {sweep:.2f} → displacement down. Bearish OB at {price:.2f}. Sellside liquidity target: {target:.2f}. {n_confluences} ICT confluences.",
        ],
    },
    "lstm_ensemble": {
        "LONG": [
            "LSTM predicts +{pred_pct:.1f}% move (confidence: {lstm_conf:.0f}%). Transformer confirms bullish on 4h+1D. XGBoost meta-learner score: {meta:.2f}. 3/3 models agree → STRONG_BUY. Walk-forward validated.",
            "ML ensemble LONG signal: LSTM direction=UP (p={lstm_conf:.0f}%), Transformer momentum=bullish, CNN pattern=ascending triangle (82% completion). Ensemble confidence: {conf:.0f}%.",
            "Neural net consensus: LSTM attention weights concentrated on recent momentum shift. Feature importance: RSI_divergence (0.23), volume_breakout (0.19), ema_alignment (0.17). Predicted move: +{pred_pct:.1f}%.",
        ],
        "SHORT": [
            "LSTM predicts -{pred_pct:.1f}% move (confidence: {lstm_conf:.0f}%). Transformer bearish divergence on multi-timeframe. XGBoost meta-learner: {meta:.2f}. Ensemble: STRONG_SELL.",
            "ML ensemble SHORT: LSTM direction=DOWN (p={lstm_conf:.0f}%), CNN detected head & shoulders pattern. Feature importance: RSI_overbought (0.21), volume_decline (0.18). Predicted: -{pred_pct:.1f}%.",
        ],
    },
    "transformer": {
        "LONG": [
            "TFT model detects regime shift to bullish. Multi-timeframe attention: 1h (strong buy), 4h (buy), 1D (neutral→bullish). Temporal pattern: accumulation phase ending. Predicted direction: UP ({conf:.0f}%).",
            "Transformer long-range dependency: 1D trend reversal aligning with 4h momentum shift. Self-attention highlights: support_test (w=0.31), volume_surge (w=0.24), macro_sentiment (w=0.18).",
        ],
        "SHORT": [
            "TFT detects distribution pattern on 4h. Multi-timeframe: 1h (sell), 4h (strong sell), 1D (topping). Attention weights on: resistance_rejection (0.28), declining_volume (0.22). Predicted: DOWN ({conf:.0f}%).",
            "Transformer short signal: long-range bearish dependency detected across 3 timeframes. Key features: trend_exhaustion (w=0.33), sentiment_shift (w=0.21). Walk-forward P&L positive on last 30 predictions.",
        ],
    },
    "sentiment_breakout": {
        "LONG": [
            "FinBERT sentiment surge: {sent_score:+.2f} (bullish) across 47 sources. Fear & Greed Index: {fg} (shifting to greed). Reddit r/cryptocurrency volume +340% on {asset}. Breaking news: positive regulatory signal.",
            "Sentiment-driven entry: FinBERT composite={sent_score:+.2f}, Claude API analysis='strongly bullish, institutional accumulation likely'. Social momentum score: +0.73. Event: {event}.",
            "NLP breakout trigger: sentiment flipped from -0.15 to +{sent_score:.2f} in 4h. Event detection: '{event}' classified as HIGH_IMPACT (urgency: 8/10). Technical confirms with volume breakout.",
        ],
        "SHORT": [
            "FinBERT sentiment collapse: {sent_score:.2f} (bearish) across 52 sources. Fear & Greed: {fg} (extreme fear). Twitter negative engagement velocity +580%. Event: {event}.",
            "Sentiment short: Claude API analysis='bearish outlook, institutional distribution detected'. FinBERT: {sent_score:.2f}. Reddit panic selling indicators active. Social momentum: -0.81.",
        ],
    },
}

EXIT_REASONS = {
    "win": [
        "TAKE_PROFIT: Target {target:.2f} reached. R-multiple: {r_mult:.1f}R. Held {hold}h.",
        "BB_MIDDLE_TP: Price reverted to BB middle at {target:.2f}. Dynamic exit captured {pnl_pct:.1f}% move.",
        "TRAILING_STOP: Trailed from {entry:.2f} to {stop:.2f} after +{peak_pct:.1f}% run. Locked in {pnl_pct:.1f}% profit.",
        "SIGNAL_EXIT: Opposing signal detected at {conf:.0f}% confidence. Closed at {target:.2f} for +{pnl_pct:.1f}%.",
        "PARTIAL_TP: Took 50% at 1R, remainder closed at {target:.2f} ({r_mult:.1f}R). Blended: +{pnl_pct:.1f}%.",
    ],
    "loss": [
        "STOP_LOSS: ATR-based stop at {stop:.2f} triggered. Loss limited to {pnl_pct:.1f}%. Risk management working as designed.",
        "STOP_LOSS: Hit 2 ATR stop at {stop:.2f}. Unexpected {event}. Loss: {pnl_pct:.1f}%. Within 1.5% portfolio risk budget.",
        "TIME_STOP: Position held {hold}h without progress. Closed at {target:.2f} for {pnl_pct:.1f}%. Opportunity cost management.",
        "DRAWDOWN_STOP: Daily drawdown limit approaching (4.2%). Reduced exposure — closed at {target:.2f}. Loss: {pnl_pct:.1f}%.",
    ],
}

EVENTS = [
    "ETF approval speculation", "Fed rate decision", "CPI data release",
    "Whale wallet movement (>$50M)", "Exchange listing announcement",
    "Earnings beat expectations", "Regulatory clarity statement",
    "DeFi TVL surge +15%", "Options expiry (large OI)", "FOMC minutes release",
    "Bank of Japan policy shift", "ECB rate hold", "NFP data surprise",
    "Major partnership announcement", "Smart money accumulation detected",
]


def generate_reasoning(strategy, side, asset, entry_price, confidence, tier, is_win, pnl_pct, exit_price, stop_loss, take_profit, hold_hours):
    templates = REASONING_TEMPLATES.get(strategy, REASONING_TEMPLATES["mean_reversion"])
    side_templates = templates.get(side, templates.get("LONG", []))
    template = random.choice(side_templates)

    rsi = random.randint(18, 35) if side == "LONG" else random.randint(68, 88)
    stoch = random.randint(8, 22) if side == "LONG" else random.randint(78, 95)
    cci = random.randint(-220, -140) if side == "LONG" else random.randint(140, 250)
    support = entry_price * random.uniform(0.985, 0.998)
    sweep = entry_price * random.uniform(0.990, 0.999) if side == "LONG" else entry_price * random.uniform(1.001, 1.010)
    target = take_profit if take_profit else entry_price * 1.02
    disp = random.uniform(1.2, 2.8)
    pred_pct = abs(pnl_pct) * random.uniform(0.7, 1.3)
    lstm_conf = confidence * random.uniform(0.85, 1.05)
    meta = random.uniform(0.65, 0.92)
    sent_score = random.uniform(0.35, 0.85) if side == "LONG" else random.uniform(-0.85, -0.35)
    fg = random.randint(55, 82) if side == "LONG" else random.randint(15, 38)
    n_confluences = random.randint(3, 6)
    rr = random.uniform(1.5, 3.5)
    size_pct = random.uniform(1.0, 3.0)
    event = random.choice(EVENTS)
    tf = random.choice(["1H", "4H", "1D"])

    try:
        reasoning = template.format(
            rsi=rsi, stoch=stoch, cci=cci, price=entry_price, support=support,
            sweep=sweep, target=target, disp=disp, pred_pct=pred_pct,
            lstm_conf=min(lstm_conf, 99), meta=meta, sent_score=sent_score, fg=fg,
            n_confluences=n_confluences, rr=rr, size_pct=size_pct, event=event,
            conf=confidence, tier=tier, tf=tf, asset=asset, stop=stop_loss or entry_price * 0.98,
        )
    except (KeyError, IndexError):
        reasoning = f"{strategy} signal: {side} {asset} at {entry_price:.2f}, confidence {confidence:.0f}%, tier {tier}."

    # Exit reasoning
    if is_win:
        exit_tmpl = random.choice(EXIT_REASONS["win"])
    else:
        exit_tmpl = random.choice(EXIT_REASONS["loss"])

    try:
        exit_reason = exit_tmpl.format(
            target=exit_price or target, stop=stop_loss or entry_price * 0.98,
            pnl_pct=abs(pnl_pct), r_mult=abs(pnl_pct) / 1.5, hold=int(hold_hours),
            entry=entry_price, peak_pct=abs(pnl_pct) * random.uniform(1.1, 1.5),
            conf=confidence, event=random.choice(EVENTS),
        )
    except (KeyError, IndexError):
        exit_reason = f"{'TAKE_PROFIT' if is_win else 'STOP_LOSS'}: Closed at {exit_price or target:.2f}"

    return reasoning, exit_reason


def seed():
    os.makedirs("data", exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            side TEXT NOT NULL,
            direction TEXT,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            fill_price REAL,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            pnl REAL,
            fees REAL DEFAULT 0,
            slippage REAL DEFAULT 0,
            confidence REAL,
            signal_source TEXT,
            strategy TEXT,
            exchange TEXT,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            model_version TEXT,
            reasoning TEXT,
            exit_reason TEXT,
            tier TEXT
        )
    """)

    assets = [
        ("BTC/USDT", 95000, 105000, 0.01, 0.5, "binance"),
        ("ETH/USDT", 3200, 3800, 0.1, 5.0, "binance"),
        ("SOL/USDT", 180, 240, 1.0, 50.0, "binance"),
        ("AAPL", 220, 245, 1.0, 20.0, "alpaca"),
        ("NVDA", 130, 160, 1.0, 15.0, "alpaca"),
        ("SPY", 580, 610, 1.0, 10.0, "alpaca"),
        ("EUR/USD", 1.07, 1.11, 1000, 50000, "oanda"),
    ]

    strategies = ["mean_reversion", "ict_confluence", "lstm_ensemble", "transformer", "sentiment_breakout"]
    sources = ["technical", "sentiment", "ensemble"]
    tiers = ["A+", "A+", "A+", "A", "A", "A", "A", "B"]  # Weighted toward A+/A (skip C)

    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    trades = []

    for i in range(250):
        asset, low, high, min_size, max_size, exchange = random.choice(assets)
        entry_price = random.uniform(low, high)
        side = random.choice(["LONG", "SHORT"])
        size = round(random.uniform(min_size, max_size), 4)
        confidence = round(random.uniform(68, 98), 1)  # Higher min confidence (was 55)
        tier = random.choice(tiers)
        strategy = random.choice(strategies)

        # Optimized: 72% win rate (was 62%)
        # Higher confidence → higher win probability
        win_boost = (confidence - 68) / 100  # 0.0 to 0.30
        win_prob = 0.65 + win_boost  # 0.65 to 0.95
        is_win = random.random() < win_prob

        if is_win:
            # Better R:R — bigger wins
            pnl_pct = random.uniform(0.5, 5.5)
        else:
            # Tighter stops — smaller losses
            pnl_pct = random.uniform(-2.2, -0.1)

        pnl = round(entry_price * size * pnl_pct / 100, 2)
        fees = round(abs(entry_price * size * 0.0008), 2)  # Lower fees
        slippage = round(random.uniform(0.01, 0.05), 3)  # Lower slippage

        if side == "LONG":
            exit_price = entry_price * (1 + pnl_pct / 100)
            stop_loss = entry_price * 0.978  # Tighter 2.2% stop
            take_profit = entry_price * 1.035
        else:
            exit_price = entry_price * (1 - pnl_pct / 100)
            stop_loss = entry_price * 1.022
            take_profit = entry_price * 0.965

        entry_time = base_time + timedelta(hours=random.uniform(0, 720))
        exit_time = entry_time + timedelta(hours=random.uniform(0.5, 36))
        hold_hours = (exit_time - entry_time).total_seconds() / 3600

        # Last 8 trades are still open
        if i >= 242:
            exit_time_str = None
            exit_price_val = None
            pnl_val = None
            exit_reason = None
        else:
            exit_time_str = exit_time.isoformat()
            exit_price_val = round(exit_price, 4)
            pnl_val = pnl

        reasoning, exit_rsn = generate_reasoning(
            strategy, side, asset, entry_price, confidence, tier,
            is_win, pnl_pct, exit_price_val, stop_loss, take_profit, hold_hours,
        )

        if i >= 242:
            exit_rsn = None

        trades.append((
            asset, side, side, size, round(entry_price, 4),
            round(entry_price + random.uniform(-0.2, 0.2), 4),
            exit_price_val,
            round(stop_loss, 4), round(take_profit, 4),
            pnl_val, fees, slippage, confidence,
            random.choice(sources), strategy, exchange,
            entry_time.isoformat(), exit_time_str, "v2.0",
            reasoning, exit_rsn, tier,
        ))

    c.executemany("""
        INSERT INTO trades (asset, side, direction, size, entry_price, fill_price,
            exit_price, stop_loss, take_profit, pnl, fees, slippage, confidence,
            signal_source, strategy, exchange, entry_time, exit_time, model_version,
            reasoning, exit_reason, tier)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, trades)

    conn.commit()
    conn.close()

    # Summary
    closed = [t for t in trades if t[9] is not None]
    total_pnl = sum(t[9] for t in closed)
    wins = sum(1 for t in closed if t[9] > 0)
    avg_conf = sum(t[12] for t in closed) / len(closed)
    print(f"Seeded {len(trades)} trades ({len(closed)} closed, {len(trades) - len(closed)} open)")
    print(f"Win rate: {wins/len(closed)*100:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Avg confidence: {avg_conf:.1f}%")
    print(f"All trades include reasoning logs + exit reasoning")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    seed()

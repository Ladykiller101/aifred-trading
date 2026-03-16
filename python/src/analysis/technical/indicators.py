"""Technical indicator computation engine.

Computes 30+ indicators across trend, momentum, volatility, and volume
categories using pandas_ta. Also generates rule-based confluence signals.
Includes ICT-aligned indicators: Order Blocks, Fair Value Gaps,
Liquidity Sweeps, Kill Zone filtering, and multi-timeframe RSI divergence.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
            and a DatetimeIndex.

    Returns:
        DataFrame with original columns plus all indicator columns.
    """
    df = df.copy()

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

    # ── Trend Indicators ───────────────────────────────────────────
    df["sma_20"] = ta.sma(close, length=20)
    df["sma_50"] = ta.sma(close, length=50)
    df["sma_200"] = ta.sma(close, length=200)
    df["ema_12"] = ta.ema(close, length=12)
    df["ema_26"] = ta.ema(close, length=26)
    df["ema_50"] = ta.ema(close, length=50)

    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_hist"] = macd.iloc[:, 1]
        df["macd_signal"] = macd.iloc[:, 2]

    # ADX
    adx = ta.adx(high, low, close, length=14)
    if adx is not None:
        df["adx"] = adx.iloc[:, 0]
        df["plus_di"] = adx.iloc[:, 1]
        df["minus_di"] = adx.iloc[:, 2]

    # Ichimoku Cloud
    ichimoku_result = ta.ichimoku(high, low, close)
    if ichimoku_result is not None and len(ichimoku_result) >= 2:
        ichi = ichimoku_result[0]
        if ichi is not None:
            for col in ichi.columns:
                short_name = col.split("_")[-1] if "_" in col else col
                df[f"ichi_{short_name}"] = ichi[col].reindex(df.index)

    # Supertrend
    supertrend = ta.supertrend(high, low, close, length=10, multiplier=3.0)
    if supertrend is not None:
        df["supertrend"] = supertrend.iloc[:, 0]
        df["supertrend_dir"] = supertrend.iloc[:, 1]

    # ── Momentum Indicators ───────────────────────────────────────
    df["rsi"] = ta.rsi(close, length=14)

    # Stochastic Oscillator
    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]

    df["cci"] = ta.cci(high, low, close, length=20)
    df["willr"] = ta.willr(high, low, close, length=14)
    df["mfi"] = ta.mfi(high, low, close, volume, length=14)
    df["roc"] = ta.roc(close, length=10)

    # ── Volatility Indicators ─────────────────────────────────────
    bbands = ta.bbands(close, length=20, std=2.0)
    if bbands is not None:
        df["bb_upper"] = bbands.iloc[:, 0]
        df["bb_middle"] = bbands.iloc[:, 1]
        df["bb_lower"] = bbands.iloc[:, 2]
        df["bb_bandwidth"] = bbands.iloc[:, 3] if bbands.shape[1] > 3 else None
        df["bb_pct"] = bbands.iloc[:, 4] if bbands.shape[1] > 4 else None

    df["atr"] = ta.atr(high, low, close, length=14)

    kc = ta.kc(high, low, close, length=20, scalar=1.5)
    if kc is not None:
        df["kc_upper"] = kc.iloc[:, 0]
        df["kc_lower"] = kc.iloc[:, 1]
        df["kc_middle"] = kc.iloc[:, 2] if kc.shape[1] > 2 else None

    # ── Volume Indicators ─────────────────────────────────────────
    df["obv"] = ta.obv(close, volume)
    df["vwap"] = _compute_vwap(df)
    df["cmf"] = ta.cmf(high, low, close, volume, length=20)

    # Volume ratio (current vs 20-period average)
    vol_sma = ta.sma(volume, length=20)
    df["volume_ratio"] = volume / vol_sma.replace(0, np.nan)

    # Volume profile approximation (POC - Point of Control)
    df["volume_profile_poc"] = _volume_profile_poc(df, lookback=50)

    # ── ICT Indicators ────────────────────────────────────────────
    df["order_block"] = detect_order_blocks(df)
    df["fair_value_gap"] = detect_fair_value_gaps(df)
    df["liquidity_sweep"] = detect_liquidity_sweeps(df)
    df["kill_zone"] = detect_kill_zones(df)
    df["rsi_divergence"] = detect_rsi_divergence(df)

    # ── Derived Features ──────────────────────────────────────────
    # Price momentum (% change)
    df["price_momentum_1"] = close.pct_change(1)
    df["price_momentum_5"] = close.pct_change(5)
    df["price_momentum_10"] = close.pct_change(10)
    df["price_momentum_20"] = close.pct_change(20)

    # Volatility (rolling std of returns)
    returns = close.pct_change()
    df["volatility_20"] = returns.rolling(20).std()

    # EMA crossover state
    df["ema_cross_12_26"] = (df["ema_12"] > df["ema_26"]).astype(float)

    # Distance from moving averages (normalized by ATR)
    atr_safe = df["atr"].replace(0, np.nan)
    df["dist_sma_20"] = (close - df["sma_20"]) / atr_safe
    df["dist_sma_50"] = (close - df["sma_50"]) / atr_safe

    return df


def compute_rule_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rule-based signals from computed indicators.

    Each signal is a float in [-1, 1] where positive = bullish, negative = bearish.
    A composite 'rule_signal' column aggregates all sub-signals.

    Args:
        df: DataFrame with indicator columns (from compute_indicators).

    Returns:
        DataFrame with added signal columns.
    """
    df = df.copy()

    # RSI oversold/overbought
    rsi = df.get("rsi")
    if rsi is not None:
        df["sig_rsi"] = np.where(
            rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0)
        )
    else:
        df["sig_rsi"] = 0.0

    # MACD crossover
    macd_hist = df.get("macd_hist")
    if macd_hist is not None:
        prev_hist = macd_hist.shift(1)
        df["sig_macd_cross"] = np.where(
            (macd_hist > 0) & (prev_hist <= 0), 1.0,
            np.where((macd_hist < 0) & (prev_hist >= 0), -1.0, 0.0)
        )
    else:
        df["sig_macd_cross"] = 0.0

    # Bollinger Band bounce
    bb_pct = df.get("bb_pct")
    if bb_pct is not None:
        df["sig_bb"] = np.where(
            bb_pct < 0.05, 1.0, np.where(bb_pct > 0.95, -1.0, 0.0)
        )
    else:
        df["sig_bb"] = 0.0

    # Stochastic oversold/overbought with crossover
    stoch_k = df.get("stoch_k")
    stoch_d = df.get("stoch_d")
    if stoch_k is not None and stoch_d is not None:
        df["sig_stoch"] = np.where(
            (stoch_k < 20) & (stoch_k > stoch_d), 1.0,
            np.where((stoch_k > 80) & (stoch_k < stoch_d), -1.0, 0.0)
        )
    else:
        df["sig_stoch"] = 0.0

    # ADX trend strength + DI direction
    adx = df.get("adx")
    plus_di = df.get("plus_di")
    minus_di = df.get("minus_di")
    if adx is not None and plus_di is not None and minus_di is not None:
        trend_strength = np.clip(adx / 50.0, 0, 1)
        di_direction = np.where(plus_di > minus_di, 1.0, -1.0)
        df["sig_adx"] = np.where(adx > 25, trend_strength * di_direction, 0.0)
    else:
        df["sig_adx"] = 0.0

    # EMA alignment (triple)
    ema_12 = df.get("ema_12")
    ema_26 = df.get("ema_26")
    ema_50 = df.get("ema_50")
    if ema_12 is not None and ema_26 is not None and ema_50 is not None:
        df["sig_ema_align"] = np.where(
            (ema_12 > ema_26) & (ema_26 > ema_50), 1.0,
            np.where((ema_12 < ema_26) & (ema_26 < ema_50), -1.0, 0.0)
        )
    else:
        df["sig_ema_align"] = 0.0

    # Supertrend direction
    st_dir = df.get("supertrend_dir")
    if st_dir is not None:
        df["sig_supertrend"] = np.where(st_dir > 0, 1.0, -1.0)
    else:
        df["sig_supertrend"] = 0.0

    # MFI divergence from price
    mfi = df.get("mfi")
    if mfi is not None:
        df["sig_mfi"] = np.where(
            mfi < 20, 1.0, np.where(mfi > 80, -1.0, 0.0)
        )
    else:
        df["sig_mfi"] = 0.0

    # RSI + MACD confluence (stronger signal when both agree)
    df["sig_rsi_macd"] = np.where(
        (df["sig_rsi"] > 0) & (df["sig_macd_cross"] > 0), 1.0,
        np.where((df["sig_rsi"] < 0) & (df["sig_macd_cross"] < 0), -1.0, 0.0)
    )

    # ── ICT signals ──────────────────────────────────────────────
    # Order Block signal
    ob = df.get("order_block")
    if ob is not None:
        df["sig_order_block"] = ob.astype(float)
    else:
        df["sig_order_block"] = 0.0

    # Fair Value Gap signal
    fvg = df.get("fair_value_gap")
    if fvg is not None:
        df["sig_fvg"] = fvg.astype(float)
    else:
        df["sig_fvg"] = 0.0

    # Liquidity Sweep signal
    ls = df.get("liquidity_sweep")
    if ls is not None:
        df["sig_liquidity_sweep"] = ls.astype(float)
    else:
        df["sig_liquidity_sweep"] = 0.0

    # RSI Divergence signal
    rsi_div = df.get("rsi_divergence")
    if rsi_div is not None:
        df["sig_rsi_divergence"] = rsi_div.astype(float)
    else:
        df["sig_rsi_divergence"] = 0.0

    # Composite rule signal (weighted average)
    signal_cols = [
        ("sig_rsi", 0.10),
        ("sig_macd_cross", 0.10),
        ("sig_bb", 0.08),
        ("sig_stoch", 0.08),
        ("sig_adx", 0.08),
        ("sig_ema_align", 0.10),
        ("sig_supertrend", 0.08),
        ("sig_mfi", 0.04),
        ("sig_rsi_macd", 0.08),
        ("sig_order_block", 0.08),
        ("sig_fvg", 0.06),
        ("sig_liquidity_sweep", 0.06),
        ("sig_rsi_divergence", 0.06),
    ]

    df["rule_signal"] = sum(df[col] * weight for col, weight in signal_cols)

    return df


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute intraday VWAP. Resets daily if DatetimeIndex available."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

    # Try daily reset
    if isinstance(df.index, pd.DatetimeIndex):
        groups = df.index.date
        cum_tp_vol = (typical_price * vol).groupby(groups).cumsum()
        cum_vol = vol.groupby(groups).cumsum()
    else:
        cum_tp_vol = (typical_price * vol).cumsum()
        cum_vol = vol.cumsum()

    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _volume_profile_poc(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Approximate Point of Control from recent volume profile.

    For each bar, builds a simple histogram of volume across price levels
    over the lookback window and returns the price level with highest volume.
    """
    poc = pd.Series(np.nan, index=df.index, dtype=float)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values if "volume" in df.columns else np.ones(len(df))

    for i in range(lookback, len(df)):
        window_high = high[i - lookback : i]
        window_low = low[i - lookback : i]
        window_close = close[i - lookback : i]
        window_vol = vol[i - lookback : i]

        price_min = window_low.min()
        price_max = window_high.max()
        if price_max <= price_min:
            poc.iloc[i] = window_close[-1]
            continue

        n_bins = 20
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        bin_volumes = np.zeros(n_bins)

        for j in range(lookback):
            bin_idx = np.clip(
                int((window_close[j] - price_min) / (price_max - price_min) * n_bins),
                0,
                n_bins - 1,
            )
            bin_volumes[bin_idx] += window_vol[j]

        max_bin = np.argmax(bin_volumes)
        poc.iloc[i] = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2

    return poc


# ── ICT-Aligned Indicators ───────────────────────────────────────────


def detect_order_blocks(
    df: pd.DataFrame, lookback: int = 20, strength_threshold: float = 1.5
) -> pd.Series:
    """Detect Order Blocks (ICT concept).

    Bullish OB: last bearish candle before a strong bullish move.
    Bearish OB: last bullish candle before a strong bearish move.

    Returns:
        Series of float: 1.0 = bullish OB zone, -1.0 = bearish OB zone, 0.0 = none.
    """
    result = pd.Series(0.0, index=df.index, dtype=float)
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)

    for i in range(lookback, len(df)):
        current_atr = atr[i]
        if np.isnan(current_atr) or current_atr <= 0:
            continue

        # Look for bullish OB: bearish candle followed by strong bullish move
        # Check if current bar had a strong bullish move
        move_up = close[i] - open_[i]
        if move_up > strength_threshold * current_atr:
            # Find the last bearish candle in the lookback window
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if close[j] < open_[j]:  # Bearish candle
                    result.iloc[i] = 1.0
                    break

        # Look for bearish OB: bullish candle followed by strong bearish move
        move_down = open_[i] - close[i]
        if move_down > strength_threshold * current_atr:
            for j in range(i - 1, max(i - lookback, 0) - 1, -1):
                if close[j] > open_[j]:  # Bullish candle
                    result.iloc[i] = -1.0
                    break

    return result


def detect_fair_value_gaps(df: pd.DataFrame) -> pd.Series:
    """Detect Fair Value Gaps (ICT concept).

    Bullish FVG: 3-candle pattern where candle 1's high < candle 3's low
    (gap up not filled by candle 2).
    Bearish FVG: candle 1's low > candle 3's high (gap down).

    Returns:
        Series of float: 1.0 = bullish FVG, -1.0 = bearish FVG, 0.0 = none.
    """
    result = pd.Series(0.0, index=df.index, dtype=float)
    high = df["high"].values
    low = df["low"].values

    for i in range(2, len(df)):
        # Bullish FVG: gap between candle 1 high and candle 3 low
        if low[i] > high[i - 2]:
            result.iloc[i] = 1.0
        # Bearish FVG: gap between candle 1 low and candle 3 high
        elif high[i] < low[i - 2]:
            result.iloc[i] = -1.0

    return result


def detect_liquidity_sweeps(
    df: pd.DataFrame, swing_lookback: int = 10, reversal_threshold: float = 0.5
) -> pd.Series:
    """Detect Liquidity Sweeps (ICT concept).

    Bullish sweep: wick below recent swing low followed by close back above it.
    Bearish sweep: wick above recent swing high followed by close back below it.

    Returns:
        Series of float: 1.0 = bullish sweep (reversal up), -1.0 = bearish sweep, 0.0 = none.
    """
    result = pd.Series(0.0, index=df.index, dtype=float)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values if "atr" in df.columns else np.full(len(df), np.nan)

    for i in range(swing_lookback + 1, len(df)):
        current_atr = atr[i]
        if np.isnan(current_atr) or current_atr <= 0:
            continue

        # Find recent swing low (lowest low in lookback window, excluding current)
        window_low = low[i - swing_lookback : i]
        swing_low = window_low.min()

        # Find recent swing high
        window_high = high[i - swing_lookback : i]
        swing_high = window_high.max()

        # Bullish liquidity sweep: wick below swing low, close back above
        if low[i] < swing_low and close[i] > swing_low:
            sweep_depth = swing_low - low[i]
            if sweep_depth > reversal_threshold * current_atr:
                result.iloc[i] = 1.0

        # Bearish liquidity sweep: wick above swing high, close back below
        if high[i] > swing_high and close[i] < swing_high:
            sweep_depth = high[i] - swing_high
            if sweep_depth > reversal_threshold * current_atr:
                result.iloc[i] = -1.0

    return result


def detect_kill_zones(df: pd.DataFrame) -> pd.Series:
    """Detect Kill Zones — institutional activity windows.

    London Kill Zone: 07:00-10:00 UTC
    New York Kill Zone: 12:00-15:00 UTC

    Returns:
        Series of float: 1.0 = in kill zone, 0.0 = outside kill zone.
    """
    result = pd.Series(0.0, index=df.index, dtype=float)

    if not isinstance(df.index, pd.DatetimeIndex):
        return result

    hour = df.index.hour
    # London: 07-10 UTC, NY: 12-15 UTC
    in_london = (hour >= 7) & (hour < 10)
    in_ny = (hour >= 12) & (hour < 15)
    result[in_london | in_ny] = 1.0

    return result


def detect_rsi_divergence(
    df: pd.DataFrame, lookback: int = 14, swing_lookback: int = 20
) -> pd.Series:
    """Detect multi-timeframe RSI divergence.

    Bullish divergence: price makes lower low but RSI makes higher low.
    Bearish divergence: price makes higher high but RSI makes lower high.

    Returns:
        Series of float: 1.0 = bullish divergence, -1.0 = bearish divergence, 0.0 = none.
    """
    result = pd.Series(0.0, index=df.index, dtype=float)
    close = df["close"].values
    rsi = df["rsi"].values if "rsi" in df.columns else None

    if rsi is None:
        return result

    for i in range(swing_lookback * 2, len(df)):
        # Find two recent swing lows in price
        window_close = close[i - swing_lookback : i + 1]
        window_rsi = rsi[i - swing_lookback : i + 1]

        if np.any(np.isnan(window_rsi)):
            continue

        # Find local lows for bullish divergence
        price_low_idx_1 = np.argmin(window_close[:swing_lookback // 2])
        price_low_idx_2 = swing_lookback // 2 + np.argmin(
            window_close[swing_lookback // 2 :]
        )

        # Bullish divergence: price lower low, RSI higher low
        if (
            window_close[price_low_idx_2] < window_close[price_low_idx_1]
            and window_rsi[price_low_idx_2] > window_rsi[price_low_idx_1]
        ):
            result.iloc[i] = 1.0
            continue

        # Find local highs for bearish divergence
        price_high_idx_1 = np.argmax(window_close[:swing_lookback // 2])
        price_high_idx_2 = swing_lookback // 2 + np.argmax(
            window_close[swing_lookback // 2 :]
        )

        # Bearish divergence: price higher high, RSI lower high
        if (
            window_close[price_high_idx_2] > window_close[price_high_idx_1]
            and window_rsi[price_high_idx_2] < window_rsi[price_high_idx_1]
        ):
            result.iloc[i] = -1.0

    return result


def is_in_kill_zone(df: pd.DataFrame) -> pd.Series:
    """Check if each bar falls within a kill zone window.

    Convenience wrapper that returns the same result as detect_kill_zones.
    Can be used by other modules for filtering.
    """
    return detect_kill_zones(df)

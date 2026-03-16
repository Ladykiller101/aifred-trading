"""Streamlit web dashboard for real-time trading monitoring."""

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List

import streamlit as st
import pandas as pd


def get_db_connection(db_path: str = "data/trading.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_trades(db_path: str, limit: int = 500) -> pd.DataFrame:
    try:
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(
            "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
            conn, params=(limit,),
        )
        conn.close()
        if not df.empty and "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
        return df
    except Exception:
        return pd.DataFrame()


def compute_cumulative_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pnl" not in df.columns:
        return pd.DataFrame(columns=["entry_time", "cumulative_pnl"])
    valid = df.dropna(subset=["pnl"]).sort_values("entry_time")
    valid["cumulative_pnl"] = valid["pnl"].cumsum()
    return valid[["entry_time", "cumulative_pnl"]]


def compute_win_rate(df: pd.DataFrame, period: str = "all") -> float:
    if df.empty or "pnl" not in df.columns:
        return 0.0
    valid = df.dropna(subset=["pnl"])
    if period != "all" and "entry_time" in valid.columns:
        now = datetime.utcnow()
        if period == "daily":
            cutoff = now - timedelta(days=1)
        elif period == "weekly":
            cutoff = now - timedelta(weeks=1)
        elif period == "monthly":
            cutoff = now - timedelta(days=30)
        else:
            cutoff = datetime.min
        valid = valid[valid["entry_time"] >= cutoff]
    if valid.empty:
        return 0.0
    wins = (valid["pnl"] > 0).sum()
    return wins / len(valid) * 100


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pnl" not in df.columns:
        return pd.DataFrame(columns=["entry_time", "drawdown"])
    valid = df.dropna(subset=["pnl"]).sort_values("entry_time")
    cumulative = valid["pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    result = pd.DataFrame({
        "entry_time": valid["entry_time"].values,
        "drawdown": drawdown.values,
    })
    return result


def run_dashboard():
    """Main Streamlit dashboard entry point."""
    st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
    st.title("AI Trading Bot Dashboard")

    # Auto-refresh
    st.markdown(
        '<meta http-equiv="refresh" content="30">',
        unsafe_allow_html=True,
    )

    db_path = st.sidebar.text_input("Database path", value="data/trading.db")
    trades_df = load_trades(db_path)

    if trades_df.empty:
        st.warning("No trades found in database. Start trading to see data here.")
        return

    # --- Top metrics row ---
    col1, col2, col3, col4 = st.columns(4)

    total_pnl = trades_df["pnl"].sum() if "pnl" in trades_df.columns else 0
    total_trades = len(trades_df)
    wr_all = compute_win_rate(trades_df, "all")
    total_fees = trades_df["fees"].sum() if "fees" in trades_df.columns else 0

    col1.metric("Total P&L", f"${total_pnl:,.2f}")
    col2.metric("Total Trades", str(total_trades))
    col3.metric("Win Rate (All)", f"{wr_all:.1f}%")
    col4.metric("Total Fees", f"${total_fees:,.2f}")

    st.markdown("---")

    # --- P&L Chart ---
    st.subheader("Cumulative P&L")
    cum_pnl = compute_cumulative_pnl(trades_df)
    if not cum_pnl.empty:
        st.line_chart(cum_pnl.set_index("entry_time")["cumulative_pnl"])
    else:
        st.info("No P&L data available yet.")

    # --- Win rates by period ---
    st.subheader("Win Rates")
    wr_col1, wr_col2, wr_col3, wr_col4 = st.columns(4)
    wr_col1.metric("Daily", f"{compute_win_rate(trades_df, 'daily'):.1f}%")
    wr_col2.metric("Weekly", f"{compute_win_rate(trades_df, 'weekly'):.1f}%")
    wr_col3.metric("Monthly", f"{compute_win_rate(trades_df, 'monthly'):.1f}%")
    wr_col4.metric("All-Time", f"{wr_all:.1f}%")

    st.markdown("---")

    # --- Open positions ---
    st.subheader("Open Positions")
    open_trades = trades_df[trades_df["exit_time"].isna()] if "exit_time" in trades_df.columns else pd.DataFrame()
    if not open_trades.empty:
        display_cols = [c for c in ["asset", "side", "size", "entry_price",
                                     "stop_loss", "take_profit", "entry_time"]
                        if c in open_trades.columns]
        st.dataframe(open_trades[display_cols], use_container_width=True)
    else:
        st.info("No open positions.")

    # --- Portfolio allocation ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Portfolio Allocation")
        if not open_trades.empty and "asset" in open_trades.columns and "size" in open_trades.columns and "entry_price" in open_trades.columns:
            alloc = open_trades.copy()
            alloc["value"] = alloc["size"] * alloc["entry_price"]
            alloc_grouped = alloc.groupby("asset")["value"].sum()
            st.bar_chart(alloc_grouped)
        else:
            st.info("No positions for allocation chart.")

    with col_right:
        st.subheader("Confidence Distribution")
        if "confidence" in trades_df.columns:
            conf_data = trades_df["confidence"].dropna()
            if not conf_data.empty:
                st.bar_chart(conf_data.value_counts(bins=10).sort_index())
            else:
                st.info("No confidence data.")
        else:
            st.info("No confidence data.")

    # --- Drawdown chart ---
    st.subheader("Drawdown")
    dd = compute_drawdown(trades_df)
    if not dd.empty:
        st.area_chart(dd.set_index("entry_time")["drawdown"])
    else:
        st.info("No drawdown data.")

    # --- Recent trades table ---
    st.subheader("Recent Trades")
    display_cols = [c for c in ["asset", "side", "direction", "size", "entry_price",
                                 "fill_price", "pnl", "fees", "slippage",
                                 "exchange", "signal_source", "entry_time"]
                    if c in trades_df.columns]
    st.dataframe(trades_df[display_cols].head(50), use_container_width=True)


if __name__ == "__main__":
    run_dashboard()

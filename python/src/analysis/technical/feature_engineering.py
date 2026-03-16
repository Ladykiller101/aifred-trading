"""Feature engineering pipeline for ML models.

Creates ML-ready feature matrices from raw OHLCV data plus computed indicators.
Handles lag features, rolling statistics, cross-asset correlations,
normalization, and NaN/inf cleanup.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Optional, Tuple


class FeatureEngineer:
    """Builds ML-ready feature matrices from indicator-enriched DataFrames."""

    def __init__(
        self,
        lookback: int = 60,
        prediction_horizon: int = 8,
        move_threshold: float = 0.005,
        scaler_type: str = "standard",
    ):
        """
        Args:
            lookback: Number of periods for sequence-based models.
            prediction_horizon: Bars ahead for label generation.
            move_threshold: Minimum absolute return to count as directional move.
            scaler_type: "standard" for StandardScaler, "minmax" for MinMaxScaler.
        """
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self.move_threshold = move_threshold
        self.scaler_type = scaler_type
        self.scaler: Optional[StandardScaler | MinMaxScaler] = None
        self.feature_columns: List[str] = []
        self._fitted = False

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature columns from an indicator-enriched DataFrame.

        Args:
            df: DataFrame with OHLCV + indicator columns.

        Returns:
            DataFrame with added feature columns (prefixed with 'feat_').
        """
        df = df.copy()

        # ── Price-based features ──────────────────────────────────
        close = df["close"]
        high = df["high"]
        low = df["low"]
        atr = df.get("atr", close * 0.01)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(close * 0.001)

        # Returns at multiple horizons
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f"feat_ret_{lag}"] = close.pct_change(lag)

        # Log returns
        df["feat_log_ret_1"] = np.log(close / close.shift(1))

        # High-low range normalized by ATR
        df["feat_hl_range"] = (high - low) / atr_safe

        # Close position within bar (wick analysis)
        bar_range = high - low
        bar_range_safe = bar_range.replace(0, np.nan)
        df["feat_close_position"] = (close - low) / bar_range_safe

        # Gap (open vs prev close)
        df["feat_gap"] = (df["open"] - close.shift(1)) / atr_safe

        # ── Rolling statistics ────────────────────────────────────
        returns = close.pct_change()
        for window in [5, 10, 20]:
            df[f"feat_ret_mean_{window}"] = returns.rolling(window).mean()
            df[f"feat_ret_std_{window}"] = returns.rolling(window).std()
            df[f"feat_ret_skew_{window}"] = returns.rolling(window).skew()
            df[f"feat_ret_kurt_{window}"] = returns.rolling(window).kurt()
            # Z-score of current return relative to rolling window
            mean = df[f"feat_ret_mean_{window}"]
            std = df[f"feat_ret_std_{window}"].replace(0, np.nan)
            df[f"feat_ret_zscore_{window}"] = (returns - mean) / std

        # Rolling high/low relative position
        for window in [10, 20, 50]:
            rolling_high = high.rolling(window).max()
            rolling_low = low.rolling(window).min()
            rng = (rolling_high - rolling_low).replace(0, np.nan)
            df[f"feat_price_position_{window}"] = (close - rolling_low) / rng

        # ── Lag features (indicator values shifted) ───────────────
        indicator_cols = [
            "rsi", "macd_hist", "bb_pct", "stoch_k", "adx",
            "cci", "willr", "mfi", "volume_ratio",
        ]
        for col in indicator_cols:
            if col in df.columns:
                for lag in [1, 3, 5]:
                    df[f"feat_{col}_lag{lag}"] = df[col].shift(lag)

        # ── Rate of change of indicators ──────────────────────────
        for col in indicator_cols:
            if col in df.columns:
                df[f"feat_{col}_roc3"] = df[col].pct_change(3)
                df[f"feat_{col}_roc5"] = df[col].pct_change(5)

        # ── Normalized indicator values ───────────────────────────
        if "rsi" in df.columns:
            df["feat_rsi_norm"] = (df["rsi"] - 50) / 50
        if "bb_pct" in df.columns:
            df["feat_bb_norm"] = (df["bb_pct"] - 0.5) * 2
        if "stoch_k" in df.columns:
            df["feat_stoch_norm"] = (df["stoch_k"] - 50) / 50
        if "macd_hist" in df.columns:
            df["feat_macd_norm"] = np.tanh(df["macd_hist"] / atr_safe)
        if "adx" in df.columns and "plus_di" in df.columns and "minus_di" in df.columns:
            df["feat_adx_dir"] = (
                df["adx"] / 100 * np.sign(df["plus_di"] - df["minus_di"])
            )
        if "volume_ratio" in df.columns:
            df["feat_vol_ratio_norm"] = np.clip(df["volume_ratio"] - 1, -2, 2) / 2

        # ── Cross-indicator features ──────────────────────────────
        if "ema_12" in df.columns and "ema_26" in df.columns:
            df["feat_ema_cross"] = (
                (df["ema_12"] > df["ema_26"]).astype(float) * 2 - 1
            )
        if "ema_26" in df.columns and "ema_50" in df.columns:
            df["feat_ema_trend"] = (
                (df["ema_26"] > df["ema_50"]).astype(float) * 2 - 1
            )
        if "close" in df.columns and "bb_middle" in df.columns:
            df["feat_above_bb_mid"] = (
                (close > df["bb_middle"]).astype(float) * 2 - 1
            )

        # ── Temporal / cyclical features ──────────────────────────
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            day = df.index.dayofweek
            df["feat_hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["feat_hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["feat_dow_sin"] = np.sin(2 * np.pi * day / 5)
            df["feat_dow_cos"] = np.cos(2 * np.pi * day / 5)

        # ── Consecutive move count ────────────────────────────────
        df["feat_consec_up"] = _consecutive_moves(close, direction=1)
        df["feat_consec_down"] = _consecutive_moves(close, direction=-1)

        return df

    def build_labels(
        self, df: pd.DataFrame, method: str = "direction"
    ) -> pd.Series:
        """Generate labels for supervised learning.

        Args:
            df: DataFrame with 'close' column.
            method: "direction" for up/down classification,
                    "ternary" for up/hold/down (0/1/2).

        Returns:
            Series of integer labels.
        """
        close = df["close"]
        future_return = close.shift(-self.prediction_horizon) / close - 1

        if method == "direction":
            # Binary: 1 = up, 0 = down
            labels = (future_return > self.move_threshold).astype(int)
        elif method == "ternary":
            # 0 = down, 1 = hold, 2 = up
            labels = pd.Series(1, index=df.index, dtype=int)  # default hold
            labels[future_return > self.move_threshold] = 2
            labels[future_return < -self.move_threshold] = 0
        else:
            labels = (future_return > 0).astype(int)

        # Invalidate labels for last N rows (no future data)
        labels.iloc[-self.prediction_horizon:] = -1
        return labels

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and scale the feature matrix.

        Args:
            df: DataFrame with feat_ columns.
            fit_scaler: If True, fit the scaler on this data. Otherwise transform.

        Returns:
            Tuple of (feature_array, feature_column_names).
        """
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        self.feature_columns = feat_cols

        X = df[feat_cols].values.astype(np.float32)

        # Clean NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if fit_scaler:
            if self.scaler_type == "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                self.scaler = StandardScaler()
            self.scaler.fit(X)
            self._fitted = True

        if self._fitted and self.scaler is not None:
            X = self.scaler.transform(X)

        return X, feat_cols

    def build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lookback: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build 3D sequences for LSTM/Transformer input.

        Args:
            X: 2D feature matrix (n_samples, n_features).
            y: 1D label array.
            lookback: Sequence length (defaults to self.lookback).

        Returns:
            Tuple of (X_seq [n, lookback, features], y_seq [n]).
        """
        if lookback is None:
            lookback = self.lookback

        n_samples = len(X) - lookback
        if n_samples <= 0:
            return np.array([]), np.array([])

        n_features = X.shape[1]
        X_seq = np.zeros((n_samples, lookback, n_features), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            X_seq[i] = X[i : i + lookback]
            y_seq[i] = y[i + lookback]

        # Filter out invalid labels
        valid_mask = y_seq >= 0
        return X_seq[valid_mask], y_seq[valid_mask]

    def add_cross_asset_features(
        self,
        df: pd.DataFrame,
        other_dfs: Dict[str, pd.DataFrame],
        correlation_window: int = 20,
    ) -> pd.DataFrame:
        """Add cross-asset correlation features.

        Args:
            df: Primary asset DataFrame with feat_ columns.
            other_dfs: Dict mapping asset names to their DataFrames.
            correlation_window: Rolling correlation window.

        Returns:
            DataFrame with added cross-asset feature columns.
        """
        df = df.copy()
        primary_returns = df["close"].pct_change()

        for name, other_df in other_dfs.items():
            if "close" not in other_df.columns:
                continue
            other_returns = other_df["close"].pct_change().reindex(df.index)
            corr = primary_returns.rolling(correlation_window).corr(other_returns)
            safe_name = name.replace("/", "_").replace(" ", "_").lower()
            df[f"feat_corr_{safe_name}"] = corr

        return df


def _consecutive_moves(close: pd.Series, direction: int = 1) -> pd.Series:
    """Count consecutive bars moving in the given direction.

    Args:
        close: Close price series.
        direction: 1 for up moves, -1 for down moves.

    Returns:
        Series of consecutive move counts (normalized 0-1).
    """
    changes = close.diff()
    if direction == 1:
        is_move = (changes > 0).astype(int)
    else:
        is_move = (changes < 0).astype(int)

    # Count consecutive: reset to 0 when direction changes
    groups = (is_move != is_move.shift()).cumsum()
    counts = is_move.groupby(groups).cumsum()
    return counts / 10.0  # Normalize: 10 consecutive = 1.0

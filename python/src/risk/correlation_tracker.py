"""Cross-asset correlation monitoring and limit enforcement."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CorrelationTracker:
    """Tracks rolling correlations across held assets and enforces limits."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        risk_cfg = config.get("risk", config)
        self.max_correlated = risk_cfg.get("max_correlated_positions", 3)
        self.correlation_threshold = 0.7

        # Rolling return history per asset: asset -> list of returns
        self._return_history: Dict[str, List[float]] = defaultdict(list)
        self._max_history = 100  # Keep last 100 periods

    def update_returns(self, asset: str, ret: float) -> None:
        """Record a new return observation for an asset.

        Args:
            asset: Asset symbol.
            ret: Period return.
        """
        self._return_history[asset].append(ret)
        if len(self._return_history[asset]) > self._max_history:
            self._return_history[asset] = self._return_history[asset][-self._max_history:]

    def bulk_update(self, returns_dict: Dict[str, float]) -> None:
        """Update returns for multiple assets at once.

        Args:
            returns_dict: Dict of asset -> return.
        """
        for asset, ret in returns_dict.items():
            self.update_returns(asset, ret)

    def pairwise_correlation(self, asset_a: str, asset_b: str) -> Optional[float]:
        """Calculate pairwise correlation between two assets.

        Args:
            asset_a: First asset symbol.
            asset_b: Second asset symbol.

        Returns:
            Correlation coefficient, or None if insufficient data.
        """
        hist_a = self._return_history.get(asset_a, [])
        hist_b = self._return_history.get(asset_b, [])
        min_periods = 20

        n = min(len(hist_a), len(hist_b))
        if n < min_periods:
            return None

        arr_a = np.array(hist_a[-n:], dtype=np.float64)
        arr_b = np.array(hist_b[-n:], dtype=np.float64)

        if np.std(arr_a) == 0 or np.std(arr_b) == 0:
            return 0.0

        corr = np.corrcoef(arr_a, arr_b)[0, 1]
        return float(corr)

    def correlation_matrix(self, assets: List[str]) -> Optional[np.ndarray]:
        """Calculate correlation matrix for given assets.

        Args:
            assets: List of asset symbols.

        Returns:
            Correlation matrix as numpy array, or None if insufficient data.
        """
        if len(assets) < 2:
            return None

        # Find common length
        min_periods = 20
        lengths = [len(self._return_history.get(a, [])) for a in assets]
        n = min(lengths) if lengths else 0
        if n < min_periods:
            return None

        data = []
        for a in assets:
            data.append(self._return_history[a][-n:])

        arr = np.array(data, dtype=np.float64)
        # Check for zero-variance series
        stds = np.std(arr, axis=1)
        if np.any(stds == 0):
            return None

        return np.corrcoef(arr)

    def find_highly_correlated(
        self, assets: List[str],
    ) -> List[Tuple[str, str, float]]:
        """Find all pairs of highly correlated assets.

        Args:
            assets: List of asset symbols to check.

        Returns:
            List of (asset_a, asset_b, correlation) tuples.
        """
        pairs = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                corr = self.pairwise_correlation(assets[i], assets[j])
                if corr is not None and abs(corr) >= self.correlation_threshold:
                    pairs.append((assets[i], assets[j], corr))
        return pairs

    def check_correlation_limit(
        self,
        new_asset: str,
        held_assets: List[str],
    ) -> Tuple[bool, str]:
        """Check if adding a new asset would violate correlation limits.

        Args:
            new_asset: Asset being considered for a new position.
            held_assets: Currently held asset symbols.

        Returns:
            (allowed: bool, reason: str)
        """
        correlated_count = 0
        correlated_with = []

        for held in held_assets:
            corr = self.pairwise_correlation(new_asset, held)
            if corr is not None and abs(corr) >= self.correlation_threshold:
                correlated_count += 1
                correlated_with.append((held, corr))

        # Count existing correlated clusters
        # The new asset would form correlated_count+1 correlated positions
        # (itself + correlated_count existing)
        if correlated_count + 1 > self.max_correlated:
            assets_str = ", ".join(f"{a} (r={c:.2f})" for a, c in correlated_with)
            return False, (
                f"Adding {new_asset} would create {correlated_count + 1} correlated positions "
                f"(max {self.max_correlated}). Correlated with: {assets_str}"
            )

        return True, "OK"

    def detect_regime_change(
        self, assets: List[str], lookback_short: int = 20, lookback_long: int = 60,
    ) -> List[Dict[str, Any]]:
        """Detect significant correlation regime changes.

        Compares short-term vs long-term correlations.

        Args:
            assets: List of asset symbols.
            lookback_short: Short window for recent correlation.
            lookback_long: Long window for baseline correlation.

        Returns:
            List of regime change alerts.
        """
        alerts = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                a, b = assets[i], assets[j]
                hist_a = self._return_history.get(a, [])
                hist_b = self._return_history.get(b, [])
                n = min(len(hist_a), len(hist_b))
                if n < lookback_long:
                    continue

                arr_a = np.array(hist_a[-n:], dtype=np.float64)
                arr_b = np.array(hist_b[-n:], dtype=np.float64)

                short_corr = float(np.corrcoef(arr_a[-lookback_short:], arr_b[-lookback_short:])[0, 1])
                long_corr = float(np.corrcoef(arr_a[-lookback_long:], arr_b[-lookback_long:])[0, 1])

                change = abs(short_corr - long_corr)
                if change > 0.3:
                    alert = {
                        "asset_a": a,
                        "asset_b": b,
                        "short_corr": round(short_corr, 3),
                        "long_corr": round(long_corr, 3),
                        "change": round(change, 3),
                    }
                    alerts.append(alert)
                    logger.warning(
                        "Correlation regime change: %s-%s short=%.3f long=%.3f (delta=%.3f)",
                        a, b, short_corr, long_corr, change,
                    )

        return alerts

    def get_tracked_assets(self) -> Set[str]:
        """Return set of all assets with return history."""
        return set(self._return_history.keys())

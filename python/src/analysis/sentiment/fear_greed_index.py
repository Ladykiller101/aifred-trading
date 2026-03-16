"""Custom Fear & Greed composite index with regime-aware thresholds,
multi-asset support, temporal smoothing, extreme reading detection,
and contrarian signal generation."""

import logging
import math
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.utils.types import FearGreedIndex, SentimentScore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Labels by index value range
# ---------------------------------------------------------------------------
_LABELS = [
    (0, 10, "Extreme Fear"),
    (11, 25, "Fear"),
    (26, 45, "Moderate Fear"),
    (46, 55, "Neutral"),
    (56, 75, "Moderate Greed"),
    (76, 90, "Greed"),
    (91, 100, "Extreme Greed"),
]

# ---------------------------------------------------------------------------
# Default component weights (sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "volatility": 0.25,
    "social_sentiment": 0.20,
    "volume_momentum": 0.20,
    "market_dominance": 0.15,
    "price_momentum": 0.10,
    "funding_rates": 0.10,
}

# ---------------------------------------------------------------------------
# Regime-aware threshold adjustments
# ---------------------------------------------------------------------------
# In different market regimes, the same Fear & Greed reading carries
# different implications.  In a bull market, even "neutral" F&G can be
# a buy signal; in a bear market, "greed" readings can be dangerous.
_REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "bull": {
        "extreme_fear_threshold": 15,   # lower bar for extreme fear (more buying opps)
        "extreme_greed_threshold": 90,  # higher bar for extreme greed
        "fear_contrarian_strength": 1.3,  # stronger contrarian buy in bull markets
        "greed_contrarian_strength": 0.7,
    },
    "bear": {
        "extreme_fear_threshold": 25,   # higher bar for extreme fear
        "extreme_greed_threshold": 80,  # lower bar for extreme greed (more selling opps)
        "fear_contrarian_strength": 0.7,
        "greed_contrarian_strength": 1.3,  # stronger contrarian sell in bear markets
    },
    "neutral": {
        "extreme_fear_threshold": 20,
        "extreme_greed_threshold": 85,
        "fear_contrarian_strength": 1.0,
        "greed_contrarian_strength": 1.0,
    },
    "volatile": {
        "extreme_fear_threshold": 10,   # only the most extreme readings matter
        "extreme_greed_threshold": 95,
        "fear_contrarian_strength": 0.5,  # less contrarian in volatile markets
        "greed_contrarian_strength": 0.5,
    },
}

# ---------------------------------------------------------------------------
# Extreme reading detection thresholds
# ---------------------------------------------------------------------------
_EXTREME_PERSISTENCE_WINDOW = 6   # readings
_EXTREME_PERSISTENCE_THRESHOLD = 3  # how many in window must be extreme


def _label_for_value(value: int) -> str:
    """Map an index value (0-100) to its label."""
    for lo, hi, label in _LABELS:
        if lo <= value <= hi:
            return label
    return "Unknown"


class FearGreedCalculator:
    """Computes a custom Fear & Greed composite index (0-100) with
    regime-aware thresholds, temporal smoothing, extreme detection,
    and multi-asset support.

    Components:
    - Market volatility (VIX proxy / implied vol)
    - Social media sentiment aggregate
    - Trading volume momentum
    - Market dominance shifts (BTC dominance for crypto)
    - Price momentum (new)
    - Funding rates / put-call ratio (new)

    Improvements over baseline:
    - **Regime-aware thresholds**: What counts as "extreme" depends on
      the current market regime (bull/bear/volatile).
    - **Temporal smoothing**: Exponential moving average prevents noisy
      single-reading whipsaws from generating false signals.
    - **Extreme reading detection**: Identifies persistent extreme
      readings (not just single spikes) which are historically the
      strongest contrarian signals.
    - **Multi-asset support**: Compute F&G per asset or per sector,
      not just a single global reading.
    - **Contrarian signal generation**: Automatically flags when readings
      reach historically profitable contrarian entry points.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        smoothing_alpha: float = 0.3,
        market_regime: str = "neutral",
        history_window: int = 50,
    ):
        self._weights = weights or DEFAULT_WEIGHTS.copy()
        total = sum(self._weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "Fear/Greed weights sum to %.2f, normalizing to 1.0", total
            )
            self._weights = {k: v / total for k, v in self._weights.items()}

        self._smoothing_alpha = smoothing_alpha  # EMA alpha (0-1)
        self._market_regime = market_regime
        self._history_window = history_window

        # Per-asset EMA state and history
        self._ema_state: Dict[str, float] = {}          # asset -> smoothed value
        self._history: Dict[str, deque] = {}             # asset -> deque of (ts, value)

        # Global (cross-asset) history
        self._global_history: deque = deque(maxlen=history_window)

    def set_market_regime(self, regime: str) -> None:
        """Update the market regime for threshold adjustment.

        Valid regimes: 'bull', 'bear', 'neutral', 'volatile'.
        """
        if regime not in _REGIME_ADJUSTMENTS:
            logger.warning("Unknown regime '%s', defaulting to 'neutral'", regime)
            regime = "neutral"
        self._market_regime = regime

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        volatility_score: Optional[float] = None,
        social_sentiment_score: Optional[float] = None,
        volume_momentum_score: Optional[float] = None,
        market_dominance_score: Optional[float] = None,
        price_momentum_score: Optional[float] = None,
        funding_rate_score: Optional[float] = None,
        asset: str = "_global",
    ) -> FearGreedIndex:
        """Compute the composite Fear & Greed Index with temporal smoothing.

        All input scores should be in [0, 100] where:
        - 0 = extreme fear signal
        - 100 = extreme greed signal

        Missing components are excluded and weights renormalized.

        Args:
            asset: Asset or sector key for per-asset tracking.
        """
        components: Dict[str, float] = {}
        active_weights: Dict[str, float] = {}

        inputs = {
            "volatility": volatility_score,
            "social_sentiment": social_sentiment_score,
            "volume_momentum": volume_momentum_score,
            "market_dominance": market_dominance_score,
            "price_momentum": price_momentum_score,
            "funding_rates": funding_rate_score,
        }

        for key, val in inputs.items():
            if val is not None and key in self._weights:
                components[key] = max(0.0, min(100.0, val))
                active_weights[key] = self._weights[key]

        if not components:
            return FearGreedIndex(value=50, label="Neutral", components={})

        # Renormalize active weights
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Weighted sum
        raw_value = sum(
            components[k] * normalized_weights[k] for k in components
        )

        # Temporal smoothing (EMA)
        smoothed = self._apply_ema(asset, raw_value)

        value = int(round(max(0, min(100, smoothed))))
        label = _label_for_value(value)

        # Record to history
        self._record_history(asset, value)

        # Extreme detection
        extreme_info = self._detect_extreme(asset, value)

        # Contrarian signal
        contrarian = self._generate_contrarian_signal(value, extreme_info)

        return FearGreedIndex(
            value=value,
            label=label,
            components={
                **components,
                "_raw_value": raw_value,
                "_smoothed_value": smoothed,
                "_regime": self._market_regime,
                "_extreme_info": extreme_info,
                "_contrarian_signal": contrarian,
            },
        )

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def _apply_ema(self, asset: str, raw_value: float) -> float:
        """Apply exponential moving average for temporal smoothing.

        Prevents noisy single readings from causing whipsaw signals.
        """
        prev = self._ema_state.get(asset)
        if prev is None:
            self._ema_state[asset] = raw_value
            return raw_value

        smoothed = self._smoothing_alpha * raw_value + (1 - self._smoothing_alpha) * prev
        self._ema_state[asset] = smoothed
        return smoothed

    def _record_history(self, asset: str, value: int) -> None:
        """Record a value to the per-asset history buffer."""
        if asset not in self._history:
            self._history[asset] = deque(maxlen=self._history_window)
        now = time.time()
        self._history[asset].append((now, value))
        self._global_history.append((now, value))

    # ------------------------------------------------------------------
    # Extreme reading detection
    # ------------------------------------------------------------------

    def _detect_extreme(self, asset: str, current_value: int) -> Dict:
        """Detect persistent extreme readings that are historically
        the strongest contrarian signals.

        A single extreme spike is often noise.  But when multiple recent
        readings cluster at extremes, it indicates genuine market
        dislocation -- and these are the points where contrarian trades
        have the highest expected value.
        """
        regime_params = _REGIME_ADJUSTMENTS.get(self._market_regime, _REGIME_ADJUSTMENTS["neutral"])
        fear_threshold = regime_params["extreme_fear_threshold"]
        greed_threshold = regime_params["extreme_greed_threshold"]

        history = self._history.get(asset)
        if not history or len(history) < _EXTREME_PERSISTENCE_WINDOW:
            return {
                "is_extreme_fear": current_value <= fear_threshold,
                "is_extreme_greed": current_value >= greed_threshold,
                "is_persistent": False,
                "persistence_count": 0,
                "regime_fear_threshold": fear_threshold,
                "regime_greed_threshold": greed_threshold,
            }

        recent = list(history)[-_EXTREME_PERSISTENCE_WINDOW:]
        fear_count = sum(1 for _, v in recent if v <= fear_threshold)
        greed_count = sum(1 for _, v in recent if v >= greed_threshold)

        is_persistent_fear = fear_count >= _EXTREME_PERSISTENCE_THRESHOLD
        is_persistent_greed = greed_count >= _EXTREME_PERSISTENCE_THRESHOLD

        return {
            "is_extreme_fear": current_value <= fear_threshold,
            "is_extreme_greed": current_value >= greed_threshold,
            "is_persistent": is_persistent_fear or is_persistent_greed,
            "is_persistent_fear": is_persistent_fear,
            "is_persistent_greed": is_persistent_greed,
            "fear_persistence_count": fear_count,
            "greed_persistence_count": greed_count,
            "regime_fear_threshold": fear_threshold,
            "regime_greed_threshold": greed_threshold,
        }

    # ------------------------------------------------------------------
    # Contrarian signal generation
    # ------------------------------------------------------------------

    def _generate_contrarian_signal(
        self, value: int, extreme_info: Dict
    ) -> Dict:
        """Generate contrarian trading signals at extreme readings.

        Historically, extreme fear is a buying opportunity and extreme
        greed is a selling opportunity.  The strength of the contrarian
        signal depends on:
        - How extreme the reading is
        - Whether the extreme is persistent (not just a single spike)
        - The current market regime
        """
        regime_params = _REGIME_ADJUSTMENTS.get(self._market_regime, _REGIME_ADJUSTMENTS["neutral"])

        signal = {
            "has_signal": False,
            "direction": "none",
            "strength": 0.0,
            "reasoning": "",
        }

        if extreme_info.get("is_extreme_fear"):
            # Contrarian BUY signal
            base_strength = (extreme_info["regime_fear_threshold"] - value) / extreme_info["regime_fear_threshold"]
            persistence_mult = 1.5 if extreme_info.get("is_persistent_fear") else 1.0
            regime_mult = regime_params["fear_contrarian_strength"]

            strength = max(0.0, min(1.0, base_strength * persistence_mult * regime_mult))
            if strength > 0.1:
                signal = {
                    "has_signal": True,
                    "direction": "BUY",
                    "strength": strength,
                    "reasoning": (
                        f"Extreme fear reading ({value}) below regime threshold "
                        f"({extreme_info['regime_fear_threshold']}). "
                        f"{'Persistent extreme. ' if extreme_info.get('is_persistent_fear') else ''}"
                        f"Regime: {self._market_regime}."
                    ),
                }

        elif extreme_info.get("is_extreme_greed"):
            # Contrarian SELL signal
            greed_threshold = extreme_info["regime_greed_threshold"]
            base_strength = (value - greed_threshold) / (100 - greed_threshold) if greed_threshold < 100 else 0
            persistence_mult = 1.5 if extreme_info.get("is_persistent_greed") else 1.0
            regime_mult = regime_params["greed_contrarian_strength"]

            strength = max(0.0, min(1.0, base_strength * persistence_mult * regime_mult))
            if strength > 0.1:
                signal = {
                    "has_signal": True,
                    "direction": "SELL",
                    "strength": strength,
                    "reasoning": (
                        f"Extreme greed reading ({value}) above regime threshold "
                        f"({greed_threshold}). "
                        f"{'Persistent extreme. ' if extreme_info.get('is_persistent_greed') else ''}"
                        f"Regime: {self._market_regime}."
                    ),
                }

        return signal

    # ------------------------------------------------------------------
    # Velocity: rate of change of F&G
    # ------------------------------------------------------------------

    def get_velocity(self, asset: str = "_global", lookback_readings: int = 10) -> float:
        """Compute the rate of change of the F&G index.

        A rapidly falling F&G (negative velocity) can indicate an emerging
        panic.  A rapidly rising F&G can indicate building euphoria.

        Returns:
            Velocity in index points per hour.  Positive = sentiment improving.
        """
        history = self._history.get(asset)
        if not history or len(history) < 3:
            return 0.0

        recent = list(history)[-lookback_readings:]
        if len(recent) < 3:
            return 0.0

        # Linear regression
        n = len(recent)
        sum_t = sum(p[0] for p in recent)
        sum_v = sum(p[1] for p in recent)
        sum_tv = sum(p[0] * p[1] for p in recent)
        sum_tt = sum(p[0] ** 2 for p in recent)
        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-12:
            return 0.0

        slope = (n * sum_tv - sum_t * sum_v) / denom
        return slope * 3600.0  # points per hour

    # ------------------------------------------------------------------
    # Component computation helpers
    # ------------------------------------------------------------------

    def compute_volatility_component(
        self, current_vol: float, avg_vol: float, regime_adjust: bool = True
    ) -> float:
        """Convert volatility reading to a 0-100 score.

        Higher volatility = more fear = lower score.

        Uses a sigmoid-like mapping instead of linear for more natural
        behavior at the extremes.

        Args:
            current_vol: Current volatility (VIX value or realized vol %).
            avg_vol: Historical average volatility for reference.
            regime_adjust: If True, adjust thresholds for current regime.
        """
        if avg_vol <= 0:
            return 50.0

        ratio = current_vol / avg_vol

        # Sigmoid mapping: centered at ratio=1.0 (neutral)
        # ratio << 1.0 -> score near 100 (low vol = greed)
        # ratio >> 1.0 -> score near 0 (high vol = fear)
        k = 3.0  # steepness
        score = 100.0 / (1.0 + math.exp(k * (ratio - 1.0)))

        if regime_adjust and self._market_regime == "volatile":
            # In volatile regimes, normalize for higher baseline vol
            score = 50.0 + (score - 50.0) * 0.7  # compress toward neutral

        return max(0.0, min(100.0, score))

    def compute_social_component(
        self, sentiment_scores: List[SentimentScore],
    ) -> float:
        """Convert social sentiment scores to a 0-100 component.

        Uses confidence-weighted averaging and filters out low-confidence
        noise scores.
        """
        if not sentiment_scores:
            return 50.0

        # Filter out very low confidence scores (likely noise)
        valid = [s for s in sentiment_scores if s.confidence > 0.1]
        if not valid:
            return 50.0

        total_weight = sum(s.confidence for s in valid)
        if total_weight == 0:
            avg = sum(s.score for s in valid) / len(valid)
        else:
            avg = sum(s.score * s.confidence for s in valid) / total_weight

        return max(0.0, min(100.0, (avg + 1.0) * 50.0))

    def compute_volume_component(
        self,
        current_volume: float,
        avg_volume: float,
        price_direction: Optional[float] = None,
    ) -> float:
        """Convert volume momentum to a 0-100 score.

        Volume alone is ambiguous: high volume in an uptrend = greed,
        high volume in a downtrend = fear.  When price_direction is
        provided, it disambiguates the signal.

        Args:
            current_volume: Current trading volume.
            avg_volume: Average trading volume over reference period.
            price_direction: Recent price change as fraction (-0.1 = -10%).
                            If None, volume is treated as a pure greed indicator.
        """
        if avg_volume <= 0:
            return 50.0

        ratio = current_volume / avg_volume

        if price_direction is not None:
            # Directional volume interpretation:
            # high volume + price up = greed, high volume + price down = fear
            if price_direction > 0:
                # Upside: volume amplifies greed
                score = 50.0 + 25.0 * (ratio - 1.0)
            else:
                # Downside: volume amplifies fear
                score = 50.0 - 25.0 * (ratio - 1.0)
        else:
            # Undirected: use sigmoid mapping
            score = 100.0 / (1.0 + math.exp(-2.0 * (ratio - 1.0)))

        return max(0.0, min(100.0, score))

    def compute_dominance_component(
        self, btc_dominance: float, btc_dominance_avg: float
    ) -> float:
        """Convert BTC dominance shift to a 0-100 score.

        Rising BTC dominance = risk-off (fear) as money flows to BTC.
        Falling BTC dominance = risk-on (greed) as money flows to alts.
        """
        if btc_dominance_avg <= 0:
            return 50.0

        shift = btc_dominance - btc_dominance_avg
        # Sigmoid mapping: shift of +10% -> ~20 (fear), shift of -10% -> ~80 (greed)
        score = 100.0 / (1.0 + math.exp(0.3 * shift))
        return max(0.0, min(100.0, score))

    def compute_price_momentum_component(
        self,
        returns_1d: float,
        returns_7d: float,
        returns_30d: float,
    ) -> float:
        """Convert multi-timeframe price returns to a 0-100 momentum score.

        Combines short, medium, and long-term momentum with increasing
        weight for longer-term trends (which are more persistent signals).

        Args:
            returns_1d: 1-day return as fraction (0.05 = +5%).
            returns_7d: 7-day return.
            returns_30d: 30-day return.
        """
        # Weight longer timeframes more
        weighted_return = 0.2 * returns_1d + 0.3 * returns_7d + 0.5 * returns_30d

        # Map to 0-100: -20% -> 0, 0% -> 50, +20% -> 100
        score = 50.0 + weighted_return * 250.0
        return max(0.0, min(100.0, score))

    def compute_funding_rate_component(
        self, funding_rate: float, avg_funding: float = 0.01
    ) -> float:
        """Convert perpetual futures funding rate to a 0-100 score.

        High positive funding = overleveraged longs = greed.
        High negative funding = overleveraged shorts = fear.

        Args:
            funding_rate: Current 8h funding rate (e.g., 0.01 = 1%).
            avg_funding: Historical average funding rate.
        """
        deviation = funding_rate - avg_funding
        # Map: deviation of -0.05 -> fear (0), deviation of +0.05 -> greed (100)
        score = 50.0 + deviation * 1000.0
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Multi-asset Fear & Greed
    # ------------------------------------------------------------------

    def compute_sector_fear_greed(
        self, asset_readings: Dict[str, int]
    ) -> Dict[str, Any]:
        """Compute aggregate and per-sector F&G from individual asset readings.

        Args:
            asset_readings: Dict mapping asset ticker to its individual F&G value.

        Returns:
            Dict with 'aggregate', 'per_asset', and 'divergence' data.
        """
        if not asset_readings:
            return {"aggregate": 50, "per_asset": {}, "divergence": 0.0}

        values = list(asset_readings.values())
        aggregate = int(round(sum(values) / len(values)))

        # Divergence: standard deviation of readings across assets
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        divergence = variance ** 0.5

        # High divergence = mixed signals, reduce confidence
        return {
            "aggregate": aggregate,
            "aggregate_label": _label_for_value(aggregate),
            "per_asset": {
                asset: {"value": v, "label": _label_for_value(v)}
                for asset, v in asset_readings.items()
            },
            "divergence": divergence,
            "consensus_strength": max(0.0, 1.0 - divergence / 50.0),
        }

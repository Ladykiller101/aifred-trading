"""Volatility regime detection and adaptive risk adjustments."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.types import VolatilityRegime

logger = logging.getLogger(__name__)


def atr_percentile(atr_values: List[float], lookback: int = 252) -> float:
    """Calculate current ATR as a percentile of its historical distribution.

    Args:
        atr_values: List of ATR values (most recent last).
        lookback: Number of periods for the historical window.

    Returns:
        Percentile rank 0-100.
    """
    if len(atr_values) < 2:
        return 50.0
    window = atr_values[-lookback:]
    current = window[-1]
    rank = np.sum(np.array(window[:-1]) <= current) / (len(window) - 1)
    return float(rank * 100.0)


def detect_regime_from_vix(
    vix: float,
    config: Optional[Dict[str, Any]] = None,
) -> VolatilityRegime:
    """Detect volatility regime from VIX level (for stocks).

    Args:
        vix: Current VIX value.
        config: Risk config dict.

    Returns:
        VolatilityRegime enum.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    vol_cfg = risk_cfg.get("volatility_regimes", {})
    high_threshold = vol_cfg.get("high_vix_threshold", 30)
    extreme_threshold = 45  # VIX above 45 is historically extreme

    if vix >= extreme_threshold:
        return VolatilityRegime.EXTREME
    elif vix >= high_threshold:
        return VolatilityRegime.HIGH
    elif vix >= 20:
        return VolatilityRegime.NORMAL
    else:
        return VolatilityRegime.LOW


def detect_regime_from_atr(
    atr_values: List[float],
    lookback: int = 252,
) -> VolatilityRegime:
    """Detect volatility regime from ATR percentile (for crypto/forex).

    Args:
        atr_values: Historical ATR values (most recent last).
        lookback: Lookback period for percentile calculation.

    Returns:
        VolatilityRegime enum.
    """
    pctile = atr_percentile(atr_values, lookback)

    if pctile >= 95:
        return VolatilityRegime.EXTREME
    elif pctile >= 80:
        return VolatilityRegime.HIGH
    elif pctile >= 30:
        return VolatilityRegime.NORMAL
    else:
        return VolatilityRegime.LOW


def detect_regime(
    vix: Optional[float] = None,
    atr_values: Optional[List[float]] = None,
    fear_greed_index: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> VolatilityRegime:
    """Detect the overall volatility regime using available data.

    Priority: VIX (stocks) > ATR percentile (crypto/forex) > fear/greed.

    Args:
        vix: Current VIX value (for stocks).
        atr_values: Historical ATR values (for crypto/forex).
        fear_greed_index: Fear & Greed index 0-100.
        config: Risk config dict.

    Returns:
        VolatilityRegime enum.
    """
    if config is None:
        config = {}
    risk_cfg = config.get("risk", config)
    vol_cfg = risk_cfg.get("volatility_regimes", {})
    extreme_fear = vol_cfg.get("extreme_fear_threshold", 20)

    # Check fear/greed for extreme conditions
    if fear_greed_index is not None and fear_greed_index <= extreme_fear:
        logger.warning("Extreme fear detected: F&G index = %d", fear_greed_index)
        return VolatilityRegime.EXTREME

    if vix is not None:
        return detect_regime_from_vix(vix, config)

    if atr_values is not None and len(atr_values) > 1:
        return detect_regime_from_atr(atr_values)

    return VolatilityRegime.NORMAL


def calculate_regime_score(
    vix: Optional[float] = None,
    atr_values: Optional[List[float]] = None,
    fear_greed_index: Optional[int] = None,
) -> float:
    """Calculate a composite regime score 0-100.

    0 = extremely calm markets, 100 = extreme volatility/fear.
    Combines available inputs into a single score.

    Args:
        vix: Current VIX value.
        atr_values: Historical ATR values.
        fear_greed_index: Fear & Greed index 0-100.

    Returns:
        Composite regime score 0-100.
    """
    components = []

    if vix is not None:
        # Map VIX: 10=0, 20=33, 30=66, 45+=100
        vix_score = min(100.0, max(0.0, (vix - 10.0) / 35.0 * 100.0))
        components.append(vix_score)

    if atr_values is not None and len(atr_values) > 1:
        pctile = atr_percentile(atr_values)
        components.append(pctile)

    if fear_greed_index is not None:
        # Invert: F&G 0 (extreme fear) = 100 regime score, F&G 100 = 0
        fg_score = 100.0 - fear_greed_index
        components.append(fg_score)

    if not components:
        return 50.0  # Default neutral

    score = sum(components) / len(components)
    return round(min(100.0, max(0.0, score)), 1)


def get_regime_adjustments(regime: VolatilityRegime) -> Dict[str, Any]:
    """Get risk parameter adjustments for a given volatility regime.

    LOW vol: tighter stops (1.5x ATR), slightly larger size, more positions.
    NORMAL: standard parameters.
    HIGH vol: wider stops (2.5x ATR), 50% size, A+ signals only, fewer positions.
    EXTREME: no new trades.

    Args:
        regime: Current volatility regime.

    Returns:
        Dict of adjustment parameters.
    """
    adjustments = {
        VolatilityRegime.LOW: {
            "position_size_multiplier": 1.05,   # Slightly larger in calm markets
            "stop_multiplier": 0.75,             # Tighter stops (1.5x ATR if base is 2)
            "max_positions": 12,
            "action": "normal",
            "require_a_plus_only": False,
        },
        VolatilityRegime.NORMAL: {
            "position_size_multiplier": 1.0,
            "stop_multiplier": 1.0,              # Standard 2x ATR
            "max_positions": 10,
            "action": "normal",
            "require_a_plus_only": False,
        },
        VolatilityRegime.HIGH: {
            "position_size_multiplier": 0.5,     # 50% size reduction
            "stop_multiplier": 1.25,             # Wider stops (2.5x ATR)
            "max_positions": 5,
            "action": "reduce_exposure",
            "require_a_plus_only": True,         # Only A+ signals
        },
        VolatilityRegime.EXTREME: {
            "position_size_multiplier": 0.0,
            "stop_multiplier": 1.5,              # Widest stops if any remain
            "max_positions": 0,
            "action": "close_all_or_hedge",
            "require_a_plus_only": True,
        },
    }
    result = adjustments.get(regime, adjustments[VolatilityRegime.NORMAL])
    logger.info("Volatility regime %s: adjustments=%s", regime.value, result)
    return result


class RegimeTransitionDetector:
    """Detects volatility regime transitions and alerts on dangerous shifts."""

    def __init__(self):
        self._regime_history: List[Dict[str, Any]] = []
        self._score_history: List[float] = []
        self._max_history = 100

    def record(self, regime: VolatilityRegime, score: float) -> Optional[Dict[str, Any]]:
        """Record a regime observation and detect transitions.

        Args:
            regime: Current detected regime.
            score: Current regime score 0-100.

        Returns:
            Transition alert dict if a dangerous transition detected, else None.
        """
        now_str = datetime.utcnow().isoformat()
        self._score_history.append(score)
        if len(self._score_history) > self._max_history:
            self._score_history = self._score_history[-self._max_history:]

        entry = {"regime": regime.value, "score": score, "timestamp": now_str}
        self._regime_history.append(entry)
        if len(self._regime_history) > self._max_history:
            self._regime_history = self._regime_history[-self._max_history:]

        # Check for transition
        if len(self._regime_history) < 2:
            return None

        prev = self._regime_history[-2]
        curr = self._regime_history[-1]

        if prev["regime"] == curr["regime"]:
            return None

        # Determine if dangerous (low -> high is the most dangerous)
        danger_transitions = {
            ("low", "high"): "HIGH",
            ("low", "extreme"): "CRITICAL",
            ("normal", "high"): "MODERATE",
            ("normal", "extreme"): "HIGH",
            ("high", "extreme"): "MODERATE",
        }
        key = (prev["regime"], curr["regime"])
        danger_level = danger_transitions.get(key, "LOW")

        alert = {
            "type": "regime_transition",
            "from_regime": prev["regime"],
            "to_regime": curr["regime"],
            "from_score": prev["score"],
            "to_score": curr["score"],
            "danger_level": danger_level,
            "timestamp": now_str,
            "message": (
                f"Volatility regime shift: {prev['regime']} -> {curr['regime']} "
                f"(score {prev['score']:.0f} -> {curr['score']:.0f}). "
                f"Danger level: {danger_level}"
            ),
        }

        if danger_level in ("HIGH", "CRITICAL"):
            logger.warning("DANGEROUS REGIME TRANSITION: %s", alert["message"])
        else:
            logger.info("Regime transition: %s", alert["message"])

        return alert

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._regime_history)

    @property
    def score_history(self) -> List[float]:
        return list(self._score_history)

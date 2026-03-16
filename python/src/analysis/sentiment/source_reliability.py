"""Track and weight sources by historical reliability with dynamic scoring,
source type hierarchy, freshness decay, and per-asset accuracy tracking."""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default decay half-life in days
DEFAULT_HALF_LIFE_DAYS = 30.0

# ---------------------------------------------------------------------------
# Source type hierarchy
# ---------------------------------------------------------------------------
# Institutional sources (SEC filings, Bloomberg terminals) are historically
# more reliable than retail social media chatter.  These priors are used
# as Bayesian priors when a source has limited track record.
_SOURCE_TYPE_PRIORS: Dict[str, float] = {
    # Tier 1: Institutional / official
    "sec_filing": 0.85,
    "central_bank": 0.85,
    "exchange_official": 0.80,
    "institutional_research": 0.78,

    # Tier 2: Major financial media
    "bloomberg": 0.75,
    "reuters": 0.75,
    "wsj": 0.72,
    "ft": 0.72,

    # Tier 3: Financial news
    "cnbc": 0.65,
    "coindesk": 0.63,
    "cointelegraph": 0.60,
    "news_major": 0.65,
    "news_minor": 0.55,

    # Tier 4: Analyst / research
    "analyst": 0.60,
    "research_report": 0.65,

    # Tier 5: Social media (verified / high-profile)
    "social_verified": 0.45,
    "crypto_influencer": 0.35,

    # Tier 6: General social / anonymous
    "social_general": 0.30,
    "reddit": 0.30,
    "forum": 0.25,
    "telegram": 0.20,
    "discord": 0.20,
    "anonymous": 0.20,

    # Default
    "unknown": 0.40,
    "finbert": 0.55,  # model-based, no source quality dimension
    "llm": 0.65,      # LLM analysis tends to be better calibrated
    "social": 0.35,    # aggregate social
    "event": 0.60,     # event-based signals
}

# Weight mapping function parameters
_WEIGHT_FLOOR = 0.05    # minimum weight (never fully ignore a source)
_WEIGHT_CEILING = 2.5   # maximum weight (cap to prevent over-reliance)

# Freshness decay: predictions older than this are heavily discounted
_FRESHNESS_HALF_LIFE_HOURS = 24.0


@dataclass
class SourceRecord:
    """A single prediction record for a source."""
    source: str
    source_type: str  # e.g., "bloomberg", "reddit", "finbert"
    asset: str
    predicted_direction: str  # "positive", "negative", "neutral"
    predicted_score: float = 0.0  # raw [-1, 1] score
    predicted_confidence: float = 0.5
    actual_outcome: Optional[str] = None  # filled when outcome is known
    actual_return: Optional[float] = None  # actual price return
    timestamp: float = field(default_factory=time.time)
    was_actionable: bool = False  # did the bot act on this signal?
    was_profitable: Optional[bool] = None  # was the resulting trade profitable?


@dataclass
class SourceStats:
    """Aggregated reliability stats for a source."""
    source: str
    source_type: str = "unknown"
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5
    weighted_accuracy: float = 0.5   # time-decay weighted
    profitability_rate: float = 0.5  # fraction of profitable signals
    reliability_score: float = 0.5   # final composite score
    calibration_error: float = 0.0   # avg |confidence - accuracy|
    direction_bias: float = 0.0      # positive = bullish bias


class SourceReliabilityTracker:
    """Tracks which sources produce actionable, accurate sentiment signals
    with dynamic scoring based on historical accuracy, source type
    hierarchy, freshness decay, and calibration tracking.

    Improvements over baseline:
    - **Source type hierarchy**: Institutional sources start with higher
      priors and receive higher weight even with limited track records.
    - **Freshness decay**: Recent prediction accuracy matters more than
      old performance.  A source that was accurate 6 months ago but has
      been wrong recently is downweighted.
    - **Per-asset tracking**: A source might be reliable for BTC analysis
      but unreliable for altcoins.  Per-asset stats enable this.
    - **Calibration tracking**: Sources whose confidence scores match
      their actual accuracy are more useful.  Overconfident sources
      are penalized.
    - **Profitability tracking**: Beyond directional accuracy, tracks
      whether acting on a source's signals was actually profitable.
    - **Direction bias detection**: Identifies sources with persistent
      bullish or bearish bias so downstream aggregation can correct.
    - **Dynamic weight mapping**: Non-linear mapping from reliability
      to weight, with floor/ceiling to prevent extreme allocations.
    """

    def __init__(
        self,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        freshness_half_life_hours: float = _FRESHNESS_HALF_LIFE_HOURS,
    ):
        self._half_life_seconds = half_life_days * 86400.0
        self._freshness_half_life_seconds = freshness_half_life_hours * 3600.0
        self._records: Dict[str, List[SourceRecord]] = defaultdict(list)

        # Per-asset records for granular tracking
        self._asset_records: Dict[Tuple[str, str], List[SourceRecord]] = defaultdict(list)

        # Prior weight: how many "virtual" observations the prior is worth.
        # Higher = more resistant to change from limited data.
        self._prior_weight = 15

        # Cache for computed stats
        self._stats_cache: Dict[str, Tuple[float, SourceStats]] = {}
        self._cache_ttl = 60.0  # recompute at most every 60 seconds

    # ------------------------------------------------------------------
    # Recording predictions and outcomes
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        source: str,
        asset: str,
        predicted_direction: str,
        predicted_score: float = 0.0,
        predicted_confidence: float = 0.5,
        source_type: str = "unknown",
        was_actionable: bool = False,
    ) -> None:
        """Record a new prediction from a source (outcome unknown yet)."""
        record = SourceRecord(
            source=source,
            source_type=source_type,
            asset=asset,
            predicted_direction=predicted_direction,
            predicted_score=predicted_score,
            predicted_confidence=predicted_confidence,
            was_actionable=was_actionable,
        )
        self._records[source].append(record)
        self._asset_records[(source, asset)].append(record)
        self._invalidate_cache(source)

    def record_outcome(
        self,
        source: str,
        asset: str,
        actual_outcome: str,
        actual_return: Optional[float] = None,
        lookback_seconds: float = 3600.0,
    ) -> None:
        """Update the most recent unresolved prediction for this source/asset.

        Args:
            source: Source identifier.
            asset: Asset ticker.
            actual_outcome: What actually happened ("positive", "negative", "neutral").
            actual_return: Actual price return (e.g., 0.05 = +5%).
            lookback_seconds: How far back to look for a matching prediction.
        """
        now = time.time()
        records = self._records.get(source, [])
        for record in reversed(records):
            if (
                record.asset == asset
                and record.actual_outcome is None
                and now - record.timestamp < lookback_seconds
            ):
                record.actual_outcome = actual_outcome
                record.actual_return = actual_return

                # Determine profitability
                if actual_return is not None:
                    if record.predicted_direction == "positive":
                        record.was_profitable = actual_return > 0
                    elif record.predicted_direction == "negative":
                        record.was_profitable = actual_return < 0
                    else:
                        record.was_profitable = abs(actual_return) < 0.01

                self._invalidate_cache(source)
                return

        logger.debug(
            "No unresolved prediction found for source=%s asset=%s", source, asset
        )

    # ------------------------------------------------------------------
    # Reliability and weight computation
    # ------------------------------------------------------------------

    def get_reliability(self, source: str, asset: Optional[str] = None) -> float:
        """Get the current reliability score for a source.

        If asset is specified, returns per-asset reliability.
        Otherwise returns global reliability.

        Returns a value in [0, 1] where 1.0 = perfectly reliable.
        """
        stats = self._compute_stats(source, asset=asset)
        return stats.reliability_score

    def get_weight(self, source: str, asset: Optional[str] = None) -> float:
        """Get a weight multiplier for this source's signals.

        Uses a non-linear mapping that:
        - Gives a meaningful floor (never fully ignores a source)
        - Has diminishing returns for very high reliability
        - Accounts for source type priors even with limited data
        """
        reliability = self.get_reliability(source, asset=asset)

        # Sigmoid-like mapping for smoother weight transitions
        # reliability 0.0 -> weight ~0.1, reliability 0.5 -> weight ~1.0, reliability 1.0 -> weight ~2.5
        weight = _WEIGHT_FLOOR + (_WEIGHT_CEILING - _WEIGHT_FLOOR) / (
            1.0 + math.exp(-6.0 * (reliability - 0.5))
        )

        return weight

    def get_all_stats(self) -> Dict[str, SourceStats]:
        """Get reliability stats for all tracked sources."""
        return {source: self._compute_stats(source) for source in self._records}

    def get_source_ranking(self) -> List[Tuple[str, float]]:
        """Get all sources ranked by reliability score (highest first)."""
        all_stats = self.get_all_stats()
        ranked = sorted(
            [(s, stats.reliability_score) for s, stats in all_stats.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------------
    # Stats computation with caching
    # ------------------------------------------------------------------

    def _compute_stats(
        self, source: str, asset: Optional[str] = None
    ) -> SourceStats:
        """Compute time-decay-weighted accuracy, calibration error,
        direction bias, and profitability for a source."""
        cache_key = f"{source}:{asset or '_global'}"
        cached = self._stats_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._cache_ttl:
            return cached[1]

        if asset:
            records = self._asset_records.get((source, asset), [])
        else:
            records = self._records.get(source, [])

        resolved = [r for r in records if r.actual_outcome is not None]

        # Determine source type from most recent record
        source_type = "unknown"
        if records:
            source_type = records[-1].source_type

        # Get prior from source type hierarchy
        prior_accuracy = _SOURCE_TYPE_PRIORS.get(source_type, 0.40)
        prior_accuracy = max(prior_accuracy, _SOURCE_TYPE_PRIORS.get(source, 0.40))

        if not resolved:
            stats = SourceStats(
                source=source,
                source_type=source_type,
                reliability_score=prior_accuracy,
            )
            self._stats_cache[cache_key] = (time.time(), stats)
            return stats

        now = time.time()
        total_weight = 0.0
        correct_weight = 0.0
        raw_correct = 0
        profitable_count = 0
        profitable_total = 0
        calibration_errors = []
        direction_sum = 0.0

        for record in resolved:
            age_seconds = now - record.timestamp

            # Exponential decay weight (long-term)
            decay = math.exp(
                -math.log(2) * age_seconds / self._half_life_seconds
            )

            # Freshness boost for very recent predictions
            freshness = math.exp(
                -math.log(2) * age_seconds / self._freshness_half_life_seconds
            )

            # Combined weight: product of decay and freshness bonus
            weight = decay * (1.0 + 0.5 * freshness)
            total_weight += weight

            is_correct = record.predicted_direction == record.actual_outcome
            if is_correct:
                correct_weight += weight
                raw_correct += 1

            # Profitability tracking
            if record.was_profitable is not None:
                profitable_total += 1
                if record.was_profitable:
                    profitable_count += 1

            # Calibration error: |predicted confidence - actual correctness|
            actual_correctness = 1.0 if is_correct else 0.0
            calibration_errors.append(
                abs(record.predicted_confidence - actual_correctness) * weight
            )

            # Direction bias: sum of predicted scores
            direction_sum += record.predicted_score * weight

        # Bayesian smoothing with source-type-informed prior
        prior_correct = self._prior_weight * prior_accuracy
        prior_total = self._prior_weight
        weighted_accuracy = (correct_weight + prior_correct) / (
            total_weight + prior_total
        )
        raw_accuracy = raw_correct / len(resolved) if resolved else prior_accuracy

        # Profitability rate
        profitability_rate = (
            profitable_count / profitable_total if profitable_total > 0 else 0.5
        )

        # Calibration error (lower = better calibrated)
        avg_calibration_error = (
            sum(calibration_errors) / total_weight if total_weight > 0 else 0.0
        )

        # Direction bias (positive = bullish bias, negative = bearish bias)
        direction_bias = direction_sum / total_weight if total_weight > 0 else 0.0

        # Composite reliability score
        # - 60% weighted accuracy
        # - 25% profitability
        # - 15% calibration quality (1 - error)
        calibration_quality = max(0.0, 1.0 - avg_calibration_error)
        reliability_score = (
            0.60 * weighted_accuracy
            + 0.25 * profitability_rate
            + 0.15 * calibration_quality
        )

        # Penalize high direction bias (biased sources are less useful)
        bias_penalty = abs(direction_bias) * 0.1
        reliability_score = max(0.0, min(1.0, reliability_score - bias_penalty))

        stats = SourceStats(
            source=source,
            source_type=source_type,
            total_predictions=len(resolved),
            correct_predictions=raw_correct,
            accuracy=raw_accuracy,
            weighted_accuracy=weighted_accuracy,
            profitability_rate=profitability_rate,
            reliability_score=reliability_score,
            calibration_error=avg_calibration_error,
            direction_bias=direction_bias,
        )

        self._stats_cache[cache_key] = (time.time(), stats)
        return stats

    def _invalidate_cache(self, source: str) -> None:
        """Invalidate cached stats for a source."""
        to_remove = [k for k in self._stats_cache if k.startswith(f"{source}:")]
        for k in to_remove:
            del self._stats_cache[k]

    # ------------------------------------------------------------------
    # Source comparison and diagnostics
    # ------------------------------------------------------------------

    def compare_sources(
        self, source_a: str, source_b: str, asset: Optional[str] = None
    ) -> Dict:
        """Compare two sources' reliability on the same asset.

        Useful for deciding which analysis method to trust when sources
        disagree.
        """
        stats_a = self._compute_stats(source_a, asset=asset)
        stats_b = self._compute_stats(source_b, asset=asset)

        return {
            source_a: {
                "reliability": stats_a.reliability_score,
                "accuracy": stats_a.accuracy,
                "profitability": stats_a.profitability_rate,
                "calibration_error": stats_a.calibration_error,
                "bias": stats_a.direction_bias,
                "n_predictions": stats_a.total_predictions,
            },
            source_b: {
                "reliability": stats_b.reliability_score,
                "accuracy": stats_b.accuracy,
                "profitability": stats_b.profitability_rate,
                "calibration_error": stats_b.calibration_error,
                "bias": stats_b.direction_bias,
                "n_predictions": stats_b.total_predictions,
            },
            "more_reliable": source_a if stats_a.reliability_score > stats_b.reliability_score else source_b,
            "reliability_delta": abs(stats_a.reliability_score - stats_b.reliability_score),
        }

    def get_bias_report(self) -> Dict[str, Dict[str, float]]:
        """Get direction bias report for all sources.

        Identifies sources that consistently lean bullish or bearish,
        which helps correct for systematic bias in aggregation.
        """
        report = {}
        for source in self._records:
            stats = self._compute_stats(source)
            if stats.total_predictions >= 5:  # need minimum data
                report[source] = {
                    "direction_bias": stats.direction_bias,
                    "bias_label": (
                        "bullish" if stats.direction_bias > 0.1
                        else "bearish" if stats.direction_bias < -0.1
                        else "neutral"
                    ),
                    "n_predictions": stats.total_predictions,
                }
        return report

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_old_records(self, max_age_days: float = 90.0) -> int:
        """Remove records older than max_age_days.

        Returns the number of records removed.
        """
        cutoff = time.time() - max_age_days * 86400.0
        removed = 0

        for source in list(self._records.keys()):
            before = len(self._records[source])
            self._records[source] = [
                r for r in self._records[source] if r.timestamp >= cutoff
            ]
            removed += before - len(self._records[source])
            if not self._records[source]:
                del self._records[source]

        for key in list(self._asset_records.keys()):
            before = len(self._asset_records[key])
            self._asset_records[key] = [
                r for r in self._asset_records[key] if r.timestamp >= cutoff
            ]
            removed += before - len(self._asset_records[key])
            if not self._asset_records[key]:
                del self._asset_records[key]

        # Clear cache after pruning
        self._stats_cache.clear()

        return removed

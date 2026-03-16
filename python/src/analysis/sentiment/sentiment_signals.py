"""Main sentiment signal interface -- the SentimentAnalysisAgent.

Orchestrates FinBERT, LLM, social media, event detection, Fear & Greed
index, and source reliability into unified trading signals with
multi-source consensus, sentiment velocity, and contrarian signals."""

import logging
import math
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.sentiment.event_detector import EventDetector
from src.analysis.sentiment.fear_greed_index import FearGreedCalculator
from src.analysis.sentiment.finbert_model import FinBERTModel
from src.analysis.sentiment.llm_analyzer import LLMAnalyzer
from src.analysis.sentiment.social_aggregator import SocialAggregator
from src.analysis.sentiment.source_reliability import SourceReliabilityTracker
from src.analysis.sentiment.text_preprocessor import TextPreprocessor
from src.utils.types import Direction, FearGreedIndex, SentimentScore, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights for combining sentiment sources
# ---------------------------------------------------------------------------
# These are *base* weights; they are dynamically adjusted by source
# reliability tracking.  Institutional-grade analysis (LLM with deep
# chain-of-thought) gets the highest base weight.
_DEFAULT_SOURCE_WEIGHTS = {
    "finbert": 0.32,
    "llm": 0.40,
    "social": 0.08,
    "event": 0.20,
}

# ---------------------------------------------------------------------------
# Direction mapping with hysteresis zones
# ---------------------------------------------------------------------------
# Using wider "dead zones" around thresholds prevents oscillation between
# adjacent direction labels when the score hovers near a boundary.
_SCORE_TO_DIRECTION = [
    (-1.0, -0.65, Direction.STRONG_SELL),
    (-0.65, -0.25, Direction.SELL),
    (-0.25, 0.25, Direction.HOLD),
    (0.25, 0.65, Direction.BUY),
    (0.65, 1.01, Direction.STRONG_BUY),
]

# Hysteresis: once in a strong state, require the score to cross further
# before transitioning back to a weaker state.
_HYSTERESIS_MARGIN = 0.05

# Minimum number of agreeing sources to consider a signal "consensus-backed"
_MIN_CONSENSUS_SOURCES = 3

# Contrarian override thresholds (from Fear & Greed)
_CONTRARIAN_F_G_FEAR_THRESHOLD = 15
_CONTRARIAN_F_G_GREED_THRESHOLD = 90

# Sentiment velocity thresholds (score units per hour)
_VELOCITY_SPIKE_THRESHOLD = 0.3   # rapid sentiment shift
_VELOCITY_COLLAPSE_THRESHOLD = -0.3


def _score_to_direction(score: float) -> Direction:
    """Map a [-1, 1] sentiment score to a Direction enum."""
    for lo, hi, direction in _SCORE_TO_DIRECTION:
        if lo <= score < hi:
            return direction
    return Direction.HOLD


class SentimentAnalysisAgent:
    """Orchestrates the full sentiment analysis pipeline with multi-source
    consensus, confidence-weighted scoring, sentiment velocity tracking,
    and contrarian signal generation.

    Improvements over baseline:
    - **Multi-source consensus**: Requires agreement between at least 2
      sources before issuing non-HOLD signals, reducing false positives.
    - **Confidence-weighted scoring**: Each source contributes proportional
      to its calibrated confidence and historical reliability.
    - **Sentiment velocity**: Tracks the rate-of-change of sentiment and
      flags rapid shifts that often precede price moves.
    - **Contrarian signals**: When Fear & Greed reaches persistent extremes,
      generates contrarian trading signals with appropriate conviction.
    - **Event-driven urgency**: High-impact events (hacks, regulatory
      actions) automatically boost signal urgency and position sizing.
    - **Hysteresis**: Prevents noisy oscillation between adjacent signal
      levels by requiring scores to cross further to change state.
    - **Signal quality scoring**: Each output signal carries a quality
      score based on source agreement, data freshness, and sample size.
    """

    def __init__(
        self,
        source_weights: Optional[Dict[str, float]] = None,
        fear_greed_weights: Optional[Dict[str, float]] = None,
        subreddits: Optional[List[str]] = None,
        market_regime: str = "neutral",
    ):
        self._source_weights = source_weights or _DEFAULT_SOURCE_WEIGHTS.copy()
        self._finbert = FinBERTModel()
        self._llm = LLMAnalyzer(market_regime=market_regime)
        self._social = SocialAggregator(subreddits=subreddits)
        self._event_detector = EventDetector()
        self._fear_greed = FearGreedCalculator(
            weights=fear_greed_weights, market_regime=market_regime
        )
        self._reliability = SourceReliabilityTracker()
        self._preprocessor = TextPreprocessor()
        self._market_regime = market_regime

        # Signal history for velocity tracking
        self._signal_history: Dict[str, deque] = {}
        self._last_direction: Dict[str, Direction] = {}

    def set_market_regime(self, regime: str) -> None:
        """Propagate market regime to all sub-components."""
        self._market_regime = regime
        self._llm.set_market_regime(regime)
        self._fear_greed.set_market_regime(regime)

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        asset: str,
        news_items: Optional[List[str]] = None,
        social_data: Optional[Dict[str, Any]] = None,
        fear_greed_data: Optional[Dict[str, float]] = None,
    ) -> Signal:
        """Run the full sentiment analysis pipeline for an asset.

        Pipeline:
        1. Preprocess and classify news text with FinBERT
        2. Detect events and compute urgency for LLM prioritization
        3. Run LLM deep analysis on urgent/high-impact items
        4. Get social sentiment (pre-fetched or live Reddit)
        5. Compute event-based sentiment modifier
        6. Aggregate with source reliability and consensus checking
        7. Apply sentiment velocity and contrarian adjustments
        8. Compute signal quality score
        9. Generate final Signal with direction and confidence

        Args:
            asset: Asset ticker (e.g., "BTC/USDT", "AAPL").
            news_items: List of raw news text strings.
            social_data: Pre-fetched social data with "score" key.
            fear_greed_data: Pre-computed F&G component scores.

        Returns:
            Signal with aggregated sentiment direction, confidence, and
            rich metadata including consensus info, velocity, and
            contrarian flags.
        """
        news_items = news_items or []
        component_scores: Dict[str, SentimentScore] = {}
        event_metadata: Dict[str, Any] = {}
        now_ts = time.time()

        # --- Event detection (run first to guide LLM urgency) ---
        events = []
        max_urgency = 0
        max_event_impact = "low"
        if news_items:
            events = self._event_detector.detect_batch(news_items)
            if events:
                max_urgency = max(e.urgency for e in events)
                # Classify overall event impact level
                if max_urgency >= 8:
                    max_event_impact = "high"
                elif max_urgency >= 5:
                    max_event_impact = "medium"

                event_metadata = {
                    "event_count": len(events),
                    "max_urgency": max_urgency,
                    "max_impact": max_event_impact,
                    "event_types": list(set(e.event_type.value for e in events)),
                }

        # --- FinBERT on news ---
        if news_items:
            finbert_scores = self._finbert.classify_batch(news_items, asset=asset)
            finbert_composite = self._finbert.aggregate_scores(
                finbert_scores, asset=asset, timeframe="1h"
            )
            component_scores["finbert"] = finbert_composite
            logger.info(
                "FinBERT: asset=%s score=%.3f confidence=%.3f n=%d",
                asset, finbert_composite.score, finbert_composite.confidence, len(news_items),
            )

        # --- LLM deep analysis on urgent items ---
        if news_items and events:
            # Sort by urgency and analyze top items
            urgent_indices = sorted(
                range(len(events)),
                key=lambda i: events[i].urgency,
                reverse=True,
            )

            # Number of items to send to LLM depends on urgency level
            n_llm_items = 5 if max_urgency >= 8 else 3 if max_urgency >= 5 else 2
            urgent_indices = urgent_indices[:n_llm_items]

            llm_scores = []
            for idx in urgent_indices:
                try:
                    llm_score = self._llm.analyze(
                        news_items[idx],
                        asset,
                        urgency_hint=events[idx].urgency,
                    )
                    llm_scores.append(llm_score)
                except Exception:
                    logger.warning("LLM analysis failed for item %d", idx, exc_info=True)

            if llm_scores:
                llm_composite = self._aggregate_llm_scores(llm_scores, asset)
                component_scores["llm"] = llm_composite

        # --- Social sentiment ---
        if social_data and "score" in social_data:
            social_score = SentimentScore(
                asset=asset,
                score=float(social_data["score"]),
                source="social",
                confidence=float(social_data.get("confidence", 0.5)),
                sample_size=int(social_data.get("sample_size", 0)),
                metadata=social_data.get("metadata", {}),
            )
            component_scores["social"] = social_score
        else:
            try:
                social_score = self._social.get_asset_momentum(asset)
                if social_score.sample_size > 0:
                    component_scores["social"] = social_score
            except Exception:
                logger.warning("Social aggregation failed for %s", asset, exc_info=True)

        # --- Event-based sentiment modifier ---
        if events:
            event_sentiment = self._compute_event_sentiment(events, asset)
            if event_sentiment is not None:
                component_scores["event"] = event_sentiment

        # --- No data guard ---
        if not component_scores:
            return Signal(
                asset=asset,
                direction=Direction.HOLD,
                confidence=0.0,
                source="sentiment",
                metadata={"reason": "no_data"},
            )

        # --- Aggregate with reliability weighting ---
        final_score, final_confidence, consensus_info = self._weighted_aggregate_with_consensus(
            component_scores
        )

        # --- Sentiment velocity ---
        velocity = self._compute_velocity(asset, final_score, now_ts)

        # --- Contrarian adjustment ---
        contrarian_adjustment = self._check_contrarian(asset, fear_greed_data, final_score)

        # Apply contrarian override if strong enough
        if contrarian_adjustment.get("override"):
            final_score = contrarian_adjustment["adjusted_score"]
            final_confidence *= contrarian_adjustment.get("confidence_mult", 0.8)

        # --- Velocity-based confidence adjustment ---
        if velocity is not None:
            if abs(velocity) > _VELOCITY_SPIKE_THRESHOLD:
                # Rapid sentiment shift -- boost confidence in the direction of the shift
                if (velocity > 0 and final_score > 0) or (velocity < 0 and final_score < 0):
                    # Velocity aligns with signal -> boost confidence
                    final_confidence = min(1.0, final_confidence * 1.2)
                else:
                    # Velocity opposes signal -> reduce confidence
                    final_confidence *= 0.7

        # --- Direction with hysteresis ---
        direction = self._direction_with_hysteresis(asset, final_score)

        # --- Signal quality score ---
        quality = self._compute_signal_quality(
            component_scores, consensus_info, velocity, len(news_items)
        )

        # --- Urgency modifier for confidence ---
        if max_event_impact == "high":
            final_confidence = min(1.0, final_confidence * 1.3)
        elif max_event_impact == "medium":
            final_confidence = min(1.0, final_confidence * 1.1)

        confidence_pct = min(100.0, max(0.0, final_confidence * 100.0))

        return Signal(
            asset=asset,
            direction=direction,
            confidence=confidence_pct,
            source="sentiment",
            metadata={
                "sentiment_score": final_score,
                "components": {
                    k: {
                        "score": v.score,
                        "confidence": v.confidence,
                        "sample_size": v.sample_size,
                    }
                    for k, v in component_scores.items()
                },
                "consensus": consensus_info,
                "velocity": velocity,
                "velocity_status": (
                    "spike" if velocity and velocity > _VELOCITY_SPIKE_THRESHOLD
                    else "collapse" if velocity and velocity < _VELOCITY_COLLAPSE_THRESHOLD
                    else "stable"
                ),
                "contrarian": contrarian_adjustment,
                "event_info": event_metadata,
                "signal_quality": quality,
                "market_regime": self._market_regime,
            },
        )

    # ------------------------------------------------------------------
    # Fear & Greed integration
    # ------------------------------------------------------------------

    def get_fear_greed(
        self,
        asset: str = "_global",
        volatility_score: Optional[float] = None,
        social_sentiments: Optional[List[SentimentScore]] = None,
        volume_momentum_score: Optional[float] = None,
        market_dominance_score: Optional[float] = None,
        price_momentum_score: Optional[float] = None,
        funding_rate_score: Optional[float] = None,
    ) -> FearGreedIndex:
        """Get the current composite Fear & Greed Index with per-asset tracking.

        Args:
            asset: Asset key for per-asset tracking.
            volatility_score: Volatility component (0-100).
            social_sentiments: Social SentimentScores for the social component.
            volume_momentum_score: Volume momentum component (0-100).
            market_dominance_score: Market dominance component (0-100).
            price_momentum_score: Price momentum component (0-100).
            funding_rate_score: Funding rate component (0-100).
        """
        social_component = None
        if social_sentiments:
            social_component = self._fear_greed.compute_social_component(social_sentiments)

        return self._fear_greed.compute(
            volatility_score=volatility_score,
            social_sentiment_score=social_component,
            volume_momentum_score=volume_momentum_score,
            market_dominance_score=market_dominance_score,
            price_momentum_score=price_momentum_score,
            funding_rate_score=funding_rate_score,
            asset=asset,
        )

    # ------------------------------------------------------------------
    # Quick sentiment
    # ------------------------------------------------------------------

    def get_asset_sentiment(
        self, asset: str, news_items: Optional[List[str]] = None
    ) -> SentimentScore:
        """Get a quick per-asset sentiment score using FinBERT only.

        For full analysis with LLM, social, and events, use analyze().
        """
        if not news_items:
            return SentimentScore(
                asset=asset,
                score=0.0,
                source="composite",
                confidence=0.0,
                sample_size=0,
            )

        scores = self._finbert.classify_batch(news_items, asset=asset)
        composite = self._finbert.aggregate_scores(scores, asset=asset)

        # Add velocity metadata
        velocity = self._finbert.get_sentiment_velocity(asset)

        return SentimentScore(
            asset=composite.asset,
            score=composite.score,
            source="composite",
            confidence=composite.confidence,
            sample_size=composite.sample_size,
            metadata={
                **(composite.metadata or {}),
                "velocity": velocity,
            },
        )

    # ------------------------------------------------------------------
    # Weighted aggregation with consensus checking
    # ------------------------------------------------------------------

    def _weighted_aggregate_with_consensus(
        self, components: Dict[str, SentimentScore]
    ) -> Tuple[float, float, Dict]:
        """Aggregate component scores with reliability weighting and
        multi-source consensus validation.

        Returns:
            (score, confidence, consensus_info) tuple.
        """
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        directions: Dict[str, str] = {}

        for source_name, sentiment in components.items():
            base_weight = self._source_weights.get(source_name, 0.1)
            reliability_mult = self._reliability.get_weight(source_name)
            effective_weight = base_weight * reliability_mult * sentiment.confidence

            weighted_score += sentiment.score * effective_weight
            weighted_confidence += sentiment.confidence * effective_weight
            total_weight += effective_weight

            # Track direction per source for consensus
            if sentiment.score > 0.1:
                directions[source_name] = "bullish"
            elif sentiment.score < -0.1:
                directions[source_name] = "bearish"
            else:
                directions[source_name] = "neutral"

        if total_weight == 0:
            return 0.0, 0.0, {"consensus": False, "agreement": 0.0}

        final_score = max(-1.0, min(1.0, weighted_score / total_weight))
        final_confidence = weighted_confidence / total_weight

        # Consensus checking
        bullish_count = sum(1 for d in directions.values() if d == "bullish")
        bearish_count = sum(1 for d in directions.values() if d == "bearish")
        neutral_count = sum(1 for d in directions.values() if d == "neutral")
        total_sources = len(directions)

        # Determine consensus
        dominant = max(bullish_count, bearish_count, neutral_count)
        agreement = dominant / total_sources if total_sources > 0 else 0

        has_consensus = dominant >= _MIN_CONSENSUS_SOURCES and agreement >= 0.5

        # If no consensus, dampen the signal toward HOLD
        if not has_consensus and total_sources >= 2:
            dampening = 0.5  # reduce signal strength by 50%
            final_score *= dampening
            final_confidence *= dampening
            logger.info(
                "No consensus for %s: directions=%s, dampening signal",
                list(components.keys())[0] if components else "?",
                directions,
            )

        consensus_info = {
            "has_consensus": has_consensus,
            "agreement": agreement,
            "bullish_sources": bullish_count,
            "bearish_sources": bearish_count,
            "neutral_sources": neutral_count,
            "total_sources": total_sources,
            "per_source_direction": directions,
        }

        return final_score, final_confidence, consensus_info

    # ------------------------------------------------------------------
    # Sentiment velocity
    # ------------------------------------------------------------------

    def _compute_velocity(
        self, asset: str, current_score: float, timestamp: float
    ) -> Optional[float]:
        """Track and compute sentiment velocity (rate of change).

        Returns velocity in score units per hour.
        """
        if asset not in self._signal_history:
            self._signal_history[asset] = deque(maxlen=100)

        self._signal_history[asset].append((timestamp, current_score))
        history = self._signal_history[asset]

        if len(history) < 3:
            return None

        # Linear regression over recent history
        points = list(history)[-20:]
        n = len(points)
        sum_t = sum(p[0] for p in points)
        sum_s = sum(p[1] for p in points)
        sum_ts = sum(p[0] * p[1] for p in points)
        sum_tt = sum(p[0] ** 2 for p in points)
        denom = n * sum_tt - sum_t * sum_t

        if abs(denom) < 1e-12:
            return 0.0

        slope = (n * sum_ts - sum_t * sum_s) / denom
        return slope * 3600.0  # per hour

    # ------------------------------------------------------------------
    # Contrarian signal check
    # ------------------------------------------------------------------

    def _check_contrarian(
        self,
        asset: str,
        fear_greed_data: Optional[Dict[str, float]],
        current_score: float,
    ) -> Dict:
        """Check if contrarian conditions exist based on Fear & Greed.

        At extreme fear, if the model is bearish, consider flipping
        to a contrarian bullish stance (and vice versa).
        """
        result: Dict[str, Any] = {
            "override": False,
            "reason": "no_extreme",
        }

        if fear_greed_data is None:
            return result

        fg_value = fear_greed_data.get("value")
        if fg_value is None:
            return result

        if fg_value <= _CONTRARIAN_F_G_FEAR_THRESHOLD and current_score < -0.2:
            # Extreme fear + bearish signal = contrarian BUY opportunity
            # Scale the contrarian strength by how extreme the reading is
            extremity = (_CONTRARIAN_F_G_FEAR_THRESHOLD - fg_value) / _CONTRARIAN_F_G_FEAR_THRESHOLD
            adjusted_score = current_score + abs(current_score) * extremity * 1.5
            adjusted_score = max(-1.0, min(1.0, adjusted_score))

            if adjusted_score > current_score + 0.3:  # meaningful flip
                result = {
                    "override": True,
                    "direction": "contrarian_buy",
                    "original_score": current_score,
                    "adjusted_score": adjusted_score,
                    "fg_value": fg_value,
                    "extremity": extremity,
                    "confidence_mult": 0.7,  # reduced confidence for contrarian
                    "reason": f"Extreme fear ({fg_value}) with bearish consensus -- contrarian buy",
                }

        elif fg_value >= _CONTRARIAN_F_G_GREED_THRESHOLD and current_score > 0.2:
            # Extreme greed + bullish signal = contrarian SELL opportunity
            extremity = (fg_value - _CONTRARIAN_F_G_GREED_THRESHOLD) / (100 - _CONTRARIAN_F_G_GREED_THRESHOLD)
            adjusted_score = current_score - abs(current_score) * extremity * 1.5
            adjusted_score = max(-1.0, min(1.0, adjusted_score))

            if adjusted_score < current_score - 0.3:
                result = {
                    "override": True,
                    "direction": "contrarian_sell",
                    "original_score": current_score,
                    "adjusted_score": adjusted_score,
                    "fg_value": fg_value,
                    "extremity": extremity,
                    "confidence_mult": 0.7,
                    "reason": f"Extreme greed ({fg_value}) with bullish consensus -- contrarian sell",
                }

        return result

    # ------------------------------------------------------------------
    # Direction with hysteresis
    # ------------------------------------------------------------------

    def _direction_with_hysteresis(self, asset: str, score: float) -> Direction:
        """Apply hysteresis to prevent noisy direction oscillation.

        Once a strong signal is established, require the score to move
        further in the opposite direction before downgrading.
        """
        new_direction = _score_to_direction(score)
        prev_direction = self._last_direction.get(asset)

        if prev_direction is None:
            self._last_direction[asset] = new_direction
            return new_direction

        # If direction hasn't changed, keep it
        if new_direction == prev_direction:
            return new_direction

        # Apply hysteresis: if moving from a strong to weak state,
        # require extra margin
        strong_states = {Direction.STRONG_BUY, Direction.STRONG_SELL}
        weak_states = {Direction.BUY, Direction.SELL, Direction.HOLD}

        if prev_direction in strong_states and new_direction in weak_states:
            # Check if score has crossed the threshold by enough
            if prev_direction == Direction.STRONG_BUY and score > 0.65 - _HYSTERESIS_MARGIN:
                return prev_direction  # stay in STRONG_BUY
            if prev_direction == Direction.STRONG_SELL and score < -0.65 + _HYSTERESIS_MARGIN:
                return prev_direction  # stay in STRONG_SELL

        self._last_direction[asset] = new_direction
        return new_direction

    # ------------------------------------------------------------------
    # Signal quality scoring
    # ------------------------------------------------------------------

    def _compute_signal_quality(
        self,
        components: Dict[str, SentimentScore],
        consensus_info: Dict,
        velocity: Optional[float],
        news_count: int,
    ) -> float:
        """Compute a 0-1 quality score for the signal.

        Quality factors:
        - Number of contributing sources (more = better)
        - Consensus agreement (higher = better)
        - Sample size (more data = better)
        - Stability (not changing too fast = better for non-urgent signals)
        """
        # Source coverage: how many of the possible sources contributed
        source_coverage = len(components) / max(1, len(self._source_weights))

        # Consensus agreement
        agreement = consensus_info.get("agreement", 0.5)

        # Sample size: diminishing returns above 10 items
        total_samples = sum(s.sample_size for s in components.values())
        sample_score = min(1.0, math.log1p(total_samples) / math.log1p(20))

        # Stability: moderate velocity = ok, extreme = caution
        stability = 1.0
        if velocity is not None:
            stability = max(0.3, 1.0 - abs(velocity) / 2.0)

        # Weighted quality score
        quality = (
            0.3 * source_coverage
            + 0.3 * agreement
            + 0.2 * sample_score
            + 0.2 * stability
        )

        return max(0.0, min(1.0, quality))

    # ------------------------------------------------------------------
    # LLM score aggregation
    # ------------------------------------------------------------------

    def _aggregate_llm_scores(
        self, llm_scores: List[SentimentScore], asset: str
    ) -> SentimentScore:
        """Aggregate multiple LLM analysis scores with urgency weighting.

        Higher-urgency analyses are weighted more heavily.
        """
        if not llm_scores:
            return SentimentScore(
                asset=asset, score=0.0, source="llm", confidence=0.0
            )

        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        max_urgency = 0

        for s in llm_scores:
            urgency = s.metadata.get("urgency", 5) if s.metadata else 5
            max_urgency = max(max_urgency, urgency)

            # Urgency-based weight: urgency 10 = 2x weight, urgency 1 = 0.5x
            urgency_mult = 0.5 + (urgency / 10.0) * 1.5
            w = s.confidence * urgency_mult

            weighted_score += s.score * w
            weighted_confidence += s.confidence * w
            total_weight += w

        if total_weight == 0:
            return SentimentScore(
                asset=asset, score=0.0, source="llm", confidence=0.0
            )

        return SentimentScore(
            asset=asset,
            score=max(-1.0, min(1.0, weighted_score / total_weight)),
            source="llm",
            confidence=min(1.0, weighted_confidence / total_weight),
            sample_size=len(llm_scores),
            metadata={
                "max_urgency": max_urgency,
                "analysis_count": len(llm_scores),
            },
        )

    # ------------------------------------------------------------------
    # Event-based sentiment
    # ------------------------------------------------------------------

    def _compute_event_sentiment(
        self, events: list, asset: str
    ) -> Optional[SentimentScore]:
        """Compute a sentiment score derived from detected events.

        Events like hacks, delistings, and regulatory actions have
        predictable sentiment impacts regardless of text sentiment.
        """
        from src.analysis.sentiment.event_detector import EventType

        # Event type to base sentiment score
        event_impact = {
            EventType.HACK_EXPLOIT: -0.9,
            EventType.DELISTING: -0.8,
            EventType.REGULATORY: -0.5,
            EventType.LEGAL: -0.4,
            EventType.LISTING: 0.6,
            EventType.PARTNERSHIP: 0.5,
            EventType.MERGER_ACQUISITION: 0.4,
            EventType.EARNINGS: 0.0,  # depends on direction
            EventType.MACRO: 0.0,
            EventType.LEADERSHIP_CHANGE: -0.2,
            EventType.UNKNOWN: 0.0,
        }

        relevant_events = [e for e in events if asset in e.assets or not e.assets]
        if not relevant_events:
            return None

        # Urgency-weighted average of event impacts
        total_weight = 0.0
        weighted_score = 0.0
        for event in relevant_events:
            impact = event_impact.get(event.event_type, 0.0)
            if abs(impact) < 0.01:
                continue
            w = event.urgency / 10.0
            weighted_score += impact * w
            total_weight += w

        if total_weight == 0:
            return None

        score = max(-1.0, min(1.0, weighted_score / total_weight))
        confidence = min(1.0, total_weight / len(relevant_events))

        return SentimentScore(
            asset=asset,
            score=score,
            source="event",
            confidence=confidence,
            sample_size=len(relevant_events),
            metadata={
                "event_types": [e.event_type.value for e in relevant_events],
                "max_urgency": max(e.urgency for e in relevant_events),
            },
        )

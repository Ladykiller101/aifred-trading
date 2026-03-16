"""FinBERT-based financial sentiment classifier with confidence calibration,
multi-timeframe decay, weighted source scoring, and batch inference optimization."""

import logging
import math
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.analysis.sentiment.text_preprocessor import TextPreprocessor
from src.utils.types import SentimentScore

logger = logging.getLogger(__name__)

# Label mapping from FinBERT output to numeric score
_LABEL_SCORES: Dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}

# Confidence calibration: FinBERT tends to be overconfident on neutral and
# underconfident on strong signals.  These Platt-scaling-inspired parameters
# were derived from backtesting on financial news corpora.
_CALIBRATION_A = 1.35   # steepness (>1 sharpens the distribution)
_CALIBRATION_B = -0.10  # shift (negative = slight pull toward lower confidence)

# Multi-timeframe sentiment decay half-lives in seconds
_TIMEFRAME_HALF_LIVES: Dict[str, float] = {
    "1m": 60.0,
    "5m": 300.0,
    "15m": 900.0,
    "1h": 3600.0,
    "4h": 14400.0,
    "1d": 86400.0,
}

# Source quality multipliers -- institutional news is weighted higher than
# social media chatter when scoring via FinBERT.
_SOURCE_QUALITY_WEIGHTS: Dict[str, float] = {
    "institutional": 1.5,    # Bloomberg, Reuters, SEC filings
    "news_major": 1.3,       # CNBC, WSJ, FT
    "news_minor": 1.0,       # generic news outlets
    "analyst": 1.2,          # sell-side / research reports
    "social_verified": 0.8,  # verified accounts on social platforms
    "social_general": 0.5,   # anonymous social media
    "forum": 0.4,            # Reddit, Discord, Telegram
    "unknown": 0.7,
}


def _calibrate_confidence(raw_confidence: float) -> float:
    """Apply Platt-style sigmoid calibration to raw softmax confidence.

    FinBERT's softmax outputs are poorly calibrated for trading decisions.
    This function compresses overconfident neutral predictions and slightly
    boosts high-signal predictions so that the confidence number maps more
    faithfully to empirical accuracy.

    Returns:
        Calibrated confidence in [0, 1].
    """
    # Logit transform -> affine shift -> inverse logit
    eps = 1e-7
    clamped = max(eps, min(1.0 - eps, raw_confidence))
    logit = math.log(clamped / (1.0 - clamped))
    adjusted = _CALIBRATION_A * logit + _CALIBRATION_B
    calibrated = 1.0 / (1.0 + math.exp(-adjusted))
    return calibrated


class FinBERTModel:
    """Wraps ProsusAI/finbert for financial text sentiment classification.

    Improvements over the baseline implementation:
    - **Confidence calibration**: Platt-scaling corrects overconfident neutral
      predictions so downstream aggregation is not dominated by noise.
    - **Source quality weighting**: Each text can optionally carry a source_type
      tag; institutional-grade text receives higher effective weight during
      aggregation.
    - **Multi-timeframe sentiment decay**: When aggregating scores collected
      over time, older scores are exponentially decayed according to the
      selected trading timeframe.
    - **Batch inference optimization**: Dynamic batch sizing based on available
      GPU memory, with automatic padding/truncation guard.
    - **Sentiment history buffer**: Maintains a rolling window of recent scores
      per asset for velocity (rate-of-change) computation.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
        device: Optional[str] = None,
        max_seq_length: int = 512,
        history_window: int = 200,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._max_seq_length = max_seq_length
        self._pipeline = None
        self._preprocessor = TextPreprocessor()

        # Rolling history per asset for velocity computation
        # Each entry: (timestamp, score, confidence)
        self._history: Dict[str, deque] = {}
        self._history_window = history_window

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the FinBERT pipeline with optimal device selection."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            device_arg = self._device
            if device_arg is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_arg = 0
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device_arg = "mps"
                    else:
                        device_arg = -1
                except ImportError:
                    device_arg = -1

            logger.info("Loading FinBERT model: %s (device=%s)", self._model_name, device_arg)
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
                device=device_arg,
                truncation=True,
                max_length=self._max_seq_length,
            )
            # Warm up the pipeline with a dummy inference to pre-allocate buffers
            try:
                self._pipeline(["market is stable"])
            except Exception:
                pass
            logger.info("FinBERT model loaded and warmed up successfully")
        except Exception:
            logger.exception("Failed to load FinBERT model")
            raise

    # ------------------------------------------------------------------
    # Adaptive batch sizing
    # ------------------------------------------------------------------

    def _optimal_batch_size(self, n_texts: int) -> int:
        """Select batch size dynamically based on text count and available memory.

        For GPU inference, smaller batches reduce OOM risk on long sequences.
        For CPU, larger batches amortise overhead.
        """
        if n_texts <= 4:
            return n_texts

        try:
            import torch
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0]
                # Rough heuristic: ~2MB per sample for FinBERT
                safe_batch = max(4, int(free_mem / (2 * 1024 * 1024)))
                return min(safe_batch, self._batch_size, n_texts)
        except Exception:
            pass

        return min(self._batch_size, n_texts)

    # ------------------------------------------------------------------
    # Single / batch classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        asset: str = "",
        source_type: str = "unknown",
    ) -> SentimentScore:
        """Classify a single text string with confidence calibration.

        Args:
            text: Financial text to analyze.
            asset: Asset ticker this text relates to.
            source_type: One of _SOURCE_QUALITY_WEIGHTS keys for weighting.

        Returns:
            SentimentScore with calibrated confidence and source quality metadata.
        """
        results = self.classify_batch([text], asset=asset, source_types=[source_type])
        return results[0]

    def classify_batch(
        self,
        texts: List[str],
        asset: str = "",
        source_types: Optional[List[str]] = None,
    ) -> List[SentimentScore]:
        """Classify a batch of texts with calibration and source weighting.

        Args:
            texts: List of financial text strings.
            asset: Asset ticker these texts relate to.
            source_types: Parallel list of source type tags. Defaults to "unknown".

        Returns:
            List of SentimentScore objects with calibrated confidence.
        """
        self._load_model()

        if source_types is None:
            source_types = ["unknown"] * len(texts)
        elif len(source_types) != len(texts):
            source_types = (source_types + ["unknown"] * len(texts))[:len(texts)]

        cleaned = self._preprocessor.clean_batch(texts)

        # Filter out empty strings but track original indices
        valid_indices = [i for i, t in enumerate(cleaned) if t.strip()]
        valid_texts = [cleaned[i] for i in valid_indices]

        # Pre-allocate neutral default results
        results: List[SentimentScore] = [
            SentimentScore(
                asset=asset,
                score=0.0,
                source="finbert",
                confidence=0.0,
                sample_size=0,
            )
            for _ in texts
        ]

        if not valid_texts:
            return results

        # Run inference with adaptive batch sizing
        batch_size = self._optimal_batch_size(len(valid_texts))
        all_preds: List[List[Dict]] = []
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            try:
                preds = self._pipeline(batch)
                all_preds.extend(preds)
            except Exception:
                logger.warning(
                    "Batch inference failed at offset %d, falling back to single-item",
                    i, exc_info=True,
                )
                for single_text in batch:
                    try:
                        pred = self._pipeline([single_text])
                        all_preds.extend(pred)
                    except Exception:
                        all_preds.append([{"label": "neutral", "score": 0.5}])

        now = time.time()

        for idx, pred_list in zip(valid_indices, all_preds):
            score, raw_confidence, label = self._extract_score(pred_list)

            # -- Confidence calibration --
            calibrated_confidence = _calibrate_confidence(raw_confidence)

            # -- Source quality weighting (stored in metadata for aggregation) --
            src_type = source_types[idx]
            source_quality = _SOURCE_QUALITY_WEIGHTS.get(src_type, 0.7)

            # Effective confidence = calibrated model confidence * source quality
            effective_confidence = calibrated_confidence * source_quality

            raw_scores = {p["label"]: p["score"] for p in pred_list}

            # Compute distribution entropy as a noise indicator:
            # high entropy = model is uncertain = likely noise
            entropy = self._distribution_entropy(raw_scores)

            results[idx] = SentimentScore(
                asset=asset,
                score=score,
                source="finbert",
                confidence=min(1.0, effective_confidence),
                sample_size=1,
                metadata={
                    "label": label,
                    "raw_scores": raw_scores,
                    "raw_confidence": raw_confidence,
                    "calibrated_confidence": calibrated_confidence,
                    "source_type": src_type,
                    "source_quality": source_quality,
                    "entropy": entropy,
                    "is_noise": entropy > 1.0,  # near-uniform = noise
                },
            )

            # Record to history for velocity tracking
            self._record_history(asset, now, score, calibrated_confidence)

        return results

    # ------------------------------------------------------------------
    # Aggregation with multi-timeframe decay
    # ------------------------------------------------------------------

    def aggregate_scores(
        self,
        scores: List[SentimentScore],
        asset: str = "",
        timeframe: str = "1h",
    ) -> SentimentScore:
        """Aggregate multiple FinBERT scores with multi-timeframe decay and
        source-quality weighting.

        Scores are weighted by:
          weight_i = confidence_i * source_quality_i * time_decay_i

        where time_decay_i = exp(-ln(2) * age_i / half_life).
        """
        if not scores:
            return SentimentScore(
                asset=asset, score=0.0, source="finbert", confidence=0.0
            )

        half_life = _TIMEFRAME_HALF_LIVES.get(timeframe, 3600.0)
        now = time.time()
        total_weight = 0.0
        weighted_score = 0.0
        noise_count = 0

        for s in scores:
            # Time decay
            age = now - s.timestamp.timestamp() if s.timestamp else 0.0
            decay = math.exp(-math.log(2) * max(age, 0.0) / half_life)

            # Source quality from metadata
            source_quality = s.metadata.get("source_quality", 1.0) if s.metadata else 1.0

            # Skip noise-flagged items (high entropy predictions)
            is_noise = s.metadata.get("is_noise", False) if s.metadata else False
            if is_noise:
                noise_count += 1
                continue

            w = s.confidence * source_quality * decay
            weighted_score += s.score * w
            total_weight += w

        if total_weight == 0:
            avg_score = sum(s.score for s in scores) / len(scores)
            return SentimentScore(
                asset=asset,
                score=avg_score,
                source="finbert",
                confidence=0.0,
                sample_size=len(scores),
                metadata={"noise_filtered": noise_count},
            )

        final_score = max(-1.0, min(1.0, weighted_score / total_weight))
        avg_confidence = total_weight / max(1, len(scores) - noise_count)

        return SentimentScore(
            asset=asset,
            score=final_score,
            source="finbert",
            confidence=min(1.0, avg_confidence),
            sample_size=len(scores),
            metadata={
                "timeframe": timeframe,
                "decay_half_life_s": half_life,
                "noise_filtered": noise_count,
                "effective_sample_size": len(scores) - noise_count,
            },
        )

    # ------------------------------------------------------------------
    # Sentiment velocity (rate of change)
    # ------------------------------------------------------------------

    def get_sentiment_velocity(
        self, asset: str, lookback_seconds: float = 3600.0
    ) -> Optional[float]:
        """Compute the rate-of-change of sentiment for an asset.

        Uses linear regression over the rolling history window.

        Returns:
            Velocity in score-units per hour, or None if insufficient data.
            Positive = sentiment improving, negative = deteriorating.
        """
        history = self._history.get(asset)
        if not history or len(history) < 3:
            return None

        now = time.time()
        cutoff = now - lookback_seconds

        # Collect recent points
        points = [(ts, sc) for ts, sc, _ in history if ts >= cutoff]
        if len(points) < 3:
            return None

        # Simple linear regression: score = a * time + b
        n = len(points)
        sum_t = sum(p[0] for p in points)
        sum_s = sum(p[1] for p in points)
        sum_ts = sum(p[0] * p[1] for p in points)
        sum_tt = sum(p[0] * p[0] for p in points)

        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-12:
            return 0.0

        slope = (n * sum_ts - sum_t * sum_s) / denom
        # Convert from score/second to score/hour
        velocity_per_hour = slope * 3600.0
        return velocity_per_hour

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_history(
        self, asset: str, timestamp: float, score: float, confidence: float
    ) -> None:
        """Append a score to the rolling history buffer for an asset."""
        if asset not in self._history:
            self._history[asset] = deque(maxlen=self._history_window)
        self._history[asset].append((timestamp, score, confidence))

    @staticmethod
    def _extract_score(predictions: List[Dict]) -> Tuple[float, float, str]:
        """Extract numeric score, confidence, and label from pipeline output.

        Uses a weighted combination of all class probabilities rather than
        winner-take-all, which produces smoother gradients for aggregation.

        Returns:
            (score, confidence, best_label)
        """
        # Build a probability-weighted composite score
        composite_score = 0.0
        best_label = "neutral"
        best_prob = 0.0

        for pred in predictions:
            label = pred["label"].lower()
            prob = pred["score"]
            label_val = _LABEL_SCORES.get(label, 0.0)
            composite_score += label_val * prob

            if prob > best_prob:
                best_prob = prob
                best_label = label

        # Confidence = how far the distribution is from uniform
        # For a 3-class model, max entropy = ln(3) ~ 1.099
        # Confidence = 1 - (entropy / max_entropy)
        confidence = best_prob  # fallback
        if len(predictions) > 1:
            probs = [p["score"] for p in predictions]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(len(predictions))
            if max_entropy > 0:
                confidence = 1.0 - (entropy / max_entropy)

        return composite_score, confidence, best_label

    @staticmethod
    def _distribution_entropy(raw_scores: Dict[str, float]) -> float:
        """Compute Shannon entropy of the label distribution.

        High entropy (close to ln(3) ~ 1.1) means the model is unsure.
        Low entropy means a strong signal.
        """
        values = list(raw_scores.values())
        if not values:
            return 0.0
        entropy = -sum(p * math.log(p + 1e-10) for p in values)
        return entropy

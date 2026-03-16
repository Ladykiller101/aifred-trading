"""Social media sentiment aggregation with source reliability weighting,
spam/bot filtering, velocity-based momentum scoring, and engagement
quality metrics."""

import hashlib
import logging
import math
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from src.analysis.sentiment.text_preprocessor import TextPreprocessor
from src.utils.types import SentimentScore

logger = logging.getLogger(__name__)

# Default subreddits to monitor, ordered by quality for weighting
DEFAULT_SUBREDDITS = ["cryptocurrency", "bitcoin", "wallstreetbets"]

# Subreddit quality tiers -- higher-quality subreddits with stricter
# moderation produce less noise.
_SUBREDDIT_QUALITY: Dict[str, float] = {
    # Tier 1: Well-moderated, substantive discussion
    "cryptocurrency": 0.7,
    "bitcoin": 0.7,
    "ethereum": 0.7,
    "investing": 0.8,
    "stocks": 0.7,
    "SecurityAnalysis": 0.9,
    # Tier 2: High volume, mixed quality
    "wallstreetbets": 0.4,
    "CryptoMoonShots": 0.2,
    "SatoshiStreetBets": 0.3,
    "pennystocks": 0.3,
    # Tier 3: News-oriented
    "finance": 0.8,
    "economics": 0.8,
}

# Ticker mention patterns for asset detection
_ASSET_ALIASES: Dict[str, str] = {
    "btc": "BTC/USDT", "bitcoin": "BTC/USDT",
    "eth": "ETH/USDT", "ethereum": "ETH/USDT",
    "sol": "SOL/USDT", "solana": "SOL/USDT",
    "bnb": "BNB/USDT", "binance coin": "BNB/USDT",
    "xrp": "XRP/USDT", "ripple": "XRP/USDT",
    "ada": "ADA/USDT", "cardano": "ADA/USDT",
    "doge": "DOGE/USDT", "dogecoin": "DOGE/USDT",
    "avax": "AVAX/USDT", "avalanche": "AVAX/USDT",
    "aapl": "AAPL", "apple": "AAPL",
    "msft": "MSFT", "microsoft": "MSFT",
    "googl": "GOOGL", "google": "GOOGL",
    "nvda": "NVDA", "nvidia": "NVDA",
    "tsla": "TSLA", "tesla": "TSLA",
    "amzn": "AMZN", "amazon": "AMZN",
    "meta": "META", "facebook": "META",
    "spy": "SPY", "qqq": "QQQ",
}

# ---------------------------------------------------------------------------
# Spam / bot detection heuristics
# ---------------------------------------------------------------------------

# Patterns commonly found in spam/scam posts
_SPAM_PATTERNS = [
    re.compile(r"(?i)\b(?:guaranteed|100%|moon|lambo|cant lose|free money)\b"),
    re.compile(r"(?i)\b(?:join|telegram|discord|whatsapp|signal group)\b.*(?:link|join|click)"),
    re.compile(r"(?i)\b(?:pump|shill|rocket|diamond hands|to the moon)\b.*\b(?:pump|shill|rocket)\b"),
    re.compile(r"(?i)(?:send|deposit)\s+\d+\s*(?:btc|eth|usdt)"),
    re.compile(r"(?i)\b(?:airdrop|giveaway|double your)\b"),
    re.compile(r"(?:[\U0001F680]){3,}"),  # 3+ rocket emojis
]

# Account age / karma thresholds for bot detection
_MIN_ACCOUNT_AGE_DAYS = 30
_MIN_KARMA = 100


def _is_spam(text: str) -> bool:
    """Check if text matches common spam/scam patterns."""
    for pattern in _SPAM_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _engagement_quality_score(
    upvotes: int, downvotes: int, num_comments: int, text_length: int
) -> float:
    """Compute an engagement quality score in [0, 1].

    Quality factors:
    - Higher upvote ratio = more community agreement
    - More comments = more discussion (substantive)
    - Longer text = more thoughtful (up to a point)
    - Very short posts with high upvotes = likely meme (low quality)
    """
    total_votes = upvotes + downvotes
    if total_votes == 0:
        vote_ratio = 0.5
    else:
        vote_ratio = upvotes / total_votes

    # Comment engagement: log scale, capped
    comment_score = min(1.0, math.log1p(num_comments) / math.log1p(100))

    # Text substance: posts between 100-2000 chars score highest
    if text_length < 20:
        text_score = 0.1
    elif text_length < 100:
        text_score = 0.4
    elif text_length < 2000:
        text_score = 0.8 + 0.2 * min(1.0, text_length / 2000)
    else:
        text_score = 0.9  # very long can be copy-paste

    # Weighted combination
    quality = 0.4 * vote_ratio + 0.3 * comment_score + 0.3 * text_score
    return max(0.0, min(1.0, quality))


class SocialAggregator:
    """Monitors Reddit for sentiment signals with source reliability weighting,
    spam/bot filtering, velocity-based momentum scoring, and engagement
    quality metrics.

    Improvements over baseline:
    - **Source reliability weighting**: Each subreddit has a quality tier.
      Posts from well-moderated subs (r/investing) weight more than meme
      subs (r/wallstreetbets).
    - **Spam/bot filtering**: Regex-based spam pattern detection plus
      engagement heuristics to filter low-quality posts before sentiment
      scoring.
    - **Engagement quality metrics**: Posts are scored by upvote ratio,
      comment count, and text length -- substantive discussion is weighted
      more than low-effort memes.
    - **Velocity-based momentum scoring**: Tracks mention velocity (mentions
      per hour) and its acceleration.  A sudden spike in mentions is often
      a leading indicator of price moves.
    - **Deduplication**: Content hashing prevents the same reposted text
      from being counted multiple times.
    - **Time-weighted aggregation**: Recent posts contribute more to the
      sentiment score than older ones.
    """

    def __init__(
        self,
        subreddits: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "ai-trading-bot:v2.0 (by /u/trading_bot)",
        requests_per_minute: int = 30,
        spam_filter_enabled: bool = True,
        min_engagement_quality: float = 0.2,
    ):
        self._subreddits = subreddits or DEFAULT_SUBREDDITS
        self._client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self._user_agent = user_agent
        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._reddit = None
        self._preprocessor = TextPreprocessor()
        self._finbert = None
        self._spam_filter_enabled = spam_filter_enabled
        self._min_engagement_quality = min_engagement_quality

        # Volume tracking for spike detection
        self._volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Mention velocity tracking: asset -> deque of (timestamp, count)
        self._mention_velocity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Content deduplication
        self._seen_hashes: Dict[str, float] = {}  # hash -> timestamp
        self._dedup_ttl = 3600.0 * 6  # 6 hours

    def _get_reddit(self):
        """Lazy-load PRAW Reddit instance."""
        if self._reddit is not None:
            return self._reddit
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            logger.info("Reddit PRAW client initialized")
            return self._reddit
        except Exception:
            logger.exception("Failed to initialize Reddit client")
            raise

    def _get_finbert(self):
        """Lazy-load FinBERT for scoring social text."""
        if self._finbert is None:
            from src.analysis.sentiment.finbert_model import FinBERTModel
            self._finbert = FinBERTModel()
        return self._finbert

    def _rate_limit(self) -> None:
        """Block until rate limit window allows another request."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Content hashing & deduplication
    # ------------------------------------------------------------------

    def _content_hash(self, text: str) -> str:
        """Compute a content hash for deduplication."""
        # Normalize: lowercase, strip whitespace, remove punctuation
        normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _is_duplicate(self, text: str) -> bool:
        """Check if we have already seen this content recently."""
        self._prune_seen_hashes()
        h = self._content_hash(text)
        if h in self._seen_hashes:
            return True
        self._seen_hashes[h] = time.time()
        return False

    def _prune_seen_hashes(self) -> None:
        """Remove expired entries from the dedup cache."""
        now = time.time()
        expired = [h for h, ts in self._seen_hashes.items() if now - ts > self._dedup_ttl]
        for h in expired:
            del self._seen_hashes[h]

    # ------------------------------------------------------------------
    # Asset detection
    # ------------------------------------------------------------------

    def _detect_asset(self, text: str) -> Optional[str]:
        """Detect which asset a text is about using keyword matching."""
        text_lower = text.lower()
        for keyword, asset in _ASSET_ALIASES.items():
            # Use word boundary matching to avoid false positives
            # e.g., "sol" inside "solution" should not match
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                return asset
        return None

    def _detect_all_assets(self, text: str) -> List[str]:
        """Detect all assets mentioned in a text."""
        text_lower = text.lower()
        assets: Set[str] = set()
        for keyword, asset in _ASSET_ALIASES.items():
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                assets.add(asset)
        return sorted(assets)

    # ------------------------------------------------------------------
    # Post fetching with quality filtering
    # ------------------------------------------------------------------

    def fetch_recent_posts(
        self, subreddit: str, limit: int = 50, sort: str = "hot"
    ) -> List[Dict]:
        """Fetch recent posts from a subreddit with spam filtering and
        engagement quality scoring.

        Returns list of dicts with quality metadata attached.
        """
        self._rate_limit()
        reddit = self._get_reddit()
        sub = reddit.subreddit(subreddit)

        getter = {"hot": sub.hot, "new": sub.new, "rising": sub.rising}.get(sort, sub.hot)

        posts = []
        spam_filtered = 0
        dedup_filtered = 0
        low_quality_filtered = 0

        for submission in getter(limit=limit):
            combined = f"{submission.title} {submission.selftext or ''}"

            # Spam filter
            if self._spam_filter_enabled and _is_spam(combined):
                spam_filtered += 1
                continue

            # Deduplication
            if self._is_duplicate(combined):
                dedup_filtered += 1
                continue

            # Engagement quality
            upvotes = max(0, submission.score)
            # Reddit's upvote_ratio gives us the ratio; back out downvotes
            ratio = getattr(submission, "upvote_ratio", 0.5)
            downvotes = int(upvotes * (1.0 - ratio) / max(ratio, 0.01)) if ratio < 1.0 else 0
            num_comments = submission.num_comments
            text_len = len(combined)

            quality = _engagement_quality_score(upvotes, downvotes, num_comments, text_len)
            if quality < self._min_engagement_quality:
                low_quality_filtered += 1
                continue

            subreddit_quality = _SUBREDDIT_QUALITY.get(subreddit, 0.5)

            posts.append({
                "title": submission.title,
                "selftext": submission.selftext or "",
                "score": upvotes,
                "downvotes": downvotes,
                "num_comments": num_comments,
                "created_utc": submission.created_utc,
                "subreddit": subreddit,
                "subreddit_quality": subreddit_quality,
                "engagement_quality": quality,
                "combined_quality": quality * subreddit_quality,
                "text_length": text_len,
            })

        if spam_filtered or dedup_filtered or low_quality_filtered:
            logger.info(
                "r/%s filtering: spam=%d dedup=%d low_quality=%d kept=%d",
                subreddit, spam_filtered, dedup_filtered, low_quality_filtered, len(posts),
            )

        return posts

    # ------------------------------------------------------------------
    # Velocity-based momentum
    # ------------------------------------------------------------------

    def _record_mention_velocity(self, asset: str, count: int) -> None:
        """Record a mention count observation for velocity tracking."""
        self._mention_velocity[asset].append((time.time(), count))

    def get_mention_velocity(self, asset: str) -> Dict[str, float]:
        """Compute mention velocity and acceleration for an asset.

        Returns:
            Dict with keys:
            - 'velocity': mentions per hour (rate of change)
            - 'acceleration': change in velocity per hour
            - 'current_rate': raw mentions per hour in latest window
        """
        history = self._mention_velocity.get(asset)
        if not history or len(history) < 2:
            return {"velocity": 0.0, "acceleration": 0.0, "current_rate": 0.0}

        points = list(history)

        # Current rate: latest observation annualized to per-hour
        if len(points) >= 2:
            dt = points[-1][0] - points[-2][0]
            if dt > 0:
                current_rate = points[-1][1] / (dt / 3600.0)
            else:
                current_rate = 0.0
        else:
            current_rate = 0.0

        # Velocity: slope of mentions over time (linear regression)
        n = len(points)
        sum_t = sum(p[0] for p in points)
        sum_c = sum(p[1] for p in points)
        sum_tc = sum(p[0] * p[1] for p in points)
        sum_tt = sum(p[0] * p[0] for p in points)
        denom = n * sum_tt - sum_t * sum_t
        if abs(denom) < 1e-12:
            velocity = 0.0
        else:
            velocity = (n * sum_tc - sum_t * sum_c) / denom * 3600.0  # per hour

        # Acceleration: difference of recent velocities
        acceleration = 0.0
        if len(points) >= 4:
            mid = len(points) // 2
            first_half = points[:mid]
            second_half = points[mid:]

            def _half_velocity(pts: list) -> float:
                n2 = len(pts)
                if n2 < 2:
                    return 0.0
                dt2 = pts[-1][0] - pts[0][0]
                if dt2 <= 0:
                    return 0.0
                total = sum(p[1] for p in pts)
                return total / (dt2 / 3600.0)

            v1 = _half_velocity(first_half)
            v2 = _half_velocity(second_half)
            time_span = (second_half[-1][0] - first_half[0][0]) / 3600.0
            if time_span > 0:
                acceleration = (v2 - v1) / time_span

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "current_rate": current_rate,
        }

    # ------------------------------------------------------------------
    # Core sentiment scoring
    # ------------------------------------------------------------------

    def get_asset_momentum(
        self, asset: str, limit_per_sub: int = 50
    ) -> SentimentScore:
        """Calculate quality-weighted social momentum score for an asset.

        Each post's contribution to the aggregate sentiment is weighted by:
          weight = engagement_quality * subreddit_quality * recency_decay

        This ensures that a thoughtful analysis on r/investing outweighs
        a low-effort meme on r/wallstreetbets.

        Returns:
            SentimentScore with velocity metadata.
        """
        all_texts: List[str] = []
        all_weights: List[float] = []
        mention_count = 0
        now = time.time()

        for subreddit in self._subreddits:
            try:
                posts = self.fetch_recent_posts(subreddit, limit=limit_per_sub)
                for post in posts:
                    combined = f"{post['title']} {post['selftext']}"
                    detected = self._detect_asset(combined)
                    if detected == asset:
                        mention_count += 1

                        # Recency decay: half-life = 6 hours
                        age_hours = (now - post["created_utc"]) / 3600.0
                        recency = math.exp(-math.log(2) * age_hours / 6.0)

                        weight = post["combined_quality"] * recency
                        all_texts.append(combined)
                        all_weights.append(weight)
            except Exception:
                logger.warning("Failed to fetch from r/%s", subreddit, exc_info=True)

        # Record velocity observation
        self._record_mention_velocity(asset, mention_count)
        velocity_data = self.get_mention_velocity(asset)

        if not all_texts:
            return SentimentScore(
                asset=asset,
                score=0.0,
                source="social",
                confidence=0.0,
                sample_size=0,
                metadata={
                    "mention_count": 0,
                    "velocity": velocity_data,
                },
            )

        # Run FinBERT on all qualifying texts
        finbert = self._get_finbert()
        scores = finbert.classify_batch(all_texts, asset=asset, source_types=["forum"] * len(all_texts))

        # Quality-weighted aggregation (not just confidence-weighted)
        total_weight = 0.0
        weighted_score = 0.0
        for score_obj, quality_weight in zip(scores, all_weights):
            w = score_obj.confidence * quality_weight
            weighted_score += score_obj.score * w
            total_weight += w

        if total_weight > 0:
            final_score = max(-1.0, min(1.0, weighted_score / total_weight))
            avg_confidence = total_weight / len(scores)
        else:
            final_score = 0.0
            avg_confidence = 0.0

        # Velocity bonus/penalty: rapid mention acceleration amplifies signal
        velocity_multiplier = 1.0
        if velocity_data["acceleration"] > 2.0:
            velocity_multiplier = 1.3  # accelerating mentions boost signal
        elif velocity_data["acceleration"] < -2.0:
            velocity_multiplier = 0.7  # decelerating mentions dampen signal

        final_score = max(-1.0, min(1.0, final_score * velocity_multiplier))

        return SentimentScore(
            asset=asset,
            score=final_score,
            source="social",
            confidence=min(1.0, avg_confidence),
            sample_size=len(all_texts),
            metadata={
                "subreddits": self._subreddits,
                "mention_count": mention_count,
                "quality_filtered_count": len(all_texts),
                "velocity": velocity_data,
                "velocity_multiplier": velocity_multiplier,
                "avg_engagement_quality": sum(all_weights) / len(all_weights) if all_weights else 0,
            },
        )

    # ------------------------------------------------------------------
    # Trending tickers with quality weighting
    # ------------------------------------------------------------------

    def get_trending_tickers(
        self, limit_per_sub: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Find trending tickers with quality-weighted mention scores.

        Returns:
            Dict mapping asset ticker to a dict with:
            - 'raw_mentions': unweighted count
            - 'quality_score': quality-weighted mention score
            - 'velocity': mention velocity data
        """
        mentions: Dict[str, int] = defaultdict(int)
        quality_scores: Dict[str, float] = defaultdict(float)

        for subreddit in self._subreddits:
            try:
                posts = self.fetch_recent_posts(subreddit, limit=limit_per_sub)
                sub_quality = _SUBREDDIT_QUALITY.get(subreddit, 0.5)

                for post in posts:
                    combined = f"{post['title']} {post['selftext']}"
                    detected_assets = self._detect_all_assets(combined)

                    for asset in detected_assets:
                        mentions[asset] += 1
                        quality_scores[asset] += post["engagement_quality"] * sub_quality

                    # Also check cashtag tickers
                    tickers = self._preprocessor.extract_tickers(combined)
                    for ticker in tickers:
                        ticker_lower = ticker.lower()
                        if ticker_lower in _ASSET_ALIASES:
                            asset = _ASSET_ALIASES[ticker_lower]
                            if asset not in detected_assets:
                                mentions[asset] += 1
                                quality_scores[asset] += post["engagement_quality"] * sub_quality
            except Exception:
                logger.warning("Failed to fetch from r/%s", subreddit, exc_info=True)

        result = {}
        for asset in sorted(mentions.keys(), key=lambda a: quality_scores[a], reverse=True):
            self._record_mention_velocity(asset, mentions[asset])
            result[asset] = {
                "raw_mentions": mentions[asset],
                "quality_score": quality_scores[asset],
                "velocity": self.get_mention_velocity(asset),
            }

        return result

    # ------------------------------------------------------------------
    # Volume spike detection
    # ------------------------------------------------------------------

    def detect_volume_spike(
        self, subreddit: str, window_size: int = 5, spike_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """Detect if current post volume is spiking above recent average.

        Returns a dict with spike detection details rather than just a bool.

        Args:
            subreddit: Subreddit to check.
            window_size: Rolling window size for average computation.
            spike_threshold: Multiplier above average that constitutes a spike.
        """
        try:
            posts = self.fetch_recent_posts(subreddit, limit=25)
            current_volume = len(posts)
        except Exception:
            return {"is_spike": False, "reason": "fetch_failed"}

        history = self._volume_history[subreddit]
        history.append(current_volume)

        if len(history) < window_size:
            return {
                "is_spike": False,
                "current_volume": current_volume,
                "reason": "insufficient_history",
                "history_length": len(history),
            }

        recent = list(history)[-window_size:]
        avg = sum(recent) / len(recent)
        std_dev = (sum((x - avg) ** 2 for x in recent) / len(recent)) ** 0.5

        is_spike = current_volume > avg * spike_threshold
        z_score = (current_volume - avg) / std_dev if std_dev > 0 else 0.0

        return {
            "is_spike": is_spike,
            "current_volume": current_volume,
            "average_volume": avg,
            "std_dev": std_dev,
            "z_score": z_score,
            "spike_ratio": current_volume / avg if avg > 0 else 0,
            "threshold": spike_threshold,
        }

    # ------------------------------------------------------------------
    # Multi-asset convenience
    # ------------------------------------------------------------------

    def get_all_sentiments(
        self, assets: List[str], limit_per_sub: int = 50
    ) -> Dict[str, SentimentScore]:
        """Get quality-weighted social sentiment scores for multiple assets."""
        results = {}
        for asset in assets:
            results[asset] = self.get_asset_momentum(asset, limit_per_sub)
        return results

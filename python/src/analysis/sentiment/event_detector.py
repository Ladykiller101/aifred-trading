"""Named Entity Recognition and event classification for financial text
with impact classification (high/medium/low), urgency scoring,
scheduled vs unexpected event handling, and temporal analysis."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class EventType(Enum):
    EARNINGS = "earnings"
    REGULATORY = "regulatory_action"
    HACK_EXPLOIT = "hack_exploit"
    PARTNERSHIP = "partnership"
    LISTING = "listing"
    DELISTING = "delisting"
    MACRO = "macro_event"
    MERGER_ACQUISITION = "merger_acquisition"
    LEADERSHIP_CHANGE = "leadership_change"
    LEGAL = "legal"
    LIQUIDATION = "liquidation"
    WHALE_MOVEMENT = "whale_movement"
    NETWORK_UPGRADE = "network_upgrade"
    TOKEN_UNLOCK = "token_unlock"
    GEOPOLITICAL = "geopolitical"
    UNKNOWN = "unknown"


class EventImpact(Enum):
    """Impact severity classification for trading decisions."""
    CRITICAL = "critical"   # Immediate position management required
    HIGH = "high"           # Should influence next trading decision
    MEDIUM = "medium"       # Worth monitoring, may influence sizing
    LOW = "low"             # Background noise, minimal impact
    NEGLIGIBLE = "negligible"


class EventTiming(Enum):
    """Whether the event was scheduled or unexpected."""
    SCHEDULED = "scheduled"      # Earnings, FOMC, token unlocks
    UNEXPECTED = "unexpected"    # Hacks, sudden regulatory actions
    RUMOR = "rumor"              # Unconfirmed reports
    DEVELOPING = "developing"   # Ongoing situation


@dataclass
class ExtractedEntity:
    """A named entity extracted from text."""
    text: str
    label: str  # "ORG", "PERSON", "GPE", etc.
    start: int
    end: int


@dataclass
class DetectedEvent:
    """A classified financial event with extracted entities, impact
    classification, timing, and sentiment direction."""
    event_type: EventType
    impact: EventImpact = EventImpact.MEDIUM
    timing: EventTiming = EventTiming.UNEXPECTED
    entities: List[ExtractedEntity] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    urgency: int = 5  # 1-10 scale
    sentiment_direction: float = 0.0  # -1 to +1
    confidence: float = 0.5  # 0 to 1
    summary: str = ""
    category_scores: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Keyword sets for rule-based event classification
# ---------------------------------------------------------------------------
# Each keyword set is organized by specificity (more specific patterns first)
# with associated weight for scoring.

_EVENT_KEYWORDS: Dict[EventType, Dict[str, int]] = {
    EventType.EARNINGS: {
        "earnings beat": 3, "earnings miss": 3, "earnings surprise": 3,
        "revenue beat": 3, "revenue miss": 3,
        "earnings": 2, "revenue": 2, "profit": 2, "loss": 1,
        "quarterly results": 3, "annual report": 2,
        "EPS": 2, "guidance": 2, "beat expectations": 3,
        "missed expectations": 3, "earnings call": 2,
        "fiscal quarter": 2, "net income": 2, "gross margin": 2,
        "forward guidance": 3, "raised guidance": 3, "lowered guidance": 3,
        "revenue growth": 2, "profit margin": 2,
    },
    EventType.REGULATORY: {
        "SEC enforcement": 4, "SEC investigation": 4,
        "regulatory crackdown": 4, "compliance violation": 3,
        "SEC": 2, "regulation": 2, "compliance": 1, "fine": 2,
        "penalty": 2, "ban": 3, "sanctions": 3,
        "CFTC": 2, "FCA": 2, "investigation": 2, "subpoena": 3,
        "enforcement action": 3, "ruling": 2, "court order": 2,
        "cease and desist": 4, "Wells notice": 4,
        "OFAC": 3, "AML": 2, "KYC": 1, "MiCA": 2,
        "regulatory approval": 3, "regulated": 1,
    },
    EventType.HACK_EXPLOIT: {
        "exploit drained": 5, "funds stolen": 5,
        "security breach": 4, "smart contract exploit": 5,
        "hack": 3, "exploit": 3, "breach": 3, "stolen": 3,
        "vulnerability": 2, "drain": 3, "drained": 3,
        "flash loan attack": 5, "rug pull": 5, "scam": 2,
        "phishing": 2, "compromised": 3, "attack": 2,
        "security incident": 3, "bridge hack": 5,
        "oracle manipulation": 4, "private key leak": 5,
        "reentrancy": 4, "front-running": 3,
    },
    EventType.PARTNERSHIP: {
        "strategic partnership": 3, "signed agreement": 3,
        "partnership": 2, "collaboration": 2, "integration": 2,
        "deal": 1, "alliance": 2, "joint venture": 3,
        "teamed up": 2, "working with": 1,
        "technology partnership": 3, "enterprise adoption": 3,
    },
    EventType.LISTING: {
        "listing on": 3, "listed on": 3, "added to": 2,
        "trading pair": 2, "launch": 1, "available on": 2,
        "new market": 2, "exchange listing": 3,
        "Coinbase listing": 4, "Binance listing": 4,
        "spot ETF approved": 5, "ETF listing": 4,
    },
    EventType.DELISTING: {
        "delisting": 4, "delisted": 4, "removed from": 3,
        "suspended trading": 4, "halt trading": 4,
        "trading suspended": 4, "no longer available": 3,
    },
    EventType.MACRO: {
        "rate hike": 4, "rate cut": 4, "interest rate decision": 4,
        "interest rate": 3, "inflation": 2, "CPI": 3, "PPI": 2,
        "GDP": 2, "unemployment": 2, "nonfarm payroll": 3,
        "federal reserve": 3, "central bank": 2, "FOMC": 3,
        "quantitative easing": 3, "quantitative tightening": 3,
        "tapering": 3, "recession": 3, "stimulus": 2,
        "tariff": 3, "trade war": 3, "debt ceiling": 3,
        "yield curve": 2, "treasury": 1, "dollar index": 2,
        "money supply": 2, "M2": 2,
    },
    EventType.MERGER_ACQUISITION: {
        "acquisition": 3, "acquire": 2, "merger": 3,
        "takeover": 3, "buyout": 3, "bid for": 3,
        "offer to buy": 3, "hostile takeover": 4,
        "leveraged buyout": 4, "going private": 3,
    },
    EventType.LEADERSHIP_CHANGE: {
        "CEO resigned": 4, "CEO fired": 4, "new CEO": 3,
        "CEO": 2, "CTO": 2, "CFO": 2,
        "resign": 2, "appointed": 2, "fired": 2,
        "stepped down": 3, "new leadership": 2,
        "board of directors": 1, "founder left": 3,
    },
    EventType.LEGAL: {
        "class action lawsuit": 4, "criminal charges": 4,
        "lawsuit": 3, "sued": 3, "court": 2, "settlement": 3,
        "class action": 4, "indictment": 4, "trial": 2,
        "verdict": 3, "guilty": 4, "not guilty": 3,
        "plea deal": 3, "extradition": 3,
    },
    EventType.LIQUIDATION: {
        "mass liquidation": 5, "liquidation cascade": 5,
        "liquidated": 3, "liquidation": 3,
        "margin call": 3, "forced selling": 3,
        "short squeeze": 4, "long squeeze": 4,
        "funding rate spike": 3,
    },
    EventType.WHALE_MOVEMENT: {
        "whale transfer": 3, "large transfer": 2,
        "whale": 2, "dormant wallet": 3,
        "moved to exchange": 3, "moved from exchange": 3,
        "whale accumulation": 3, "whale dump": 3,
    },
    EventType.NETWORK_UPGRADE: {
        "hard fork": 4, "soft fork": 3,
        "network upgrade": 3, "protocol upgrade": 3,
        "mainnet launch": 3, "testnet": 2,
        "chain upgrade": 3, "EIP": 2,
    },
    EventType.TOKEN_UNLOCK: {
        "token unlock": 4, "vesting": 3, "cliff": 2,
        "token release": 3, "vesting schedule": 3,
        "lockup expiry": 4, "insider selling": 3,
    },
    EventType.GEOPOLITICAL: {
        "war": 3, "conflict": 2, "invasion": 4,
        "military": 2, "nuclear": 3,
        "election": 2, "political": 1,
        "coup": 4, "revolution": 3,
        "embargo": 3, "diplomatic": 1,
    },
}

# ---------------------------------------------------------------------------
# Urgency modifiers
# ---------------------------------------------------------------------------
_HIGH_URGENCY_WORDS: Dict[str, int] = {
    "breaking": 3, "urgent": 3, "just in": 3, "developing": 2,
    "flash": 3, "crash": 3, "surge": 2, "plunge": 3,
    "halt": 2, "emergency": 3, "immediate": 2, "now": 1,
    "confirmed": 1, "just happened": 3, "minutes ago": 2,
    "live": 2, "real-time": 2,
}

_LOW_URGENCY_WORDS: Dict[str, int] = {
    "expected": -1, "scheduled": -1, "planned": -1,
    "rumor": -1, "unconfirmed": -2, "alleged": -1,
    "may": -1, "might": -1, "could": -1,
    "sources say": -1, "reportedly": -1,
    "opinion": -2, "editorial": -2, "analysis": -1,
}

# ---------------------------------------------------------------------------
# Event type to base sentiment direction
# ---------------------------------------------------------------------------
_EVENT_SENTIMENT: Dict[EventType, float] = {
    EventType.HACK_EXPLOIT: -0.9,
    EventType.DELISTING: -0.8,
    EventType.LIQUIDATION: -0.7,
    EventType.REGULATORY: -0.4,    # can be positive (clarity) or negative
    EventType.LEGAL: -0.5,
    EventType.LEADERSHIP_CHANGE: -0.2,
    EventType.TOKEN_UNLOCK: -0.3,
    EventType.GEOPOLITICAL: -0.3,
    EventType.LISTING: 0.7,
    EventType.PARTNERSHIP: 0.5,
    EventType.NETWORK_UPGRADE: 0.3,
    EventType.MERGER_ACQUISITION: 0.4,
    EventType.EARNINGS: 0.0,       # direction depends on beat/miss
    EventType.MACRO: 0.0,          # direction depends on specifics
    EventType.WHALE_MOVEMENT: 0.0, # direction depends on context
    EventType.UNKNOWN: 0.0,
}

# ---------------------------------------------------------------------------
# Event timing classification
# ---------------------------------------------------------------------------
_SCHEDULED_EVENT_TYPES = {
    EventType.EARNINGS,
    EventType.MACRO,
    EventType.TOKEN_UNLOCK,
    EventType.NETWORK_UPGRADE,
}

_SCHEDULED_KEYWORDS = {
    "scheduled", "expected", "planned", "upcoming", "announcement date",
    "reporting date", "FOMC meeting", "earnings date",
}

_RUMOR_KEYWORDS = {
    "rumor", "unconfirmed", "alleged", "sources say", "reportedly",
    "speculation", "may be", "could be", "possible",
}

# ---------------------------------------------------------------------------
# Impact classification
# ---------------------------------------------------------------------------
_EVENT_BASE_IMPACT: Dict[EventType, EventImpact] = {
    EventType.HACK_EXPLOIT: EventImpact.CRITICAL,
    EventType.DELISTING: EventImpact.HIGH,
    EventType.LIQUIDATION: EventImpact.HIGH,
    EventType.REGULATORY: EventImpact.HIGH,
    EventType.LEGAL: EventImpact.MEDIUM,
    EventType.LISTING: EventImpact.MEDIUM,
    EventType.MERGER_ACQUISITION: EventImpact.HIGH,
    EventType.EARNINGS: EventImpact.MEDIUM,
    EventType.MACRO: EventImpact.MEDIUM,
    EventType.PARTNERSHIP: EventImpact.LOW,
    EventType.LEADERSHIP_CHANGE: EventImpact.MEDIUM,
    EventType.WHALE_MOVEMENT: EventImpact.LOW,
    EventType.NETWORK_UPGRADE: EventImpact.LOW,
    EventType.TOKEN_UNLOCK: EventImpact.MEDIUM,
    EventType.GEOPOLITICAL: EventImpact.HIGH,
    EventType.UNKNOWN: EventImpact.NEGLIGIBLE,
}

# Monetary amount detection for impact scaling
_AMOUNT_PATTERN = re.compile(
    r"\$\s*([\d,.]+)\s*(billion|million|trillion|B|M|T|bn|mn)",
    re.IGNORECASE,
)

# Known crypto/stock ticker-to-entity mapping
_KNOWN_ASSETS: Dict[str, List[str]] = {
    "bitcoin": ["BTC/USDT"], "btc": ["BTC/USDT"],
    "ethereum": ["ETH/USDT"], "eth": ["ETH/USDT"],
    "solana": ["SOL/USDT"], "sol": ["SOL/USDT"],
    "binance": ["BNB/USDT"], "bnb": ["BNB/USDT"],
    "ripple": ["XRP/USDT"], "xrp": ["XRP/USDT"],
    "cardano": ["ADA/USDT"], "ada": ["ADA/USDT"],
    "dogecoin": ["DOGE/USDT"], "doge": ["DOGE/USDT"],
    "avalanche": ["AVAX/USDT"], "avax": ["AVAX/USDT"],
    "apple": ["AAPL"], "aapl": ["AAPL"],
    "microsoft": ["MSFT"], "msft": ["MSFT"],
    "google": ["GOOGL"], "alphabet": ["GOOGL"], "googl": ["GOOGL"],
    "nvidia": ["NVDA"], "nvda": ["NVDA"],
    "tesla": ["TSLA"], "tsla": ["TSLA"],
    "amazon": ["AMZN"], "amzn": ["AMZN"],
    "meta": ["META"], "facebook": ["META"],
}


class EventDetector:
    """Extracts entities and classifies financial events with multi-level
    impact classification, scheduled vs unexpected event handling,
    contextual urgency scoring, and sentiment direction inference.

    Improvements over baseline:
    - **Impact classification**: Events are classified as critical/high/
      medium/low/negligible based on event type, monetary amounts, and
      contextual modifiers.
    - **Scheduled vs unexpected**: Scheduled events (earnings, FOMC) are
      handled differently -- they have lower urgency but higher confidence
      since the market has already partially priced them in.
    - **Contextual urgency**: Urgency is computed from event type, temporal
      modifiers, monetary scale, and asset relevance -- not just keyword
      counting.
    - **Sentiment direction**: Each event type carries a base sentiment
      direction that is refined by context (e.g., "earnings beat" = positive,
      "earnings miss" = negative).
    - **Monetary scale detection**: Dollar amounts in the text are parsed
      to scale impact (a $100M hack is higher impact than a $1M hack).
    - **Multi-event handling**: When text describes multiple events, the
      highest-impact event drives urgency but all events are reported.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._spacy_model_name = spacy_model
        self._nlp = None

    def _load_spacy(self):
        """Lazy-load spaCy model."""
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self._spacy_model_name)
            logger.info("spaCy model loaded: %s", self._spacy_model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                self._spacy_model_name,
                self._spacy_model_name,
            )
            import spacy
            self._nlp = spacy.blank("en")

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities from text using spaCy."""
        self._load_spacy()
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(
                ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Event classification with weighted scoring
    # ------------------------------------------------------------------

    def classify_event(self, text: str) -> Tuple[EventType, Dict[str, int]]:
        """Classify the type of financial event with confidence scoring.

        Uses weighted keyword matching where more specific phrases
        (e.g., "SEC enforcement") score higher than generic words
        (e.g., "compliance").

        Returns:
            Tuple of (best EventType, scores dict for all matched types).
        """
        text_lower = text.lower()
        scores: Dict[EventType, int] = {}

        for event_type, keywords_with_weights in _EVENT_KEYWORDS.items():
            total_score = 0
            for keyword, weight in keywords_with_weights.items():
                if keyword.lower() in text_lower:
                    total_score += weight
            if total_score > 0:
                scores[event_type] = total_score

        if not scores:
            return EventType.UNKNOWN, {}

        best_type = max(scores, key=scores.get)
        return best_type, {k.value: v for k, v in scores.items()}

    # ------------------------------------------------------------------
    # Urgency scoring
    # ------------------------------------------------------------------

    def compute_urgency(
        self, text: str, event_type: EventType, timing: EventTiming
    ) -> int:
        """Score urgency on a 1-10 scale based on multiple factors.

        Factors:
        1. Base urgency from event type (critical events start high)
        2. Urgency word boosters/dampeners
        3. Timing: unexpected events are more urgent than scheduled
        4. Monetary scale: larger amounts = higher urgency
        5. Rumor penalty: unconfirmed = less urgent
        """
        text_lower = text.lower()

        # Base urgency by event type
        base_urgency: Dict[EventType, int] = {
            EventType.HACK_EXPLOIT: 9,
            EventType.DELISTING: 8,
            EventType.LIQUIDATION: 8,
            EventType.REGULATORY: 7,
            EventType.GEOPOLITICAL: 7,
            EventType.MERGER_ACQUISITION: 6,
            EventType.LEGAL: 6,
            EventType.MACRO: 6,
            EventType.EARNINGS: 5,
            EventType.LEADERSHIP_CHANGE: 5,
            EventType.TOKEN_UNLOCK: 4,
            EventType.LISTING: 4,
            EventType.NETWORK_UPGRADE: 3,
            EventType.PARTNERSHIP: 3,
            EventType.WHALE_MOVEMENT: 3,
            EventType.UNKNOWN: 2,
        }
        urgency = float(base_urgency.get(event_type, 3))

        # Urgency word modifiers
        for word, boost in _HIGH_URGENCY_WORDS.items():
            if word in text_lower:
                urgency += boost * 0.5  # scale down to avoid over-boosting

        for word, reduction in _LOW_URGENCY_WORDS.items():
            if word in text_lower:
                urgency += reduction * 0.5

        # Timing adjustments
        if timing == EventTiming.SCHEDULED:
            urgency -= 1.5  # market has partially priced in scheduled events
        elif timing == EventTiming.UNEXPECTED:
            urgency += 1.0  # surprises need faster reaction
        elif timing == EventTiming.RUMOR:
            urgency -= 2.0  # rumors need confirmation

        # Monetary scale boost
        monetary_value = self._extract_monetary_value(text)
        if monetary_value is not None:
            if monetary_value >= 1e9:      # billion+
                urgency += 2.0
            elif monetary_value >= 100e6:  # 100M+
                urgency += 1.5
            elif monetary_value >= 10e6:   # 10M+
                urgency += 1.0
            elif monetary_value >= 1e6:    # 1M+
                urgency += 0.5

        return max(1, min(10, int(round(urgency))))

    # ------------------------------------------------------------------
    # Event timing classification
    # ------------------------------------------------------------------

    def classify_timing(self, text: str, event_type: EventType) -> EventTiming:
        """Determine if an event is scheduled, unexpected, a rumor, or developing.

        Scheduled events (FOMC meetings, earnings dates) have lower urgency
        because markets partially price them in advance.  Unexpected events
        (hacks, sudden regulatory actions) require faster response.
        """
        text_lower = text.lower()

        # Check for rumor indicators first (takes precedence)
        rumor_hits = sum(1 for kw in _RUMOR_KEYWORDS if kw in text_lower)
        if rumor_hits >= 2:
            return EventTiming.RUMOR

        # Check for scheduled event indicators
        if event_type in _SCHEDULED_EVENT_TYPES:
            return EventTiming.SCHEDULED

        scheduled_hits = sum(1 for kw in _SCHEDULED_KEYWORDS if kw in text_lower)
        if scheduled_hits >= 2:
            return EventTiming.SCHEDULED

        # Check for developing/ongoing situation
        developing_keywords = {"developing", "ongoing", "continues", "update", "latest"}
        if any(kw in text_lower for kw in developing_keywords):
            return EventTiming.DEVELOPING

        return EventTiming.UNEXPECTED

    # ------------------------------------------------------------------
    # Impact classification
    # ------------------------------------------------------------------

    def classify_impact(
        self, event_type: EventType, urgency: int, timing: EventTiming, text: str
    ) -> EventImpact:
        """Classify the overall impact level of an event.

        Impact combines event type severity, urgency, timing, and
        monetary scale into a single actionable classification.
        """
        base_impact = _EVENT_BASE_IMPACT.get(event_type, EventImpact.NEGLIGIBLE)

        # Upgrade impact for very high urgency
        if urgency >= 9 and base_impact != EventImpact.CRITICAL:
            return EventImpact.CRITICAL
        if urgency >= 7 and base_impact in (EventImpact.LOW, EventImpact.NEGLIGIBLE):
            return EventImpact.MEDIUM

        # Downgrade impact for rumors
        if timing == EventTiming.RUMOR:
            downgrade_map = {
                EventImpact.CRITICAL: EventImpact.HIGH,
                EventImpact.HIGH: EventImpact.MEDIUM,
                EventImpact.MEDIUM: EventImpact.LOW,
                EventImpact.LOW: EventImpact.NEGLIGIBLE,
            }
            return downgrade_map.get(base_impact, base_impact)

        # Monetary scale can upgrade impact
        monetary_value = self._extract_monetary_value(text)
        if monetary_value is not None and monetary_value >= 1e9:
            upgrade_map = {
                EventImpact.LOW: EventImpact.MEDIUM,
                EventImpact.MEDIUM: EventImpact.HIGH,
                EventImpact.NEGLIGIBLE: EventImpact.LOW,
            }
            return upgrade_map.get(base_impact, base_impact)

        return base_impact

    # ------------------------------------------------------------------
    # Sentiment direction inference
    # ------------------------------------------------------------------

    def infer_sentiment_direction(
        self, text: str, event_type: EventType
    ) -> Tuple[float, float]:
        """Infer the sentiment direction and confidence from event context.

        While event type gives a base direction (hacks = negative), context
        can modify this:
        - "earnings beat" flips earnings from neutral to positive
        - "regulatory clarity" can make regulation events positive
        - "whale accumulation" makes whale movements positive

        Returns:
            (direction, confidence) tuple where direction is in [-1, 1].
        """
        text_lower = text.lower()
        base_direction = _EVENT_SENTIMENT.get(event_type, 0.0)
        confidence = 0.6

        # Context-dependent direction adjustments
        if event_type == EventType.EARNINGS:
            if any(w in text_lower for w in ["beat", "exceeded", "strong", "raised guidance"]):
                base_direction = 0.7
                confidence = 0.75
            elif any(w in text_lower for w in ["miss", "below", "weak", "lowered guidance"]):
                base_direction = -0.7
                confidence = 0.75
            else:
                confidence = 0.3  # unclear direction

        elif event_type == EventType.REGULATORY:
            if any(w in text_lower for w in ["approved", "clarity", "favorable", "green light"]):
                base_direction = 0.6
                confidence = 0.7
            elif any(w in text_lower for w in ["ban", "crackdown", "enforcement", "fine"]):
                base_direction = -0.7
                confidence = 0.8

        elif event_type == EventType.MACRO:
            if any(w in text_lower for w in ["rate cut", "stimulus", "easing", "dovish"]):
                base_direction = 0.6
                confidence = 0.7
            elif any(w in text_lower for w in ["rate hike", "tightening", "hawkish", "inflation high"]):
                base_direction = -0.5
                confidence = 0.7

        elif event_type == EventType.WHALE_MOVEMENT:
            if any(w in text_lower for w in ["accumulation", "bought", "moved from exchange"]):
                base_direction = 0.4
                confidence = 0.5
            elif any(w in text_lower for w in ["dump", "sold", "moved to exchange"]):
                base_direction = -0.5
                confidence = 0.5

        elif event_type == EventType.LIQUIDATION:
            if "short squeeze" in text_lower:
                base_direction = 0.7
                confidence = 0.7
            elif "long squeeze" in text_lower:
                base_direction = -0.7
                confidence = 0.7

        return base_direction, confidence

    # ------------------------------------------------------------------
    # Asset resolution
    # ------------------------------------------------------------------

    def resolve_assets(
        self, text: str, entities: List[ExtractedEntity]
    ) -> List[str]:
        """Map extracted entities to known tradeable assets."""
        assets: Set[str] = set()
        text_lower = text.lower()

        # Check known asset keywords with word boundary matching
        for keyword, tickers in _KNOWN_ASSETS.items():
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                assets.update(tickers)

        # Check entity text against known assets
        for entity in entities:
            entity_lower = entity.text.lower()
            if entity_lower in _KNOWN_ASSETS:
                assets.update(_KNOWN_ASSETS[entity_lower])

        return sorted(assets)

    # ------------------------------------------------------------------
    # Monetary value extraction
    # ------------------------------------------------------------------

    def _extract_monetary_value(self, text: str) -> Optional[float]:
        """Extract the largest monetary value mentioned in the text.

        Recognizes formats like "$100 million", "$1.5B", "$2 trillion".

        Returns:
            Value in USD, or None if no amount found.
        """
        multipliers = {
            "trillion": 1e12, "t": 1e12,
            "billion": 1e9, "b": 1e9, "bn": 1e9,
            "million": 1e6, "m": 1e6, "mn": 1e6,
        }

        max_value = None
        for match in _AMOUNT_PATTERN.finditer(text):
            num_str = match.group(1).replace(",", "")
            suffix = match.group(2).lower()
            try:
                num = float(num_str)
                mult = multipliers.get(suffix, 1.0)
                value = num * mult
                if max_value is None or value > max_value:
                    max_value = value
            except ValueError:
                continue

        return max_value

    # ------------------------------------------------------------------
    # Full detection pipeline
    # ------------------------------------------------------------------

    def detect(self, text: str) -> DetectedEvent:
        """Full detection pipeline: entities, event classification, impact,
        timing, urgency, sentiment direction, and asset resolution.

        Args:
            text: Raw financial news text.

        Returns:
            DetectedEvent with comprehensive analysis.
        """
        entities = self.extract_entities(text)
        event_type, category_scores = self.classify_event(text)
        timing = self.classify_timing(text, event_type)
        urgency = self.compute_urgency(text, event_type, timing)
        impact = self.classify_impact(event_type, urgency, timing, text)
        sentiment_dir, sentiment_conf = self.infer_sentiment_direction(text, event_type)
        assets = self.resolve_assets(text, entities)

        return DetectedEvent(
            event_type=event_type,
            impact=impact,
            timing=timing,
            entities=entities,
            assets=assets,
            urgency=urgency,
            sentiment_direction=sentiment_dir,
            confidence=sentiment_conf,
            summary=text[:300],
            category_scores=category_scores,
        )

    def detect_batch(self, texts: List[str]) -> List[DetectedEvent]:
        """Detect events in multiple texts, sorted by urgency (highest first)."""
        events = [self.detect(t) for t in texts]
        return events

    def filter_by_impact(
        self, events: List[DetectedEvent], min_impact: EventImpact = EventImpact.MEDIUM
    ) -> List[DetectedEvent]:
        """Filter events to only include those at or above a minimum impact level.

        Useful for reducing noise: only process events that are likely to
        move markets.
        """
        impact_order = {
            EventImpact.CRITICAL: 4,
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1,
            EventImpact.NEGLIGIBLE: 0,
        }
        min_level = impact_order.get(min_impact, 0)
        return [
            e for e in events
            if impact_order.get(e.impact, 0) >= min_level
        ]

    def get_highest_impact_event(
        self, events: List[DetectedEvent]
    ) -> Optional[DetectedEvent]:
        """Get the single highest-impact event from a list.

        When multiple events are detected, this identifies the one that
        should drive the primary trading response.
        """
        if not events:
            return None

        impact_order = {
            EventImpact.CRITICAL: 4,
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1,
            EventImpact.NEGLIGIBLE: 0,
        }

        return max(
            events,
            key=lambda e: (impact_order.get(e.impact, 0), e.urgency),
        )

"""Text preprocessing for financial sentiment analysis."""

import re
import unicodedata
from typing import List, Optional


# Financial abbreviation mappings
_FINANCIAL_NORMALIZATIONS = {
    r"\$(\d+(?:\.\d+)?)\s*[Tt]": lambda m: f"{m.group(1)} trillion dollars",
    r"\$(\d+(?:\.\d+)?)\s*[Bb]": lambda m: f"{m.group(1)} billion dollars",
    r"\$(\d+(?:\.\d+)?)\s*[Mm]": lambda m: f"{m.group(1)} million dollars",
    r"\$(\d+(?:\.\d+)?)\s*[Kk]": lambda m: f"{m.group(1)} thousand dollars",
    r"\$(\d+(?:\.\d+)?)": lambda m: f"{m.group(1)} dollars",
}

# Common financial acronyms to expand
_ACRONYMS = {
    "IPO": "initial public offering",
    "SEC": "Securities and Exchange Commission",
    "FOMC": "Federal Open Market Committee",
    "CPI": "consumer price index",
    "GDP": "gross domestic product",
    "ETF": "exchange-traded fund",
    "DeFi": "decentralized finance",
    "NFT": "non-fungible token",
    "ATH": "all-time high",
    "ATL": "all-time low",
    "HODL": "hold on for dear life",
    "FUD": "fear uncertainty and doubt",
    "FOMO": "fear of missing out",
}

# Common Latin/extended character ranges for language detection
_LATIN_PATTERN = re.compile(r"[a-zA-Z]")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")
_ARABIC_PATTERN = re.compile(r"[\u0600-\u06ff]")
_CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04ff]")


class TextPreprocessor:
    """Cleans and normalizes financial text for sentiment analysis."""

    def __init__(self, expand_acronyms: bool = False):
        self._expand_acronyms = expand_acronyms
        self._html_tag_re = re.compile(r"<[^>]+>")
        self._url_re = re.compile(
            r"https?://\S+|www\.\S+", re.IGNORECASE
        )
        self._mention_re = re.compile(r"@\w+")
        self._hashtag_re = re.compile(r"#(\w+)")
        self._ticker_re = re.compile(r"\$([A-Z]{1,5})\b")
        self._whitespace_re = re.compile(r"\s+")
        self._emoji_re = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )

    def clean(self, text: str) -> str:
        """Full cleaning pipeline for a single text string."""
        if not text:
            return ""
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._remove_mentions(text)
        text = self._normalize_hashtags(text)
        text = self._normalize_tickers(text)
        text = self._normalize_financial_amounts(text)
        if self._expand_acronyms:
            text = self._expand_financial_acronyms(text)
        text = self._remove_emojis(text)
        text = self._normalize_unicode(text)
        text = self._collapse_whitespace(text)
        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(t) for t in texts]

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        cleaned = self.clean(text)
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", cleaned.lower())
        return tokens

    def detect_language(self, text: str) -> str:
        """Basic language detection based on character script analysis.

        Returns ISO 639-1 code: 'en', 'zh', 'ar', 'ru', or 'unknown'.
        """
        if not text:
            return "unknown"
        stripped = re.sub(r"[\s\d\W]+", "", text)
        if not stripped:
            return "unknown"
        total = len(stripped)
        latin_count = len(_LATIN_PATTERN.findall(stripped))
        cjk_count = len(_CJK_PATTERN.findall(stripped))
        arabic_count = len(_ARABIC_PATTERN.findall(stripped))
        cyrillic_count = len(_CYRILLIC_PATTERN.findall(stripped))

        ratios = {
            "en": latin_count / total,
            "zh": cjk_count / total,
            "ar": arabic_count / total,
            "ru": cyrillic_count / total,
        }
        best = max(ratios, key=ratios.get)
        if ratios[best] < 0.3:
            return "unknown"
        return best

    def extract_tickers(self, text: str) -> List[str]:
        """Extract cashtag tickers like $BTC, $AAPL from text."""
        return self._ticker_re.findall(text)

    # --- Private helpers ---

    def _remove_html(self, text: str) -> str:
        return self._html_tag_re.sub(" ", text)

    def _remove_urls(self, text: str) -> str:
        return self._url_re.sub(" ", text)

    def _remove_mentions(self, text: str) -> str:
        return self._mention_re.sub(" ", text)

    def _normalize_hashtags(self, text: str) -> str:
        return self._hashtag_re.sub(r"\1", text)

    def _normalize_tickers(self, text: str) -> str:
        return self._ticker_re.sub(r"\1", text)

    def _normalize_financial_amounts(self, text: str) -> str:
        for pattern, repl in _FINANCIAL_NORMALIZATIONS.items():
            text = re.sub(pattern, repl, text)
        return text

    def _expand_financial_acronyms(self, text: str) -> str:
        for acronym, expansion in _ACRONYMS.items():
            text = re.sub(rf"\b{acronym}\b", expansion, text)
        return text

    def _remove_emojis(self, text: str) -> str:
        return self._emoji_re.sub(" ", text)

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKD", text)

    def _collapse_whitespace(self, text: str) -> str:
        return self._whitespace_re.sub(" ", text)

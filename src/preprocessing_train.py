"""
Lightweight preprocessing for TRAINING the Tamil hate-speech model.

Why light?
- Your dataset is already Tamil Unicode.
- Heavy cleaning (stopwords, ASCII filters, etc.) reduced F1.
- Keep only robust, language-agnostic noise removal.

Public API (compatible with your code):
- clean_text(text: str) -> str
- batch_clean_text(texts: Iterable[str]) -> List[str]
"""

from typing import Iterable, List
import re
import unicodedata

# ---------- Core regex ----------
TA_BLOCK = r"\u0B80-\u0BFF"  # Tamil Unicode block

# Remove literal escaped sequences like "\n", "\r", "\t"
ESCAPED_SEQ_RE = re.compile(r"\\[nrt]")

URL_RE       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE   = re.compile(r"@[A-Za-z0-9_]+")

# Zero-width & specials
ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")

# Emojis/pictographs (broad coverage)
EMOJI_RE = re.compile(
    "["                               # common emoji & pictographic ranges
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

# Keep Tamil letters + word chars + whitespace + some basic punctuation; drop the rest
ALLOWED_CHARS_RE = re.compile(rf"[^{TA_BLOCK}\w\s.,!?;:()'\-]")

MULTISPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """NFC normalization so Tamil glyphs are consistent."""
    return unicodedata.normalize("NFC", text or "")


def clean_text(text: str) -> str:
    """
    Minimal, safe cleaning for training:
    - normalize -> strip escaped seq -> strip urls/mentions -> remove zero-width -> remove emoji
    - lowercase -> drop disallowed symbols -> normalize spaces
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    t = normalize_unicode(text)
    t = ESCAPED_SEQ_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = ZERO_WIDTH_RE.sub("", t)
    t = EMOJI_RE.sub(" ", t)
    t = t.lower()
    t = ALLOWED_CHARS_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def batch_clean_text(texts: Iterable[str]) -> List[str]:
    """Apply clean_text over an iterable of strings."""
    return [clean_text(t) for t in texts]


__all__ = ["clean_text", "batch_clean_text", "normalize_unicode"]

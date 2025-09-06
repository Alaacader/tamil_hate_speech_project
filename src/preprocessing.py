# src/preprocessing.py
"""
Preprocessing pipeline for Tamil hate-speech detection.

Adds to a basic cleaner:
- Unicode normalization (NFC)
- URL / @mention stripping
- Zero-width & special char removal
- Emoji & symbol removal
- Escaped sequence cleanup (\n, \r, \t)
- Lowercasing
- Tokenization
- Romanized-Tamil normalization + direct mapping to Tamil (heuristic)
- High-precision abusive romanized detection
- (Optional) drop non-abusive ASCII tokens
- Tamil stopword removal
- Whitespace normalization

Public API (backward compatible):
- clean_text(text: str, *, only_abusive_romanized: bool = False) -> str
- batch_clean_text(texts: Iterable[str]) -> List[str]
"""

from typing import Iterable, List
import re
import unicodedata
DROP_NON_ABUSIVE_ASCII = False
# Remove digits entirely (set to False if you want to keep numbers)
REMOVE_DIGITS = True
# =========================
# Optional import (kept for future sentence ops)
# =========================
try:
    from indicnlp.tokenize import sentence_tokenize  # noqa: F401
except Exception:
    sentence_tokenize = None  # noqa: F401

# =========================
# Unicode ranges & regexes
# =========================
TA_BLOCK = r"\u0B80-\u0BFF"  # Tamil Unicode block
TA_CHAR_RE = re.compile(f"[{TA_BLOCK}]")

# Zero-width & specials
_TAMIL_PUNCT_PATTERN = r"[\u200B\u200C\u200D\uFEFF]"

# URLs, mentions, digits, escaped sequences
URL_RE         = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE     = re.compile(r"@[A-Za-z0-9_]+")
NUMBER_RE      = re.compile(r"\d+")
ESCAPED_SEQ_RE = re.compile(r"\\[nrt]")  # literal "\n" "\r" "\t"

# Emoji / pictographs / symbols
EMOJI_RE = re.compile(
    "["  # common emoji & pictographic ranges
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

# =========================
# Tamil stopwords (extend as needed)
# =========================
TAMIL_STOPWORDS = {
    "அது", "இது", "என்று", "ஒரு", "உள்ள", "ஆம்", "இல்லை",
    "என்", "உன்", "எங்கள்", "உங்கள்",
    "அவன்", "அவள்", "நான்", "நீ", "நாம்", "நீங்கள்", "அவர்கள்",
    "அங்கே", "இங்கே", "மற்றும்", "ஆனால்", "போன்ற",
    "எது", "எப்படி", "எப்போது", "ஏன்", "எங்கே", "மேலும்",
}

# =========================
# Romanized Tamil handling
# =========================

# High-precision abusive romanized patterns (extend gradually)
ABUSIVE_ROMAN_PATTERNS = [
    # Genital/sexual
    r"\bpund[aaiy]+\b",                 # punda, pundai, pundiy
    r"\bthe(v|w)d[iy]y?[ae]+\b",        # thevdiya, thevidiyae
    r"\boomb[uiy]+\b",                  # oombu, oombi
    r"\bsod[aaiy]+\b",                  # soda, soday, sodai

    # Mental/insulting
    r"\blo+osu+\b",                     # loosu, loooosu
    r"\bpaithiy(am|amda|amdi)?\b",      # paithiyam, paithiyamda
    r"\bsappa(y|i)?\b",                 # sappai, sappay
    r"\bkevalam\b",                     # kevalam (disgusting)
    r"\bmuttal+\b",                     # muttal, muttala

    # Animal-related insults
    r"\bnay(y|ee|i)\b",                 # nayi, nayee
    r"\beri+panni+\b",                  # eri panni
    r"\bkedi+\b",                       # kedi

    # Violence / threats
    r"\bserupp(a|adi)\b",               # seruppadi
    r"\bthall[aaiy]+\b",                # thalla, thallay
    r"\badi+pavi+\b",                   # adipavi

    # Gender-based insults (note: context can be sensitive)
    r"\bpomb(a|alai|alaiya|alaiye)\b",  # pombala, pombalai
    r"\baravani+\b",                    # aravani (may be used derogatorily)
    r"\bthirunangai+\b",                # thirunangai (sometimes misused)

    # Misc
    r"\bsaniy(an|am|a)\b",              # saniyan, saniyam
    r"\bseththu?(po|poda)\b",           # seththu po, seththu poda
    r"\bsuththi?(po|poda)\b",           # suththi po, poda
    r"\byarda+\b",                      # yarda (rude vocative)
]
ABUSIVE_ROMAN_RES = [re.compile(p, re.IGNORECASE) for p in ABUSIVE_ROMAN_PATTERNS]

# Romanized abusive → Tamil mapping (extend as you see patterns)
ROMANIZED_TO_TAMIL = {
    "muttal": "முட்டாள்",
    "loosu": "லூசு",
    "punda": "புண்ட",
    "pundai": "புண்டை",
    "thevdiya": "தேவடியா",
    "thevidiyae": "தேவடியே",
    "saniyan": "சணியன்",
    "suththi": "சுத்தி",
    "seththu": "செத்து",
    "sodai": "சோடை",
    "soday": "சோடை",
    "sappai": "சப்பை",
    "paithiyam": "பைத்தியம்",
    "kevalam": "கேவலம்",
    "yarda": "யார்டா",
    "nayee": "நாயீ",
    "nayi": "நாய்",
    "pombalai": "பொம்பளை",
    "aravani": "அரவாணி",
}

# De-noising helpers for romanized tokens
RE_REPEAT = re.compile(r"(.)\1{2,}")   # collapse 3+ repeats -> keep 2
RE_DASHES = re.compile(r"[-_]+")
RE_APOST  = re.compile(r"[’`]+")

def _looks_ascii(token: str) -> bool:
    """Heuristic: whether the token is purely ASCII (potential romanized Tamil)."""
    try:
        token.encode("ascii")
        return True
    except Exception:
        return False

def _normalize_romanized(token: str) -> str:
    """
    Normalize romanized Tamil tokens and map to Tamil if in dictionary.
    - Only runs for ASCII tokens
    - Lowercase
    - Reduce elongations (loooosu -> loousu)
    - Strip stray punctuation
    - Map to Tamil if available in ROMANIZED_TO_TAMIL
    """
    if not _looks_ascii(token):
        return token
    t = token.lower()
    t = RE_REPEAT.sub(r"\1\1", t)
    t = RE_DASHES.sub("-", t)
    t = RE_APOST.sub("'", t)
    t = t.strip("'.-")
    # direct mapping to Tamil if known
    return ROMANIZED_TO_TAMIL.get(t, t)

def _is_romanized_abusive(token: str) -> bool:
    """Return True if token (ASCII or mapped) matches abusive romanized patterns."""
    # If it became Tamil via mapping, keep it (considered abusive term)
    if TA_CHAR_RE.search(token):
        return True
    if not _looks_ascii(token):
        return False
    t = _normalize_romanized(token)
    if len(t) <= 2:
        return False
    return any(rx.search(t) for rx in ABUSIVE_ROMAN_RES)

# =========================
# Core cleaning utilities
# =========================

def normalize_unicode(text: str) -> str:
    """NFC normalization so Tamil glyphs are consistent."""
    return unicodedata.normalize("NFC", text or "")

def _strip_noise(text: str) -> str:
    """
    Remove urls, mentions, zero-width chars, emoji, unwanted symbols; lowercase.
    Also removes literal escaped sequences like '\n', '\r', '\t'.
    """
    t = normalize_unicode(text)
    t = ESCAPED_SEQ_RE.sub(" ", t)   # remove literal \n \r \t
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = re.sub(_TAMIL_PUNCT_PATTERN, "", t)
    t = EMOJI_RE.sub(" ", t)
    t = t.lower()
    # Remove characters outside Tamil block / word chars / whitespace / light punctuation
    t = ALLOWED_CHARS_RE.sub(" ", t)
    if REMOVE_DIGITS:
        t = NUMBER_RE.sub(" ", t)
    # Whitespace normalize
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization (robust for Tamil + code-mix)."""
    if not text:
        return []
    return text.split()

def _remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out Tamil stopwords (extend set as needed)."""
    return [tok for tok in tokens if tok not in TAMIL_STOPWORDS]

def _keep_token(tok: str) -> bool:
    """
    Keep token if:
    - It contains Tamil characters, OR
    - It is an abusive romanized token (ASCII matching patterns OR mapped to Tamil).
    Drop other ASCII noise (helps because model is Tamil-script trained).
    """
    if TA_CHAR_RE.search(tok):
        return True
    return _is_romanized_abusive(tok)

# =========================
# Public API
# =========================

def clean_text(text: str, *, only_abusive_romanized: bool = False) -> str:
    """
    Clean a single string and return a space-joined token string.
    Steps: strip noise -> tokenize -> romanized normalize -> (optional ASCII filtering) ->
           stopword removal -> join.

    If only_abusive_romanized=True, keep ONLY tokens detected as romanized abusive
    (useful for diagnostics).
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    t = _strip_noise(text)
    tokens = _tokenize(t)
    tokens = [_normalize_romanized(tok) for tok in tokens]

    if only_abusive_romanized:
        tokens = [tok for tok in tokens if _is_romanized_abusive(tok)]
    else:
        if DROP_NON_ABUSIVE_ASCII:
            tokens = [tok for tok in tokens if _keep_token(tok)]
        tokens = _remove_stopwords(tokens)

    return " ".join(tokens)

def batch_clean_text(texts: Iterable[str]) -> List[str]:
    """Apply clean_text over an iterable of strings."""
    return [clean_text(t) for t in texts]

__all__ = [
    "clean_text",
    "batch_clean_text",
    "normalize_unicode",
]

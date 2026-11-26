from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")

# German orthography: fold umlauts and ß for BM25 and quote matching. bge-m3
# still sees the original text, so this only makes lexical matching robust.
_DE_FOLD = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
    }
)


def normalize_text(text: str) -> str:
    """Lowercase + NFKC + whitespace collapse, used for stable ids and matching."""
    value = unicodedata.normalize("NFKC", text or "")
    return _WS_RE.sub(" ", value).strip().lower()


def fold_german(text: str) -> str:
    return (text or "").translate(_DE_FOLD)


_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                         "—": "-", "―": "-", "−": "-"})


def contains_quote(haystack: str, quote: str, min_len: int = 12) -> bool:
    """True if a verbatim quote occurs in the text, tolerating PDF artefacts.

    PDF extractors differ on dashes, spacing and superscripts ("% vs %",
    "10⁶ vs 106"). Exact containment is tried first; a whitespace-free pass
    catches the rest while the minimum length keeps common fragments from
    resolving to dozens of chunks.
    """
    hay = normalize_text(haystack).translate(_DASHES)
    needle = normalize_text(quote).translate(_DASHES)
    if len(needle) < min_len:
        return False
    if needle in hay:
        return True
    hay_ns = hay.replace("-", "").replace(" ", "")
    needle_ns = needle.replace("-", "").replace(" ", "")
    return len(needle_ns) >= min_len and needle_ns in hay_ns

"""
text_quality.py — Decide how to extract text from each PDF page.

For every page we measure a few cheap "quality" signals on its embedded text
layer, then route that page to the right extraction method:

    "direct"  → trust the embedded text (fast, clean, no OCR)
    "ocr"     → the text layer is missing or gibberish; rasterize + OCR instead

Why decide PER PAGE (not per file)? A single PDF can be mixed — a scanned cover
before a digital body, or an English preface before Hindi chapters. A per-page
verdict is more robust than one verdict for the whole document.

The signals (see page_metrics):
    deva%     — % of characters that are Devanagari (real Hindi Unicode)
    latin%    — % that are plain ASCII letters (English)
    junk%     — % that are odd symbols (© ° ¢ …), the fingerprint of legacy fonts
    engword%  — % of English tokens that are REAL dictionary words (checks meaning)
"""

from __future__ import annotations  # 3.9-safe: makes annotations lazy strings

import re

# ── Decision thresholds (tunable) ────────────────────────────────────
MIN_DEVA_PCT    = 30   # ≥ this % Devanagari → accept as real Hindi
MAX_JUNK_HINDI  = 5    # Hindi pages may carry a few extra symbols
MIN_LATIN_PCT   = 40   # ≥ this % Latin letters → looks English-ish
MAX_JUNK_ENG    = 8    # English pages should be almost symbol-free (tuned: clean≤4.9, gibberish=13.7)
MIN_ENGWORD_PCT = 35   # ≥ this % of tokens must be real words (tuned: gibberish=16.6, clean≥53)

# Punctuation we must NOT count as "junk"
COMMON_PUNCT = set(".,;:!?()[]{}\"'`-–—/%&@#*+=<>|\\ ।॥₹’‘“”…•·")

# Devanagari Unicode block: U+0900 ('ऀ') … U+097F ('ॿ')
_DEVA_LO, _DEVA_HI = "ऀ", "ॿ"


def _load_english_words() -> set:
    """Load real English words for the 'real-word' check.
    Returns an empty set if no system wordlist exists (e.g. some Linux images)."""
    for path in ("/usr/share/dict/words", "/usr/dict/words"):
        try:
            with open(path) as f:
                return {w.strip().lower() for w in f if len(w.strip()) > 1}
        except FileNotFoundError:
            continue
    return set()


WORDS = _load_english_words()


def page_metrics(text: str) -> dict | None:
    """Return quality percentages for one page, or None if the page has no text.

    Keys (all 0–100): deva, latin, junk, engword
    """
    non_space = [c for c in text if not c.isspace()]
    total = len(non_space)
    if total == 0:
        return None

    deva = sum(1 for c in non_space if _DEVA_LO <= c <= _DEVA_HI)
    latin = sum(1 for c in non_space if c.isascii() and c.isalpha())
    junk = sum(
        1 for c in non_space
        if not c.isalnum()
        and not (_DEVA_LO <= c <= _DEVA_HI)
        and c not in COMMON_PUNCT
    )

    tokens = re.findall(r"[A-Za-z]{2,}", text)
    real = sum(1 for t in tokens if t.lower() in WORDS)
    engword = (real / len(tokens) * 100) if tokens else 0.0

    return {
        "deva": 100 * deva / total,
        "latin": 100 * latin / total,
        "junk": 100 * junk / total,
        "engword": engword,
    }


def choose_method(text: str) -> str:
    """Decide how to extract ONE page: 'direct' (trust text layer) or 'ocr'."""
    m = page_metrics(text)
    if m is None:
        return "ocr"  # no text layer at all → scanned image page

    # real Hindi (Unicode Devanagari, few stray symbols)
    if m["deva"] > MIN_DEVA_PCT and m["junk"] < MAX_JUNK_HINDI:
        return "direct"

    # real English (Latin letters, symbol-free, and actually real words)
    if (
        m["latin"] > MIN_LATIN_PCT
        and m["junk"] < MAX_JUNK_ENG
        and m["engword"] > MIN_ENGWORD_PCT
    ):
        return "direct"

    # otherwise: legacy-font gibberish (looks like letters, isn't real words)
    return "ocr"

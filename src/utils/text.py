from __future__ import annotations

import re

_BOILERPLATE_PATTERNS = [
    r"\bi am writing to file a complaint\b",
    r"\bthis is a complaint\b",
    r"\bto whom it may concern\b",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_narrative(text: str) -> str:
    """Clean complaint narrative text for embedding.

    Minimal, conservative cleaning:
    - lowercasing
    - remove boilerplate phrases
    - keep punctuation lightly (remove repeated non-word noise)
    - normalize whitespace
    """
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.lower()

    for pat in _BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text)

    # Remove long runs of non-alphanumeric symbols (but keep sentence punctuation)
    text = re.sub(r"[^\w\s\.,;:!?\-\(\)\"']+", " ", text)

    return normalize_whitespace(text)

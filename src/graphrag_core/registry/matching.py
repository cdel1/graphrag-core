"""Fuzzy matching utilities for entity deduplication."""

from __future__ import annotations

import re
import string
from difflib import SequenceMatcher

_PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, sort tokens."""
    name = _PUNCTUATION_RE.sub("", name.lower())
    tokens = sorted(name.split())
    return " ".join(tokens)


def fuzzy_score(a: str, b: str) -> float:
    """Token-normalized SequenceMatcher ratio."""
    norm_a = normalize_name(a)
    norm_b = normalize_name(b)
    return SequenceMatcher(None, norm_a, norm_b).ratio()

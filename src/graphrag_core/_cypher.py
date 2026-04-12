"""Shared Cypher safety utilities."""

from __future__ import annotations

import re

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_DEPTH = 10


def validate_identifier(value: str, kind: str) -> str:
    """Reject identifiers that could cause Cypher injection."""
    if not SAFE_IDENTIFIER.match(value):
        raise ValueError(f"Invalid {kind}: {value!r}")
    return value

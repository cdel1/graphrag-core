"""Typed exception hierarchy (ADR-0006b Rule 4 — seeded by ADR-0033).

Only the GraphStore family base exists today; further families
(IngestionError, ExtractionError, ...) are added when a decision
forces them, not preemptively.
"""

from __future__ import annotations


class GraphStoreError(Exception):
    """A GraphStore operation failed. Raised by flush() when durability
    cannot be guaranteed; implementations wrap backend-native errors."""

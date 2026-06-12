"""Typed exception hierarchy (ADR-0006b Rule 4 — seeded by ADR-0033).

Only the GraphStore family base exists today; further families
(IngestionError, ExtractionError, ...) are added when a decision
forces them, not preemptively.
"""

from __future__ import annotations


class GraphStoreError(Exception):
    """A GraphStore operation failed. Raised by flush() when durability
    cannot be guaranteed; implementations wrap backend-native errors."""


class MissingEndpointError(GraphStoreError):
    """merge_relationship was called with a source or target node that does
    not exist in the store (strict merge — ADR-0034)."""

    def __init__(self, source_id: str, target_id: str) -> None:
        super().__init__(
            f"merge_relationship: missing endpoint(s) — "
            f"source={source_id!r}, target={target_id!r}"
        )
        self.source_id = source_id
        self.target_id = target_id

"""Produced-graph edge label → DocRED relation type mapping.

Conservative start: identity mapping (edge type IS the DocRED relation).
Extend as the framework's default extraction starts emitting these relations
with non-DocRED-shaped names.
"""

from __future__ import annotations


def graph_edge_to_docred_relation(edge_type: str) -> str | None:
    return edge_type

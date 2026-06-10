"""Generic Tier-1 deterministic invariant checks.

Reference: docs/research/2026-06-05-evals-doctrine.md §3 Tier 1.
Reference: docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.3.
"""

from __future__ import annotations

from graphrag_core.interfaces import GraphStore

from graphrag_core.eval.models import Violation


_INTELLIGENCE_LABELS = {"Topic", "Risk", "Recommendation"}


class ProvenanceCompletenessCheck:
    """Every :Claim must carry SOURCED_FROM."""

    async def check(self, graph_store: GraphStore, manifest) -> list[Violation]:
        all_rels = await graph_store.list_relationships()
        sourced_from_sources = {
            rel.source_id for rel in all_rels if rel.type == "SOURCED_FROM"
        }
        claims = [n for n in await graph_store.list_nodes() if n.label == "Claim"]
        violations: list[Violation] = []
        for claim in claims:
            if claim.id not in sourced_from_sources:
                violations.append(Violation(
                    check="provenance_completeness",
                    message=f"Claim {claim.id} missing SOURCED_FROM edge",
                ))
        return violations


class NoOrphanIntelligenceCheck:
    """No Topic/Risk/Recommendation without a path to a Claim."""

    async def check(self, graph_store: GraphStore, manifest) -> list[Violation]:
        all_nodes = await graph_store.list_nodes()
        node_by_id = {n.id: n for n in all_nodes}
        all_rels = await graph_store.list_relationships()
        outgoing: dict[str, list[str]] = {}
        for rel in all_rels:
            outgoing.setdefault(rel.source_id, []).append(rel.target_id)

        violations: list[Violation] = []
        for node in all_nodes:
            if node.label not in _INTELLIGENCE_LABELS:
                continue
            related_ids = outgoing.get(node.id, [])
            grounding_claims = [
                rid for rid in related_ids
                if node_by_id.get(rid) and node_by_id[rid].label == "Claim"
            ]
            if not grounding_claims:
                violations.append(Violation(
                    check="no_orphan_intelligence",
                    message=f"{node.label} {node.id} has no grounding Claim",
                ))
        return violations


class SchemaConformanceCheck:
    """Every node label is in the allowed set."""

    def __init__(self, allowed_labels: set[str]) -> None:
        self.allowed_labels = allowed_labels

    async def check(self, graph_store: GraphStore, manifest) -> list[Violation]:
        nodes = await graph_store.list_nodes()
        violations: list[Violation] = []
        for node in nodes:
            if node.label not in self.allowed_labels:
                violations.append(Violation(
                    check="schema_conformance",
                    message=f"node {node.id} has unknown label {node.label!r}",
                ))
        return violations

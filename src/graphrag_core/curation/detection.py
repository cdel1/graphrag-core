"""BB5: Deterministic detection layer for graph quality issues."""

from __future__ import annotations

import uuid
from collections import defaultdict

from graphrag_core.interfaces import EntityRegistry, GraphStore
from graphrag_core.models import CurationIssue, GraphNode, OntologySchema
from graphrag_core.registry.matching import fuzzy_score

_PAIRWISE_CAP = 1000
_FUZZY_THRESHOLD = 0.7


class DeterministicDetectionLayer:
    """Finds duplicates, orphans, and schema violations without LLM calls."""

    def __init__(self, entity_registry: EntityRegistry | None = None) -> None:
        self._registry = entity_registry

    async def detect(
        self, graph_store: GraphStore, schema: OntologySchema
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []

        nodes = await graph_store.list_nodes()

        issues.extend(await self._detect_duplicates(nodes))
        issues.extend(await self._detect_orphans(nodes, graph_store))
        issues.extend(await self._detect_schema_violations(graph_store, schema))

        return issues

    async def _detect_duplicates(self, nodes: list[GraphNode]) -> list[CurationIssue]:
        issues: list[CurationIssue] = []

        groups: dict[str, list[GraphNode]] = defaultdict(list)
        for node in nodes:
            groups[node.label].append(node)

        for label, group in groups.items():
            if self._registry is not None:
                issues.extend(await self._detect_duplicates_with_registry(group))
            else:
                if len(group) > _PAIRWISE_CAP:
                    issues.append(CurationIssue(
                        id=str(uuid.uuid4()),
                        issue_type="skipped_detection",
                        severity="warning",
                        affected_nodes=[],
                        suggested_action=f"Register entities in EntityRegistry for label '{label}' ({len(group)} nodes exceeds pairwise cap of {_PAIRWISE_CAP})",
                        auto_fixable=False,
                        source_layer="deterministic",
                    ))
                else:
                    issues.extend(self._detect_duplicates_pairwise(group))

        return issues

    async def _detect_duplicates_with_registry(
        self, nodes: list[GraphNode]
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []
        entity_to_nodes: dict[str, list[str]] = defaultdict(list)

        for node in nodes:
            name = node.properties.get("name", "")
            if not name:
                continue
            matches = await self._registry.lookup(name, node.label, match_strategy="fuzzy")
            if matches:
                entity_to_nodes[matches[0].entity_id].append(node.id)

        for entity_id, node_ids in entity_to_nodes.items():
            if len(node_ids) > 1:
                issues.append(CurationIssue(
                    id=str(uuid.uuid4()),
                    issue_type="duplicate",
                    severity="warning",
                    affected_nodes=node_ids,
                    suggested_action=f"Merge nodes {node_ids} — they match registry entity '{entity_id}'",
                    auto_fixable=False,
                    source_layer="deterministic",
                ))

        return issues

    def _detect_duplicates_pairwise(
        self, nodes: list[GraphNode]
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, a in enumerate(nodes):
            name_a = a.properties.get("name", "")
            if not name_a:
                continue
            for b in nodes[i + 1:]:
                name_b = b.properties.get("name", "")
                if not name_b:
                    continue
                pair = (min(a.id, b.id), max(a.id, b.id))
                if pair in seen_pairs:
                    continue
                score = fuzzy_score(name_a, name_b)
                if score >= _FUZZY_THRESHOLD:
                    seen_pairs.add(pair)
                    issues.append(CurationIssue(
                        id=str(uuid.uuid4()),
                        issue_type="duplicate",
                        severity="warning",
                        affected_nodes=[a.id, b.id],
                        suggested_action=f"Merge '{name_a}' and '{name_b}' (similarity: {score:.2f})",
                        auto_fixable=False,
                        source_layer="deterministic",
                    ))

        return issues

    async def _detect_orphans(
        self, nodes: list[GraphNode], graph_store: GraphStore
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []

        for node in nodes:
            related = await graph_store.get_related(node.id)
            if not related:
                issues.append(CurationIssue(
                    id=str(uuid.uuid4()),
                    issue_type="orphan",
                    severity="info",
                    affected_nodes=[node.id],
                    suggested_action=f"Node '{node.id}' ({node.label}) has no relationships",
                    auto_fixable=False,
                    source_layer="deterministic",
                ))

        return issues

    async def _detect_schema_violations(
        self, graph_store: GraphStore, schema: OntologySchema
    ) -> list[CurationIssue]:
        await graph_store.apply_schema(schema)
        violations = await graph_store.validate_schema()

        return [
            CurationIssue(
                id=str(uuid.uuid4()),
                issue_type="schema_violation",
                severity="error",
                affected_nodes=[v.node_id],
                suggested_action=v.message,
                auto_fixable=False,
                source_layer="deterministic",
            )
            for v in violations
        ]

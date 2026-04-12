"""In-memory GraphStore implementation for testing."""

from __future__ import annotations

from graphrag_core.models import (
    AuditTrail,
    GraphNode,
    GraphRelationship,
    OntologySchema,
    ProvenanceStep,
    SchemaViolation,
)


class InMemoryGraphStore:
    """Dict-based GraphStore for unit tests and lightweight usage."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._relationships: list[GraphRelationship] = []
        self._provenance: dict[str, list[str]] = {}  # node_id -> [chunk_ids]
        self._schema: OntologySchema | None = None

    async def merge_node(self, node: GraphNode, import_run_id: str) -> str:
        existing = self._nodes.get(node.id)
        if existing:
            merged_props = {**existing.properties, **node.properties}
            self._nodes[node.id] = GraphNode(id=node.id, label=node.label, properties=merged_props)
        else:
            self._nodes[node.id] = node
        return node.id

    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str:
        for i, existing in enumerate(self._relationships):
            if (
                existing.source_id == rel.source_id
                and existing.target_id == rel.target_id
                and existing.type == rel.type
            ):
                self._relationships[i] = rel
                return f"{rel.source_id}-{rel.type}-{rel.target_id}"
        self._relationships.append(rel)
        return f"{rel.source_id}-{rel.type}-{rel.target_id}"

    async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None:
        if node_id not in self._provenance:
            self._provenance[node_id] = []
        if chunk_id not in self._provenance[node_id]:
            self._provenance[node_id].append(chunk_id)

    async def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    async def get_audit_trail(self, node_id: str) -> AuditTrail:
        chain: list[ProvenanceStep] = []
        node = self._nodes.get(node_id)
        if node:
            chain.append(ProvenanceStep(level="node", id=node_id, metadata={"label": node.label}))
        for chunk_id in self._provenance.get(node_id, []):
            chain.append(ProvenanceStep(level="chunk", id=chunk_id, metadata={}))
        return AuditTrail(node_id=node_id, provenance_chain=chain)

    async def get_related(
        self, node_id: str, rel_type: str | None = None, depth: int = 1
    ) -> list[GraphNode]:
        if depth < 1:
            return []

        related_ids: set[str] = set()
        for rel in self._relationships:
            if rel.source_id == node_id and (rel_type is None or rel.type == rel_type):
                related_ids.add(rel.target_id)
            if rel.target_id == node_id and (rel_type is None or rel.type == rel_type):
                related_ids.add(rel.source_id)

        result = [self._nodes[nid] for nid in related_ids if nid in self._nodes]

        if depth > 1:
            for nid in list(related_ids):
                deeper = await self.get_related(nid, rel_type, depth - 1)
                for node in deeper:
                    if node.id != node_id and node.id not in related_ids:
                        related_ids.add(node.id)
                        result.append(node)

        return result

    async def apply_schema(self, schema: OntologySchema) -> None:
        self._schema = schema

    async def validate_schema(self) -> list[SchemaViolation]:
        if self._schema is None:
            return []

        violations: list[SchemaViolation] = []
        label_to_required: dict[str, list[str]] = {}
        for nt in self._schema.node_types:
            label_to_required[nt.label] = nt.required_properties

        for node in self._nodes.values():
            required = label_to_required.get(node.label, [])
            for prop_name in required:
                if prop_name not in node.properties:
                    violations.append(
                        SchemaViolation(
                            node_id=node.id,
                            violation_type="missing_property",
                            message=f"Required property '{prop_name}' is missing on {node.label} '{node.id}'",
                        )
                    )

        return violations

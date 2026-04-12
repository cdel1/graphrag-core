"""Tests for InMemoryGraphStore."""

from __future__ import annotations

import pytest

from graphrag_core.models import (
    GraphNode,
    GraphRelationship,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
)


class TestInMemoryGraphStoreWrite:
    @pytest.mark.asyncio
    async def test_merge_node_stores_and_returns_id(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        node = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        result_id = await store.merge_node(node, import_run_id="run-1")

        assert result_id == "n1"

    @pytest.mark.asyncio
    async def test_merge_node_updates_existing(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        node_v1 = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        node_v2 = GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"})

        await store.merge_node(node_v1, import_run_id="run-1")
        await store.merge_node(node_v2, import_run_id="run-2")

        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.properties["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_merge_relationship_stores_and_returns_id(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")

        rel = GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT")
        result_id = await store.merge_relationship(rel, import_run_id="run-1")

        assert result_id is not None
        assert len(result_id) > 0

    @pytest.mark.asyncio
    async def test_record_provenance(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")

        trail = await store.get_audit_trail("n1")
        assert trail.node_id == "n1"
        assert len(trail.provenance_chain) >= 1
        chunk_steps = [s for s in trail.provenance_chain if s.level == "chunk"]
        assert len(chunk_steps) == 1
        assert chunk_steps[0].id == "chunk-0"


class TestInMemoryGraphStoreRead:
    @pytest.mark.asyncio
    async def test_get_node_returns_none_for_missing(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        result = await store.get_node("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_related_returns_neighbors(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )

        related = await store.get_related("n1")
        assert len(related) == 1
        assert related[0].id == "n2"

    @pytest.mark.asyncio
    async def test_get_related_filters_by_rel_type(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_node(GraphNode(id="n3", label="Person", properties={"name": "Bob"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n3", type="KNOWS"), "run-1"
        )

        related = await store.get_related("n1", rel_type="WORKS_AT")
        assert len(related) == 1
        assert related[0].id == "n2"

    @pytest.mark.asyncio
    async def test_get_related_respects_depth(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="a", label="X", properties={}), "run-1")
        await store.merge_node(GraphNode(id="b", label="X", properties={}), "run-1")
        await store.merge_node(GraphNode(id="c", label="X", properties={}), "run-1")
        await store.merge_relationship(GraphRelationship(source_id="a", target_id="b", type="LINK"), "run-1")
        await store.merge_relationship(GraphRelationship(source_id="b", target_id="c", type="LINK"), "run-1")

        depth_1 = await store.get_related("a", depth=1)
        assert {n.id for n in depth_1} == {"b"}

        depth_2 = await store.get_related("a", depth=2)
        assert {n.id for n in depth_2} == {"b", "c"}

    @pytest.mark.asyncio
    async def test_audit_trail_empty_provenance(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        trail = await store.get_audit_trail("n1")
        assert trail.node_id == "n1"
        assert len(trail.provenance_chain) == 1
        assert trail.provenance_chain[0].level == "node"


class TestInMemoryGraphStoreSchema:
    @pytest.mark.asyncio
    async def test_validate_finds_missing_required_property(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Company",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                    required_properties=["name"],
                ),
            ],
            relationship_types=[],
        )
        await store.apply_schema(schema)
        await store.merge_node(GraphNode(id="n1", label="Company", properties={}), "run-1")

        violations = await store.validate_schema()
        assert len(violations) == 1
        assert violations[0].node_id == "n1"
        assert violations[0].violation_type == "missing_property"

    @pytest.mark.asyncio
    async def test_validate_passes_for_valid_nodes(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Company",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                    required_properties=["name"],
                ),
            ],
            relationship_types=[],
        )
        await store.apply_schema(schema)
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        violations = await store.validate_schema()
        assert violations == []

    @pytest.mark.asyncio
    async def test_validate_without_schema_returns_empty(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        violations = await store.validate_schema()
        assert violations == []

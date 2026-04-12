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

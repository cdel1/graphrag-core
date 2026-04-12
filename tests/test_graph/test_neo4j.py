"""Integration tests for Neo4jGraphStore. Requires running Neo4j."""

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

pytestmark = pytest.mark.integration


@pytest.fixture
async def store():
    from graphrag_core.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(database="test")
    # Wipe the test database before each test
    async with store._driver.session(database="test") as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield store
    await store.close()


class TestNeo4jMergeNode:
    @pytest.mark.asyncio
    async def test_merge_and_retrieve_node(self, store):
        node = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        result_id = await store.merge_node(node, import_run_id="run-1")
        assert result_id == "n1"

        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.label == "Company"
        assert retrieved.properties["name"] == "Acme"

    @pytest.mark.asyncio
    async def test_merge_node_updates_properties(self, store):
        node_v1 = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        node_v2 = GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"})

        await store.merge_node(node_v1, import_run_id="run-1")
        await store.merge_node(node_v2, import_run_id="run-2")

        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.properties["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_node_returns_none_for_missing(self, store):
        result = await store.get_node("nonexistent")
        assert result is None


class TestNeo4jMergeRelationship:
    @pytest.mark.asyncio
    async def test_merge_relationship(self, store):
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")

        rel = GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT")
        result_id = await store.merge_relationship(rel, import_run_id="run-1")
        assert result_id == "n1-WORKS_AT-n2"


class TestNeo4jProvenance:
    @pytest.mark.asyncio
    async def test_record_and_retrieve_provenance(self, store):
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")
        await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")

        trail = await store.get_audit_trail("n1")
        assert trail.node_id == "n1"
        assert len(trail.provenance_chain) >= 2
        levels = [s.level for s in trail.provenance_chain]
        assert "node" in levels
        assert "chunk" in levels

    @pytest.mark.asyncio
    async def test_multiple_provenance_links(self, store):
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")
        await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")
        await store.record_provenance(node_id="n1", chunk_id="chunk-1", import_run_id="run-1")

        trail = await store.get_audit_trail("n1")
        chunk_steps = [s for s in trail.provenance_chain if s.level == "chunk"]
        assert len(chunk_steps) == 2
        chunk_ids = {s.id for s in chunk_steps}
        assert chunk_ids == {"chunk-0", "chunk-1"}


class TestNeo4jGetRelated:
    @pytest.mark.asyncio
    async def test_get_related_returns_neighbors(self, store):
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )

        related = await store.get_related("n1")
        assert len(related) == 1
        assert related[0].id == "n2"

    @pytest.mark.asyncio
    async def test_get_related_filters_by_type(self, store):
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


class TestNeo4jSchema:
    @pytest.mark.asyncio
    async def test_apply_schema_creates_constraints(self, store):
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
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Updated"}), "run-2")

        node = await store.get_node("n1")
        assert node is not None
        assert node.properties["name"] == "Updated"


class TestNeo4jProtocol:
    def test_satisfies_graph_store_protocol(self):
        from graphrag_core.graph.neo4j import Neo4jGraphStore
        from graphrag_core.interfaces import GraphStore

        store = Neo4jGraphStore()
        assert isinstance(store, GraphStore)

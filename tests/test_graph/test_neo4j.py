"""Integration tests for Neo4jGraphStore. Requires running Neo4j."""

from __future__ import annotations

import os

import pytest

NEO4J_TEST_DB = os.environ.get("NEO4J_TEST_DATABASE", "neo4j")

from graphrag_core.exceptions import MissingEndpointError
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

    store = Neo4jGraphStore(database=NEO4J_TEST_DB)
    # Wipe the test database before each test
    async with store._driver.session(database=NEO4J_TEST_DB) as session:
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

        trail = await store.get_provenance("n1")
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

        trail = await store.get_provenance("n1")
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


class TestNeo4jStrictMerge:
    @pytest.mark.asyncio
    async def test_merge_relationship_missing_endpoint_raises(self, store):
        rel = GraphRelationship(source_id="ghost-a", target_id="ghost-b", type="REL", properties={})
        with pytest.raises(MissingEndpointError):
            await store.merge_relationship(rel, import_run_id="run-1")

    @pytest.mark.asyncio
    async def test_merge_relationship_with_endpoints_succeeds(self, store):
        await store.merge_node(GraphNode(id="a", label="Entity", properties={}), "run-1")
        await store.merge_node(GraphNode(id="b", label="Entity", properties={}), "run-1")
        rel = GraphRelationship(source_id="a", target_id="b", type="REL", properties={})
        assert await store.merge_relationship(rel, import_run_id="run-1") == "a-REL-b"


class TestNeo4jClear:
    @pytest.mark.asyncio
    async def test_clear_removes_everything(self, store):
        await store.merge_node(GraphNode(id="a", label="Entity", properties={}), "run-1")
        await store.merge_node(GraphNode(id="b", label="Entity", properties={}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="a", target_id="b", type="REL", properties={}), "run-1"
        )
        await store.clear()
        assert await store.list_nodes() == []
        assert await store.count_relationships() == 0


class TestNeo4jProvenanceEdgeDirection:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provenance_edge_is_node_to_chunk_from_chunk(self, store):
        await store.merge_node(GraphNode(id="claim-1", label="Claim", properties={}), "run-1")
        await store.record_provenance("claim-1", "chunk-1", "run-1")
        trail = await store.get_provenance("claim-1")
        chunk_steps = [s for s in trail.provenance_chain if s.level == "chunk"]
        assert [s.id for s in chunk_steps] == ["chunk-1"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_from_chunk_edge_direction(self, store):
        from neo4j import AsyncDriver

        await store.merge_node(GraphNode(id="claim-1", label="Claim", properties={}), "run-1")
        await store.record_provenance("claim-1", "chunk-1", "run-1")
        async with store._driver.session(database=store._database) as session:
            result = await session.run(
                "MATCH (n {id:'claim-1'})-[:FROM_CHUNK]->(c:Chunk {id:'chunk-1'}) RETURN count(*) AS c"
            )
            rec = await result.single()
        assert rec["c"] == 1


class TestNeo4jNestedPropertyRoundTrip:
    """Verify that nested-map properties survive a write → read round-trip."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_nested_dict_property_round_trips(self, store):
        """A node with a dict-valued property reads back equal to what was written."""
        props = {"meta": {"source": "doc-1", "confidence": 0.9}}
        node = GraphNode(id="rt-1", label="Entity", properties=props)
        await store.merge_node(node, import_run_id="run-1")

        retrieved = await store.get_node("rt-1")
        assert retrieved is not None
        assert retrieved.properties == props

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_attestation_shape_round_trips(self, store):
        """The per-property attestation shape (list-of-dicts) round-trips exactly."""
        props = {"name": [{"value": "Zone A", "run_id": "run-1"}]}
        node = GraphNode(id="rt-2", label="Entity", properties=props)
        await store.merge_node(node, import_run_id="run-1")

        retrieved = await store.get_node("rt-2")
        assert retrieved is not None
        assert retrieved.properties == props

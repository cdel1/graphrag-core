"""Integration tests for Neo4jHybridSearch. Requires running Neo4j."""

from __future__ import annotations

import os

import pytest

from graphrag_core.models import GraphNode

pytestmark = pytest.mark.integration

NEO4J_TEST_DB = os.environ.get("NEO4J_TEST_DATABASE", "neo4j")


@pytest.fixture
async def search_engine():
    from graphrag_core.graph.neo4j import Neo4jGraphStore
    from graphrag_core.search.neo4j import Neo4jHybridSearch

    store = Neo4jGraphStore(database=NEO4J_TEST_DB)
    engine = Neo4jHybridSearch(
        database=NEO4J_TEST_DB,
        vector_index_name="test_vector_idx",
        fulltext_index_name="test_fulltext_idx",
    )

    # Wipe database
    async with store._driver.session(database=NEO4J_TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")
        try:
            await session.run("DROP INDEX test_vector_idx IF EXISTS")
        except Exception:
            pass
        try:
            await session.run("DROP INDEX test_fulltext_idx IF EXISTS")
        except Exception:
            pass

    # Create test nodes with embeddings
    await store.merge_node(
        GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1"
    )
    await store.merge_node(
        GraphNode(id="n2", label="Company", properties={"name": "Globex Industries"}), "run-1"
    )
    await store.merge_node(
        GraphNode(id="n3", label="Person", properties={"name": "Alice Smith"}), "run-1"
    )

    # Add embeddings directly to nodes
    async with store._driver.session(database=NEO4J_TEST_DB) as session:
        await session.run(
            "MATCH (n {id: 'n1'}) SET n.embedding = $emb",
            emb=[1.0, 0.0, 0.0],
        )
        await session.run(
            "MATCH (n {id: 'n2'}) SET n.embedding = $emb",
            emb=[0.0, 1.0, 0.0],
        )
        await session.run(
            "MATCH (n {id: 'n3'}) SET n.embedding = $emb",
            emb=[0.7, 0.7, 0.0],
        )

    # Add a relationship for graph search
    from graphrag_core.models import GraphRelationship
    await store.merge_relationship(
        GraphRelationship(source_id="n3", target_id="n1", type="WORKS_AT"), "run-1"
    )

    # Create indexes
    await engine.ensure_indexes(
        vector_dimensions=3,
        vector_node_label="Company",
        vector_property="embedding",
        fulltext_node_labels=["Company", "Person"],
        fulltext_properties=["name"],
    )

    # Wait for indexes to come online
    import asyncio
    await asyncio.sleep(1)

    await store.close()
    yield engine
    await engine.close()


class TestNeo4jVectorSearch:
    @pytest.mark.asyncio
    async def test_returns_nearest_neighbors(self, search_engine):
        results = await search_engine.vector_search(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "vector"
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_respects_top_k(self, search_engine):
        results = await search_engine.vector_search(
            query_embedding=[1.0, 0.0, 0.0], top_k=1
        )
        assert len(results) == 1


class TestNeo4jFulltextSearch:
    @pytest.mark.asyncio
    async def test_matches_on_name(self, search_engine):
        results = await search_engine.fulltext_search(query="Acme", top_k=10)

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "fulltext"

    @pytest.mark.asyncio
    async def test_filters_by_node_types(self, search_engine):
        results = await search_engine.fulltext_search(
            query="A", node_types=["Person"], top_k=10
        )

        assert all(r.label == "Person" for r in results)


class TestNeo4jGraphSearch:
    @pytest.mark.asyncio
    async def test_traverses_from_start_node(self, search_engine):
        results = await search_engine.graph_search(
            start_node_id="n3", pattern="WORKS_AT", depth=1
        )

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "graph"
        assert results[0].score == 1.0


class TestNeo4jHybridSearch:
    @pytest.mark.asyncio
    async def test_fuses_vector_and_fulltext(self, search_engine):
        results = await search_engine.hybrid_search(
            query="Acme", embedding=[1.0, 0.0, 0.0], top_k=10
        )

        assert len(results) >= 1
        assert results[0].source == "hybrid"
        assert results[0].node_id == "n1"


class TestNeo4jEnsureIndexes:
    @pytest.mark.asyncio
    async def test_idempotent(self, search_engine):
        await search_engine.ensure_indexes(
            vector_dimensions=3,
            vector_node_label="Company",
            vector_property="embedding",
            fulltext_node_labels=["Company", "Person"],
            fulltext_properties=["name"],
        )


class TestNeo4jHybridSearchProtocol:
    def test_satisfies_search_engine_protocol(self):
        from graphrag_core.interfaces import SearchEngine
        from graphrag_core.search.neo4j import Neo4jHybridSearch

        engine = Neo4jHybridSearch()
        assert isinstance(engine, SearchEngine)

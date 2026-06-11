"""Tests for InMemorySearchEngine."""

from __future__ import annotations

import math

import pytest

from graphrag_core.models import GraphNode


def _nodes() -> list[GraphNode]:
    return [
        GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}),
        GraphNode(id="n2", label="Company", properties={"name": "Globex Inc"}),
        GraphNode(id="n3", label="Person", properties={"name": "Alice"}),
    ]


def _embeddings() -> dict[str, list[float]]:
    return {
        "n1": [1.0, 0.0, 0.0],
        "n2": [0.0, 1.0, 0.0],
        "n3": [0.7, 0.7, 0.0],
    }


class TestInMemoryVectorSearch:
    @pytest.mark.asyncio
    async def test_returns_nearest_by_cosine(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.vector_search(query_embedding=[1.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3
        assert results[0].node_id == "n1"
        assert results[0].source == "vector"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.vector_search(query_embedding=[1.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].node_id == "n1"

    @pytest.mark.asyncio
    async def test_no_embeddings_returns_empty(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.vector_search(query_embedding=[1.0, 0.0, 0.0])

        assert results == []

    @pytest.mark.asyncio
    async def test_filters_by_property(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.vector_search(
            query_embedding=[0.7, 0.7, 0.0], top_k=10, filters={"label": "Company"}
        )

        assert all(r.label == "Company" for r in results)


class TestInMemoryFulltextSearch:
    @pytest.mark.asyncio
    async def test_matches_on_property_values(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="acme", top_k=10)

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "fulltext"

    @pytest.mark.asyncio
    async def test_bm25_term_frequency_beats_substring_length(self):
        """Bug #2: BM25 ranks by TF, not by substring length ratio.

        Substring impl picks 'single' (exact match score 1.0 > len-ratio 5/17).
        BM25 picks 'frequent' (TF=3 dominates over TF=1).
        """
        from graphrag_core.models import GraphNode
        from graphrag_core.search.memory import InMemorySearchEngine

        nodes = [
            GraphNode(id="single", label="Doc", properties={"text": "apple"}),
            GraphNode(id="frequent", label="Doc", properties={"text": "apple apple apple"}),
        ]
        engine = InMemorySearchEngine(nodes=nodes)
        results = await engine.fulltext_search(query="apple", top_k=2)
        assert results[0].node_id == "frequent", (
            f"BM25 should rank 'frequent' first; got {results[0].node_id}"
        )

    @pytest.mark.asyncio
    async def test_bm25_idf_weighting(self):
        """Bug #2: rare terms get higher IDF weight than common ones."""
        from graphrag_core.models import GraphNode
        from graphrag_core.search.memory import InMemorySearchEngine

        nodes = [
            GraphNode(id="common1", label="Doc", properties={"text": "apple"}),
            GraphNode(id="common2", label="Doc", properties={"text": "apple"}),
            GraphNode(id="common3", label="Doc", properties={"text": "apple"}),
            GraphNode(id="rare", label="Doc", properties={"text": "zebra"}),
        ]
        engine = InMemorySearchEngine(nodes=nodes)
        apple_results = await engine.fulltext_search(query="apple", top_k=5)
        zebra_results = await engine.fulltext_search(query="zebra", top_k=5)
        assert zebra_results[0].score > apple_results[0].score, (
            "rare-term hit should outscore common-term hit (IDF weighting)"
        )

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="ALICE", top_k=10)

        assert len(results) >= 1
        assert results[0].node_id == "n3"

    @pytest.mark.asyncio
    async def test_filters_by_node_types(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        nodes = _nodes()
        engine = InMemorySearchEngine(nodes=nodes)
        results = await engine.fulltext_search(query="acme alice", node_types=["Person"], top_k=10)

        assert all(r.label == "Person" for r in results)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="zzzznonexistent", top_k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        # All three nodes contain "inc" or "corp" or "alice"; query matches all three
        results = await engine.fulltext_search(query="acme globex alice", top_k=1)

        assert len(results) == 1


class TestInMemoryGraphSearch:
    @pytest.mark.asyncio
    async def test_returns_empty_without_graph_store(self):
        """When no graph_store is provided, graph_search returns [] (Protocol still satisfied)."""
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.graph_search(start_node_id="n1", pattern="WORKS_AT")

        assert results == []

    @pytest.mark.asyncio
    async def test_delegates_to_graph_store_when_provided(self):
        """Bug #7: graph_search delegates to GraphStore.get_related when wired."""
        from graphrag_core.graph.memory import InMemoryGraphStore
        from graphrag_core.models import GraphNode, GraphRelationship
        from graphrag_core.search.memory import InMemorySearchEngine

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="root", label="Topic", properties={"name": "Root"}), "r")
        await store.merge_node(GraphNode(id="rel", label="Claim", properties={"name": "Rel"}), "r")
        await store.merge_node(
            GraphNode(id="other", label="Topic", properties={"name": "Other"}), "r"
        )
        await store.merge_relationship(
            GraphRelationship(source_id="root", target_id="rel", type="ABOUT", properties={}),
            "r",
        )

        nodes = await store.list_nodes()
        engine = InMemorySearchEngine(nodes=nodes, graph_store=store)
        results = await engine.graph_search(start_node_id="root", pattern="ABOUT")

        assert len(results) == 1
        assert results[0].node_id == "rel"
        assert results[0].source == "graph"


class TestInMemoryHybridSearch:
    @pytest.mark.asyncio
    async def test_fuses_vector_and_fulltext(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.hybrid_search(
            query="Acme", embedding=[1.0, 0.0, 0.0], top_k=10
        )

        assert len(results) >= 1
        assert results[0].source == "hybrid"
        assert results[0].node_id == "n1"


class TestInMemorySearchEngineProtocol:
    def test_satisfies_search_engine_protocol(self):
        from graphrag_core.interfaces import SearchEngine
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=[])
        assert isinstance(engine, SearchEngine)


class TestInMemoryVectorSearchDimensionality:
    @pytest.mark.asyncio
    async def test_rejects_wrong_dimensionality(self):
        """Bug #6: vector_search must raise ValueError on dim mismatch (INTERFACE.md contract)."""
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())  # 3D stored
        with pytest.raises(ValueError, match="dimensionality|dimension"):
            await engine.vector_search(query_embedding=[1.0, 0.0])  # 2D query

    @pytest.mark.asyncio
    async def test_empty_embeddings_does_not_raise_on_mismatch(self):
        """When no embeddings stored, no dim to compare against — return [] silently."""
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())  # no embeddings
        results = await engine.vector_search(query_embedding=[1.0, 0.0])
        assert results == []

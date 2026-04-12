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

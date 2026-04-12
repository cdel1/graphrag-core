"""In-memory SearchEngine implementation for testing."""

from __future__ import annotations

import math

from graphrag_core.models import GraphNode, SearchResult
from graphrag_core.search.fusion import reciprocal_rank_fusion


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemorySearchEngine:
    """Dict-based SearchEngine for unit tests."""

    def __init__(
        self,
        nodes: list[GraphNode],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self._nodes = {n.id: n for n in nodes}
        self._embeddings = embeddings or {}

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if not self._embeddings:
            return []

        scored: list[tuple[str, float]] = []
        for node_id, emb in self._embeddings.items():
            node = self._nodes.get(node_id)
            if node is None:
                continue
            if filters and filters.get("label") and node.label != filters["label"]:
                continue
            score = _cosine_similarity(query_embedding, emb)
            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                node_id=nid,
                label=self._nodes[nid].label,
                score=score,
                source="vector",
                properties=self._nodes[nid].properties,
            )
            for nid, score in scored[:top_k]
        ]

    async def fulltext_search(
        self, query: str, node_types: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        return []

    async def graph_search(
        self, start_node_id: str, pattern: str, depth: int = 2
    ) -> list[SearchResult]:
        return []

    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]:
        return []

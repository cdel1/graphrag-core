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
        query_lower = query.lower()
        scored: list[tuple[str, float]] = []

        for node in self._nodes.values():
            if node_types and node.label not in node_types:
                continue
            searchable = " ".join(str(v) for v in node.properties.values()).lower()
            if query_lower in searchable:
                best_score = 0.0
                for val in node.properties.values():
                    val_lower = str(val).lower()
                    if val_lower == query_lower:
                        best_score = max(best_score, 1.0)
                    elif query_lower in val_lower:
                        best_score = max(best_score, len(query_lower) / len(val_lower))
                scored.append((node.id, best_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                node_id=nid,
                label=self._nodes[nid].label,
                score=score,
                source="fulltext",
                properties=self._nodes[nid].properties,
            )
            for nid, score in scored[:top_k]
        ]

    async def graph_search(
        self, start_node_id: str, pattern: str, depth: int = 2
    ) -> list[SearchResult]:
        return []

    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]:
        vector_results = await self.vector_search(query_embedding=embedding, top_k=top_k)
        fulltext_results = await self.fulltext_search(query=query, top_k=top_k)

        return reciprocal_rank_fusion([vector_results, fulltext_results], top_k=top_k)

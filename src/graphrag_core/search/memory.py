"""In-memory SearchEngine implementation for tests and lightweight demos."""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter

from graphrag_core.models import GraphNode, SearchResult
from graphrag_core.search.fusion import reciprocal_rank_fusion


_TOKEN_RE = re.compile(r"\w+")
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
        *,
        graph_store: object | None = None,
    ) -> None:
        self._nodes = {n.id: n for n in nodes}
        self._embeddings = embeddings or {}
        self._graph_store = graph_store
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_freq: dict[str, int] = {}
        for node_id, node in self._nodes.items():
            text = " ".join(str(v) for v in node.properties.values())
            tokens = _tokenize(text)
            self._doc_tokens[node_id] = tokens
            for term in set(tokens):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        n_docs = len(self._doc_tokens)
        total_len = sum(len(t) for t in self._doc_tokens.values())
        self._avgdl = (total_len / n_docs) if n_docs else 0.0

    def _bm25_score(self, query_tokens: list[str], node_id: str) -> float:
        n_docs = len(self._doc_tokens)
        if n_docs == 0 or self._avgdl == 0:
            return 0.0
        tokens = self._doc_tokens.get(node_id, [])
        if not tokens:
            return 0.0
        doc_len = len(tokens)
        tf = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            num = f * (_BM25_K1 + 1)
            den = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / self._avgdl)
            score += idf * num / den
        return score

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if not self._embeddings:
            return []

        first_dim = len(next(iter(self._embeddings.values())))
        if len(query_embedding) != first_dim:
            raise ValueError(
                f"query_embedding dimensionality {len(query_embedding)} "
                f"does not match stored embedding dimension {first_dim}"
            )

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
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[str, float]] = []
        for node_id, node in self._nodes.items():
            if node_types and node.label not in node_types:
                continue
            score = self._bm25_score(query_tokens, node_id)
            if score > 0:
                scored.append((node_id, score))

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
        if self._graph_store is None:
            return []
        related = await self._graph_store.get_related(  # type: ignore[attr-defined]
            start_node_id, rel_type=pattern, depth=depth
        )
        return [
            SearchResult(
                node_id=node.id,
                label=node.label,
                score=1.0 / (rank + 1),
                source="graph",
                properties=node.properties,
            )
            for rank, node in enumerate(related)
        ]

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        *,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        candidate_k = top_k * 2
        vector_results, fulltext_results = await asyncio.gather(
            self.vector_search(query_embedding=embedding, top_k=candidate_k),
            self.fulltext_search(query=query, top_k=candidate_k),
        )
        return reciprocal_rank_fusion(
            [vector_results, fulltext_results], top_k=top_k, k=rrf_k
        )

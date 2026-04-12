"""Reciprocal Rank Fusion for combining search results."""

from __future__ import annotations

from graphrag_core.models import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 10,
    k: int = 60,
) -> list[SearchResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            scores[result.node_id] = scores.get(result.node_id, 0.0) + 1.0 / (k + rank)
            if result.node_id not in best_result:
                best_result[result.node_id] = result

    sorted_ids = sorted(scores, key=lambda nid: scores[nid], reverse=True)[:top_k]

    return [
        SearchResult(
            node_id=nid,
            label=best_result[nid].label,
            score=scores[nid],
            source="hybrid",
            properties=best_result[nid].properties,
        )
        for nid in sorted_ids
    ]

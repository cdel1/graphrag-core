"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from graphrag_core.models import SearchResult


def _result(node_id: str, score: float = 0.0, source: str = "test") -> SearchResult:
    return SearchResult(node_id=node_id, label=node_id, score=score, source=source)


class TestReciprocalRankFusion:
    def test_fuses_two_lists(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        list_a = [_result("n1"), _result("n2"), _result("n3")]
        list_b = [_result("n2"), _result("n1"), _result("n4")]

        results = reciprocal_rank_fusion([list_a, list_b], top_k=10)

        ids = [r.node_id for r in results]
        assert "n1" in ids
        assert "n2" in ids
        assert "n3" in ids
        assert "n4" in ids
        n1_idx = ids.index("n1")
        n3_idx = ids.index("n3")
        assert n1_idx < n3_idx

    def test_overlapping_results_score_higher(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        list_a = [_result("n1"), _result("n2")]
        list_b = [_result("n1"), _result("n3")]

        results = reciprocal_rank_fusion([list_a, list_b], top_k=10)

        assert results[0].node_id == "n1"

    def test_single_list_passthrough(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        items = [_result("n1"), _result("n2"), _result("n3")]
        results = reciprocal_rank_fusion([items], top_k=10)

        assert len(results) == 3
        assert results[0].node_id == "n1"
        assert results[1].node_id == "n2"
        assert results[2].node_id == "n3"

    def test_empty_lists_returns_empty(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion([], top_k=10)
        assert results == []

        results = reciprocal_rank_fusion([[], []], top_k=10)
        assert results == []

    def test_respects_top_k(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        items = [_result(f"n{i}") for i in range(20)]
        results = reciprocal_rank_fusion([items], top_k=5)

        assert len(results) == 5

    def test_result_source_is_hybrid(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion(
            [[_result("n1", source="vector")], [_result("n1", source="fulltext")]],
            top_k=10,
        )
        assert results[0].source == "hybrid"

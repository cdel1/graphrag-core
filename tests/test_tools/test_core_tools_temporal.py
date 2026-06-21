"""BB7 v0.6.0: Temporal tools over the provenance-aware audit trail."""

import pytest

from graphrag_core.graph.memory import InMemoryGraphStore
from graphrag_core.models import GraphNode, GraphRelationship
from graphrag_core.tools.core_tools_temporal import (
    _compare_periods,
    _find_trend,
    _get_node_history,
    _resolve_period,
    make_compare_periods_tool,
    make_find_trend_tool,
    make_get_node_history_tool,
    register_temporal_tools,
)
from graphrag_core.tools.library import ToolLibrary


# ---------------------------------------------------------------------------
# Fixtures

async def _make_anchor_with_neighbors_in_periods(
    store: InMemoryGraphStore,
    anchor_id: str,
    periods_to_neighbor_count: dict[str, int],
    rel_type: str = "ABOUT",
) -> None:
    """Create an anchor node + N claims per period, all linked via rel_type.

    Each claim chained: claim -SOURCED(prov)- chunk -CHUNKED_FROM- doc{period: P}.
    """
    await store.merge_node(
        GraphNode(id=anchor_id, label="Entity", properties={}),
        import_run_id="run-1",
    )
    chunk_counter = 0
    for period, n_claims in periods_to_neighbor_count.items():
        doc_id = f"doc:{period}"
        await store.merge_node(
            GraphNode(id=doc_id, label="Document", properties={"period": period}),
            import_run_id="run-1",
        )
        for _ in range(n_claims):
            cid = f"c:{chunk_counter}"
            chk = f"chunk:{chunk_counter}"
            chunk_counter += 1
            await store.merge_node(GraphNode(id=cid, label="Claim", properties={}), "run-1")
            await store.merge_node(GraphNode(id=chk, label="Chunk", properties={}), "run-1")
            await store.record_provenance(cid, chk, "run-1")
            await store.merge_relationship(
                GraphRelationship(source_id=chk, target_id=doc_id,
                                  type="CHUNKED_FROM", properties={}),
                import_run_id="run-1",
            )
            await store.merge_relationship(
                GraphRelationship(source_id=cid, target_id=anchor_id,
                                  type=rel_type, properties={}),
                import_run_id="run-1",
            )


# ---------------------------------------------------------------------------
# _resolve_period

@pytest.mark.asyncio
async def test_resolve_period_walks_audit_trail():
    store = InMemoryGraphStore()
    await store.merge_node(
        GraphNode(id="doc:1", label="Document", properties={"period": "2026-Q2"}),
        "run-1",
    )
    await store.merge_node(GraphNode(id="claim:1", label="Claim", properties={}), "run-1")
    await store.merge_node(GraphNode(id="chunk:1", label="Chunk", properties={}), "run-1")
    await store.record_provenance("claim:1", "chunk:1", "run-1")
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:1",
                          type="CHUNKED_FROM", properties={}), "run-1",
    )
    assert await _resolve_period(store, "claim:1") == "2026-Q2"


@pytest.mark.asyncio
async def test_resolve_period_returns_none_when_no_document():
    store = InMemoryGraphStore()
    assert await _resolve_period(store, "missing-node") is None


# ---------------------------------------------------------------------------
# get_node_history

@pytest.mark.asyncio
async def test_get_node_history_groups_by_period():
    store = InMemoryGraphStore()
    await _make_anchor_with_neighbors_in_periods(
        store, "e:1", {"2026-Q1": 1, "2026-Q2": 1, "2026-Q3": 1},
    )
    history = await _get_node_history(store, "e:1", rel_type="ABOUT")
    assert set(history.periods.keys()) == {"2026-Q1", "2026-Q2", "2026-Q3"}
    assert all(len(v) == 1 for v in history.periods.values())


@pytest.mark.asyncio
async def test_get_node_history_period_bounds():
    store = InMemoryGraphStore()
    await _make_anchor_with_neighbors_in_periods(
        store, "e:1", {"2026-Q1": 1, "2026-Q2": 1, "2026-Q3": 1},
    )
    history = await _get_node_history(
        store, "e:1", rel_type="ABOUT",
        from_period="2026-Q2", to_period="2026-Q2",
    )
    assert set(history.periods.keys()) == {"2026-Q2"}


@pytest.mark.asyncio
async def test_get_node_history_rel_type_filter():
    """rel_type filter omits neighbors connected by other edge types."""
    store = InMemoryGraphStore()
    await _make_anchor_with_neighbors_in_periods(
        store, "e:1", {"2026-Q1": 1, "2026-Q2": 1}, rel_type="ABOUT",
    )
    # Add an unrelated edge
    await store.merge_node(
        GraphNode(id="doc:other", label="Document", properties={"period": "2026-Q4"}),
        "run-1",
    )
    await store.merge_node(GraphNode(id="stk:1", label="Stakeholder", properties={}), "run-1")
    await store.merge_node(GraphNode(id="chunk:99", label="Chunk", properties={}), "run-1")
    await store.record_provenance("stk:1", "chunk:99", "run-1")
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:99", target_id="doc:other",
                          type="CHUNKED_FROM", properties={}), "run-1",
    )
    await store.merge_relationship(
        GraphRelationship(source_id="stk:1", target_id="e:1",
                          type="ASSERTS", properties={}), "run-1",
    )

    history = await _get_node_history(store, "e:1", rel_type="ABOUT")
    assert "2026-Q4" not in history.periods  # ASSERTS-only stakeholder excluded
    assert set(history.periods.keys()) == {"2026-Q1", "2026-Q2"}


# ---------------------------------------------------------------------------
# compare_periods

@pytest.mark.asyncio
async def test_compare_periods_added_removed():
    """Neighbors in to_period but not from_period -> added; converse -> removed."""
    store = InMemoryGraphStore()
    await _make_anchor_with_neighbors_in_periods(
        store, "e:1", {"2026-Q1": 1, "2026-Q2": 1}, rel_type="ABOUT",
    )
    diff = await _compare_periods(
        store, "e:1", period_from="2026-Q1", period_to="2026-Q2", rel_type="ABOUT",
    )
    # Q1 claim is removed, Q2 claim is added
    assert len(diff.added) == 1
    assert len(diff.removed) == 1


# ---------------------------------------------------------------------------
# find_trend

@pytest.mark.asyncio
@pytest.mark.parametrize("counts,expected", [
    ({"2026-Q1": 1, "2026-Q2": 2, "2026-Q3": 3}, "increasing"),
    ({"2026-Q1": 3, "2026-Q2": 2, "2026-Q3": 1}, "decreasing"),
    ({"2026-Q1": 2, "2026-Q2": 2, "2026-Q3": 2}, "stable"),
    ({"2026-Q1": 1}, "insufficient_data"),
])
async def test_find_trend_directions(counts, expected):
    store = InMemoryGraphStore()
    await _make_anchor_with_neighbors_in_periods(store, "e:1", counts, rel_type="ABOUT")
    trend = await _find_trend(store, "e:1", rel_type="ABOUT")
    assert trend.direction == expected


# ---------------------------------------------------------------------------
# Registration

@pytest.mark.asyncio
async def test_register_temporal_tools_registers_all_three():
    store = InMemoryGraphStore()
    library = ToolLibrary()
    register_temporal_tools(library, store)
    names = {t.name for t in library.list_tools()}
    assert names == {"get_node_history", "compare_periods", "find_trend"}


# ---------------------------------------------------------------------------
# Handler exception-safety (Task 13)

@pytest.mark.asyncio
async def test_temporal_tool_handler_returns_success_for_unknown_node():
    """Per BB7 contract: handlers never raise — they return ToolResult."""
    store = InMemoryGraphStore()
    library = ToolLibrary()
    register_temporal_tools(library, store)

    result = await library.execute("get_node_history", node_id="does-not-exist")
    assert result.success is True  # success with empty periods, not an error
    assert result.data["periods"] == {}


@pytest.mark.asyncio
async def test_temporal_tool_handler_wraps_internal_error_as_tool_result():
    """If the handler's internal logic raises, it's wrapped in a ToolResult."""
    # Construct a faulty graph_store that raises on get_related
    class FaultyStore:
        async def get_related(self, *a, **kw):
            raise RuntimeError("boom")

        async def get_audit_trail(self, *a, **kw):
            from graphrag_core.models import ProvenanceTrail
            return ProvenanceTrail(node_id="x", provenance_chain=[])

    tool = make_get_node_history_tool(FaultyStore())
    result = await tool.handler(node_id="anything")
    assert result.success is False
    assert "boom" in (result.error or "")


# ---------------------------------------------------------------------------
# compare_periods edge cases

@pytest.mark.asyncio
async def test_compare_periods_same_period_returns_empty_diff():
    """When period_from == period_to, added and removed are both empty."""
    store = InMemoryGraphStore()
    await _make_anchor_with_neighbors_in_periods(
        store, "e:1", {"2026-Q2": 2}, rel_type="ABOUT",
    )
    diff = await _compare_periods(
        store, "e:1", period_from="2026-Q2", period_to="2026-Q2", rel_type="ABOUT",
    )
    assert diff.added == []
    assert diff.removed == []

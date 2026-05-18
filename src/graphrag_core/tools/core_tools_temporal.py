"""BB7: Temporal tools over the provenance-aware audit trail.

These tools route through GraphStore.get_audit_trail to resolve a node's
source-document period — no hardcoded Lacuna labels or edge names. The
optional rel_type kwarg lets callers filter neighbors at the call site
(e.g., Lacuna passes rel_type="ABOUT" for claim-only history). Without
rel_type, all incident edges (including structural provenance edges like
EXTRACTED_FROM, CHUNKED_FROM) are counted as neighbors — almost never the
desired semantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from graphrag_core.models import ToolParameter, ToolResult
from graphrag_core.tools.library import Tool, ToolLibrary

if TYPE_CHECKING:
    from graphrag_core.interfaces import GraphStore


@dataclass
class NodeHistory:
    """A node's neighbors grouped by their resolved source-document period."""

    node_id: str
    periods: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class PeriodDiff:
    """Set-diff of a node's neighbors between two periods."""

    node_id: str
    period_from: str
    period_to: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


@dataclass
class TrendSignal:
    """Direction of a node's neighbor-count change across periods.

    `direction` is neutral on whether more neighbors is good or bad —
    that's a domain interpretation owned by consumers. Possible values:
    "increasing", "decreasing", "stable", "insufficient_data".
    """

    node_id: str
    direction: str
    counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Period resolution

async def _resolve_period(graph_store: "GraphStore", node_id: str) -> str | None:
    """Walk the audit trail; return the document-level period or None."""
    trail = await graph_store.get_audit_trail(node_id)
    for step in trail.provenance_chain:
        if step.level == "document":
            return step.metadata.get("period")
    return None


# ---------------------------------------------------------------------------
# get_node_history

async def _get_node_history(
    graph_store: "GraphStore",
    node_id: str,
    rel_type: str | None = None,
    from_period: str | None = None,
    to_period: str | None = None,
) -> NodeHistory:
    neighbors = await graph_store.get_related(node_id, rel_type=rel_type, depth=1)
    periods: dict[str, list[str]] = {}
    for n in neighbors:
        period = await _resolve_period(graph_store, n.id)
        if period is None:
            continue
        if from_period and period < from_period:
            continue
        if to_period and period > to_period:
            continue
        periods.setdefault(period, []).append(n.id)
    return NodeHistory(node_id=node_id, periods=periods)


def make_get_node_history_tool(graph_store: "GraphStore") -> Tool:
    async def handler(*, node_id, rel_type=None,
                      from_period=None, to_period=None) -> ToolResult:
        try:
            history = await _get_node_history(
                graph_store, node_id, rel_type, from_period, to_period,
            )
            return ToolResult(success=True, data={
                "node_id": history.node_id,
                "periods": history.periods,
            })
        except Exception as e:
            return ToolResult(success=False, error=f"handler error: {e}")

    return Tool(
        name="get_node_history",
        description="Group a node's neighbors by their resolved source-document period.",
        parameters={
            "node_id": ToolParameter(name="node_id", type="string",
                description="Anchor node ID", required=True),
            "rel_type": ToolParameter(name="rel_type", type="string",
                description="Filter neighbors by relationship type. Strongly recommended: "
                "without it, all edges (including structural provenance) are counted as neighbors, "
                "which is rarely what callers want for semantic analysis.",
                required=False),
            "from_period": ToolParameter(name="from_period", type="string",
                description="Lower bound (inclusive, lex order)", required=False),
            "to_period": ToolParameter(name="to_period", type="string",
                description="Upper bound (inclusive, lex order)", required=False),
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# compare_periods

async def _compare_periods(
    graph_store: "GraphStore",
    node_id: str,
    period_from: str,
    period_to: str,
    rel_type: str | None = None,
) -> PeriodDiff:
    neighbors = await graph_store.get_related(node_id, rel_type=rel_type, depth=1)
    from_ids: set[str] = set()
    to_ids: set[str] = set()
    for n in neighbors:
        period = await _resolve_period(graph_store, n.id)
        if period == period_from:
            from_ids.add(n.id)
        if period == period_to:
            to_ids.add(n.id)
    return PeriodDiff(
        node_id=node_id,
        period_from=period_from,
        period_to=period_to,
        added=sorted(to_ids - from_ids),
        removed=sorted(from_ids - to_ids),
    )


def make_compare_periods_tool(graph_store: "GraphStore") -> Tool:
    async def handler(*, node_id, period_from, period_to, rel_type=None) -> ToolResult:
        try:
            diff = await _compare_periods(graph_store, node_id, period_from,
                                          period_to, rel_type)
            return ToolResult(success=True, data={
                "node_id": diff.node_id,
                "period_from": diff.period_from,
                "period_to": diff.period_to,
                "added": diff.added,
                "removed": diff.removed,
            })
        except Exception as e:
            return ToolResult(success=False, error=f"handler error: {e}")

    return Tool(
        name="compare_periods",
        description="Set-diff of a node's neighbors between two periods (added / removed).",
        parameters={
            "node_id": ToolParameter(name="node_id", type="string",
                description="Anchor node ID", required=True),
            "period_from": ToolParameter(name="period_from", type="string",
                description="Baseline period", required=True),
            "period_to": ToolParameter(name="period_to", type="string",
                description="Comparison period", required=True),
            "rel_type": ToolParameter(name="rel_type", type="string",
                description="Filter neighbors by relationship type. Strongly recommended: "
                "without it, all edges (including structural provenance) are counted as neighbors, "
                "which is rarely what callers want for semantic analysis.",
                required=False),
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# find_trend

async def _find_trend(
    graph_store: "GraphStore",
    node_id: str,
    rel_type: str | None = None,
) -> TrendSignal:
    neighbors = await graph_store.get_related(node_id, rel_type=rel_type, depth=1)
    counts: dict[str, int] = {}
    for n in neighbors:
        period = await _resolve_period(graph_store, n.id)
        if period is None:
            continue
        counts[period] = counts.get(period, 0) + 1

    periods = sorted(counts.keys())
    if len(periods) < 2:
        direction = "insufficient_data"
    else:
        first, last = counts[periods[0]], counts[periods[-1]]
        if last > first:
            direction = "increasing"
        elif last < first:
            direction = "decreasing"
        else:
            direction = "stable"

    return TrendSignal(node_id=node_id, direction=direction, counts=counts)


def make_find_trend_tool(graph_store: "GraphStore") -> Tool:
    async def handler(*, node_id, rel_type=None) -> ToolResult:
        try:
            trend = await _find_trend(graph_store, node_id, rel_type)
            return ToolResult(success=True, data={
                "node_id": trend.node_id,
                "direction": trend.direction,
                "counts": trend.counts,
            })
        except Exception as e:
            return ToolResult(success=False, error=f"handler error: {e}")

    return Tool(
        name="find_trend",
        description="Direction of a node's neighbor-count change across periods.",
        parameters={
            "node_id": ToolParameter(name="node_id", type="string",
                description="Anchor node ID", required=True),
            "rel_type": ToolParameter(name="rel_type", type="string",
                description="Filter neighbors by relationship type. Strongly recommended: "
                "without it, all edges (including structural provenance) are counted as neighbors, "
                "which is rarely what callers want for semantic analysis.",
                required=False),
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Bulk registration

def register_temporal_tools(library: ToolLibrary, graph_store: "GraphStore") -> None:
    """Register all 3 BB7 temporal tools on the given library."""
    library.register(make_get_node_history_tool(graph_store))
    library.register(make_compare_periods_tool(graph_store))
    library.register(make_find_trend_tool(graph_store))

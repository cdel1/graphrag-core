"""BB7: Domain-agnostic core tools for graph queries."""

from __future__ import annotations

from graphrag_core.models import ToolParameter, ToolResult
from graphrag_core.tools.library import Tool, ToolLibrary


def make_get_entity_tool(graph_store: object) -> Tool:
    """Create a tool that retrieves a node by ID."""

    async def handler(*, entity_id: str) -> ToolResult:
        node = await graph_store.get_node(entity_id)
        if node is None:
            return ToolResult(success=False, error=f"Entity '{entity_id}' not found")
        return ToolResult(success=True, data=node.model_dump())

    return Tool(
        name="get_entity",
        description="Retrieve an entity (node) by its ID",
        parameters={
            "entity_id": ToolParameter(name="entity_id", type="string", description="The node ID to retrieve"),
        },
        handler=handler,
    )


def make_search_entities_tool(search_engine: object) -> Tool:
    """Create a tool that searches entities by text query."""

    async def handler(*, query: str, node_types: list[str] | None = None, top_k: int = 10) -> ToolResult:
        results = await search_engine.fulltext_search(query=query, node_types=node_types, top_k=top_k)
        return ToolResult(success=True, data=[r.model_dump() for r in results])

    return Tool(
        name="search_entities",
        description="Search for entities by text query using fulltext search",
        parameters={
            "query": ToolParameter(name="query", type="string", description="Search query text"),
            "node_types": ToolParameter(name="node_types", type="list[string]", description="Filter by node types", required=False),
            "top_k": ToolParameter(name="top_k", type="integer", description="Max results to return", required=False),
        },
        handler=handler,
    )


def make_get_audit_trail_tool(graph_store: object) -> Tool:
    """Create a tool that retrieves the provenance chain for a node."""

    async def handler(*, node_id: str) -> ToolResult:
        trail = await graph_store.get_audit_trail(node_id)
        return ToolResult(success=True, data=trail.model_dump())

    return Tool(
        name="get_audit_trail",
        description="Retrieve the provenance audit trail for a node",
        parameters={
            "node_id": ToolParameter(name="node_id", type="string", description="The node ID to trace"),
        },
        handler=handler,
    )


def make_get_related_tool(graph_store: object) -> Tool:
    """Create a tool that finds nodes related to a given node."""

    async def handler(*, node_id: str, rel_type: str | None = None, depth: int = 1) -> ToolResult:
        nodes = await graph_store.get_related(node_id, rel_type=rel_type, depth=depth)
        return ToolResult(success=True, data=[n.model_dump() for n in nodes])

    return Tool(
        name="get_related",
        description="Find nodes related to a given node by relationship type and depth",
        parameters={
            "node_id": ToolParameter(name="node_id", type="string", description="Starting node ID"),
            "rel_type": ToolParameter(name="rel_type", type="string", description="Relationship type filter", required=False),
            "depth": ToolParameter(name="depth", type="integer", description="Traversal depth", required=False),
        },
        handler=handler,
    )


def register_core_tools(library: ToolLibrary, graph_store: object, search_engine: object) -> None:
    """Register all 4 domain-agnostic core tools."""
    library.register(make_get_entity_tool(graph_store))
    library.register(make_search_entities_tool(search_engine))
    library.register(make_get_audit_trail_tool(graph_store))
    library.register(make_get_related_tool(graph_store))

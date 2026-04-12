"""Tests for core tool factories."""

from __future__ import annotations

import pytest

from graphrag_core.models import GraphNode, GraphRelationship


async def _store_with_data():
    from graphrag_core.graph.memory import InMemoryGraphStore

    store = InMemoryGraphStore()
    await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1")
    await store.merge_node(GraphNode(id="n2", label="Person", properties={"name": "Alice"}), "run-1")
    await store.merge_relationship(
        GraphRelationship(source_id="n2", target_id="n1", type="WORKS_AT"), "run-1"
    )
    await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")
    return store


def _search_engine():
    from graphrag_core.search.memory import InMemorySearchEngine

    nodes = [
        GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}),
        GraphNode(id="n2", label="Person", properties={"name": "Alice"}),
    ]
    return InMemorySearchEngine(nodes=nodes)


class TestGetEntityTool:
    @pytest.mark.asyncio
    async def test_returns_existing_node(self):
        from graphrag_core.tools.core_tools import make_get_entity_tool

        store = await _store_with_data()
        tool = make_get_entity_tool(store)

        result = await tool.handler(entity_id="n1")
        assert result.success is True
        assert result.data["id"] == "n1"
        assert result.data["label"] == "Company"

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_node(self):
        from graphrag_core.tools.core_tools import make_get_entity_tool

        store = await _store_with_data()
        tool = make_get_entity_tool(store)

        result = await tool.handler(entity_id="nonexistent")
        assert result.success is False
        assert result.error is not None


class TestSearchEntitiesTool:
    @pytest.mark.asyncio
    async def test_returns_search_results(self):
        from graphrag_core.tools.core_tools import make_search_entities_tool

        engine = _search_engine()
        tool = make_search_entities_tool(engine)

        result = await tool.handler(query="acme", top_k=5)
        assert result.success is True
        assert len(result.data) >= 1


class TestGetAuditTrailTool:
    @pytest.mark.asyncio
    async def test_returns_provenance(self):
        from graphrag_core.tools.core_tools import make_get_audit_trail_tool

        store = await _store_with_data()
        tool = make_get_audit_trail_tool(store)

        result = await tool.handler(node_id="n1")
        assert result.success is True
        assert result.data["node_id"] == "n1"
        assert len(result.data["provenance_chain"]) >= 1


class TestGetRelatedTool:
    @pytest.mark.asyncio
    async def test_returns_connected_nodes(self):
        from graphrag_core.tools.core_tools import make_get_related_tool

        store = await _store_with_data()
        tool = make_get_related_tool(store)

        result = await tool.handler(node_id="n2", rel_type="WORKS_AT", depth=1)
        assert result.success is True
        assert len(result.data) >= 1
        assert result.data[0]["id"] == "n1"


class TestRegisterCoreTools:
    @pytest.mark.asyncio
    async def test_registers_all_four(self):
        from graphrag_core.tools.core_tools import register_core_tools
        from graphrag_core.tools.library import ToolLibrary

        store = await _store_with_data()
        engine = _search_engine()
        library = ToolLibrary()

        register_core_tools(library, store, engine)

        tools = library.list_tools()
        names = {t.name for t in tools}
        assert names == {"get_entity", "search_entities", "get_audit_trail", "get_related"}

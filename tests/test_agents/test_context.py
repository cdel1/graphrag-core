"""Tests for AgentContext."""

from __future__ import annotations


class TestAgentContext:
    def test_holds_live_objects(self):
        from graphrag_core.agents.context import AgentContext

        context = AgentContext(
            graph_store="fake_store",
            tool_library="fake_library",
            search_engine="fake_engine",
        )
        assert context.graph_store == "fake_store"
        assert context.tool_library == "fake_library"
        assert context.search_engine == "fake_engine"

    def test_workflow_state_defaults_to_empty(self):
        from graphrag_core.agents.context import AgentContext

        context = AgentContext(
            graph_store="s", tool_library="t", search_engine="e"
        )
        assert context.workflow_state == {}

    def test_workflow_state_is_mutable(self):
        from graphrag_core.agents.context import AgentContext

        context = AgentContext(
            graph_store="s", tool_library="t", search_engine="e"
        )
        context.workflow_state["key"] = "value"
        assert context.workflow_state["key"] == "value"

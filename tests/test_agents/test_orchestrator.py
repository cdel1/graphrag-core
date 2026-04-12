"""Tests for SequentialOrchestrator."""

from __future__ import annotations

import pytest

from graphrag_core.agents.context import AgentContext
from graphrag_core.models import AgentResult


class FakeAgent:
    def __init__(self, name: str, success: bool = True, data: dict | None = None):
        self.name = name
        self._success = success
        self._data = data or {}

    async def execute(self, context: AgentContext) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=self._success,
            data=self._data,
        )


class StateWritingAgent:
    def __init__(self, name: str, key: str, value: str):
        self.name = name
        self._key = key
        self._value = value

    async def execute(self, context: AgentContext) -> AgentResult:
        context.workflow_state[self._key] = self._value
        return AgentResult(agent_name=self.name, success=True)


class StateReadingAgent:
    def __init__(self, name: str, key: str):
        self.name = name
        self._key = key

    async def execute(self, context: AgentContext) -> AgentResult:
        value = context.workflow_state.get(self._key, "NOT_FOUND")
        return AgentResult(agent_name=self.name, success=True, data={"read_value": value})


class TestSequentialOrchestrator:
    @pytest.mark.asyncio
    async def test_runs_agents_in_order(self):
        from graphrag_core.agents.orchestrator import SequentialOrchestrator

        agents = [FakeAgent("first"), FakeAgent("second"), FakeAgent("third")]
        context = AgentContext(graph_store=None, tool_library=None, search_engine=None)

        orch = SequentialOrchestrator()
        result = await orch.run_workflow("wf-1", agents, context)

        assert result.success is True
        assert result.workflow_id == "wf-1"
        assert len(result.agent_results) == 3
        assert [r.agent_name for r in result.agent_results] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_stops_on_first_failure(self):
        from graphrag_core.agents.orchestrator import SequentialOrchestrator

        agents = [
            FakeAgent("ok"),
            FakeAgent("fail", success=False),
            FakeAgent("never_reached"),
        ]
        context = AgentContext(graph_store=None, tool_library=None, search_engine=None)

        orch = SequentialOrchestrator()
        result = await orch.run_workflow("wf-1", agents, context)

        assert result.success is False
        assert len(result.agent_results) == 2
        assert result.agent_results[1].agent_name == "fail"

    @pytest.mark.asyncio
    async def test_agents_share_workflow_state(self):
        from graphrag_core.agents.orchestrator import SequentialOrchestrator

        agents = [
            StateWritingAgent("writer", key="msg", value="hello"),
            StateReadingAgent("reader", key="msg"),
        ]
        context = AgentContext(graph_store=None, tool_library=None, search_engine=None)

        orch = SequentialOrchestrator()
        result = await orch.run_workflow("wf-1", agents, context)

        assert result.success is True
        reader_result = result.agent_results[1]
        assert reader_result.data["read_value"] == "hello"

    @pytest.mark.asyncio
    async def test_empty_agent_list_returns_success(self):
        from graphrag_core.agents.orchestrator import SequentialOrchestrator

        context = AgentContext(graph_store=None, tool_library=None, search_engine=None)

        orch = SequentialOrchestrator()
        result = await orch.run_workflow("wf-1", [], context)

        assert result.success is True
        assert result.agent_results == []

    def test_satisfies_orchestrator_protocol(self):
        from graphrag_core.agents.orchestrator import SequentialOrchestrator
        from graphrag_core.interfaces import Orchestrator

        orch = SequentialOrchestrator()
        assert isinstance(orch, Orchestrator)

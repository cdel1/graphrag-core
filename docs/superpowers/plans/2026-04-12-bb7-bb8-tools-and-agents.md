# BB7 + BB8: Core Tool Library & Multi-Agent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement BB7 (ToolLibrary + 4 domain-agnostic core tools) and BB8 (Agent/Orchestrator/ReportRenderer Protocols + SequentialOrchestrator), completing all 8 building blocks of graphrag-core.

**Architecture:** BB7 is a concrete ToolLibrary class + 4 core tool factory functions that wrap GraphStore/SearchEngine methods. BB8 adds Agent, Orchestrator, ReportRenderer Protocols plus a SequentialOrchestrator that runs agents in order. AgentContext is a dataclass holding live service instances. No heavy dependencies (no LangGraph).

**Tech Stack:** Python 3.12+, Pydantic v2, dataclasses, pytest + pytest-asyncio

**Design spec:** `docs/superpowers/specs/2026-04-12-bb7-bb8-tools-and-agents-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `src/graphrag_core/tools/__init__.py` | Re-exports Tool, ToolLibrary, register_core_tools |
| `src/graphrag_core/tools/library.py` | Tool dataclass, ToolLibrary class |
| `src/graphrag_core/tools/core_tools.py` | 4 core tool factories + register_core_tools |
| `src/graphrag_core/agents/__init__.py` | Re-exports AgentContext, SequentialOrchestrator |
| `src/graphrag_core/agents/context.py` | AgentContext dataclass |
| `src/graphrag_core/agents/orchestrator.py` | SequentialOrchestrator |
| `tests/test_tools/__init__.py` | Package marker |
| `tests/test_tools/test_library.py` | ToolLibrary unit tests |
| `tests/test_tools/test_core_tools.py` | Core tool unit tests |
| `tests/test_agents/__init__.py` | Package marker |
| `tests/test_agents/test_context.py` | AgentContext unit tests |
| `tests/test_agents/test_orchestrator.py` | SequentialOrchestrator unit tests |

### Modified files

| File | Change |
|---|---|
| `src/graphrag_core/models.py` | Add BB7 + BB8 models |
| `src/graphrag_core/interfaces.py` | Add Agent, Orchestrator, ReportRenderer Protocols |
| `src/graphrag_core/__init__.py` | Add BB7 + BB8 re-exports |
| `tests/test_interfaces.py` | Add Protocol conformance tests |

---

## Task 1: Add BB7 + BB8 models

**Files:**
- Modify: `src/graphrag_core/models.py`

- [ ] **Step 1: Add models**

Append after the BB6 section (after line 203) in `src/graphrag_core/models.py`:

```python


# ---------------------------------------------------------------------------
# BB7: Core Tool Library
# ---------------------------------------------------------------------------

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None


# ---------------------------------------------------------------------------
# BB8: Multi-Agent Orchestration
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: dict[str, Any] = {}
    error: str | None = None


class WorkflowResult(BaseModel):
    workflow_id: str
    success: bool
    agent_results: list[AgentResult]


class ReportData(BaseModel):
    title: str
    sections: list[dict[str, Any]]


class RenderConfig(BaseModel):
    format: str = "markdown"
    options: dict[str, Any] = {}
```

- [ ] **Step 2: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 139 passed, 19 skipped

- [ ] **Step 3: Commit**

```bash
git add src/graphrag_core/models.py
git commit -m "feat: add BB7 tool and BB8 agent models"
```

---

## Task 2: Add BB8 Protocols

**Files:**
- Modify: `src/graphrag_core/interfaces.py`
- Modify: `tests/test_interfaces.py`

- [ ] **Step 1: Add imports and Protocols to interfaces.py**

Update the import block at the top of `src/graphrag_core/interfaces.py` to include the new model types:

```python
from graphrag_core.models import (
    AgentResult,
    ApplyResult,
    ApprovalBatch,
    AuditTrail,
    ChunkConfig,
    CurationIssue,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    KnownEntity,
    OntologySchema,
    ParsedDocument,
    RegistryMatch,
    RenderConfig,
    ReportData,
    SchemaViolation,
    SearchResult,
    WorkflowResult,
)
```

Append after the BB6 section:

```python


# ---------------------------------------------------------------------------
# BB8: Multi-Agent Orchestration
# ---------------------------------------------------------------------------

@runtime_checkable
class Agent(Protocol):
    """A single agent with a defined role."""

    name: str

    async def execute(self, context: object) -> AgentResult: ...


@runtime_checkable
class Orchestrator(Protocol):
    """Coordinates multi-agent workflows."""

    async def run_workflow(
        self, workflow_id: str, agents: list[Agent], context: object
    ) -> WorkflowResult: ...


@runtime_checkable
class ReportRenderer(Protocol):
    """Renders structured report data into output format."""

    async def render(
        self, report_data: ReportData, template: str, config: RenderConfig
    ) -> bytes: ...
```

Note: `context` is typed as `object` in the Protocol signatures to avoid importing `AgentContext` (which is a dataclass in `agents/context.py`, not in `models.py`). This avoids circular imports. Concrete implementations will type-hint it as `AgentContext`.

- [ ] **Step 2: Add Protocol conformance tests**

Read `tests/test_interfaces.py` first, then add the new imports and test classes.

Add to the imports:

```python
from graphrag_core.interfaces import (
    Agent,
    ApprovalGateway,
    Chunker,
    DetectionLayer,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    GraphStore,
    LLMClient,
    LLMCurationLayer,
    Orchestrator,
    ReportRenderer,
    SearchEngine,
)
from graphrag_core.models import (
    AgentResult,
    ApplyResult,
    ApprovalBatch,
    AuditTrail,
    ChunkConfig,
    CurationIssue,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    KnownEntity,
    OntologySchema,
    ParsedDocument,
    RegistryMatch,
    RenderConfig,
    ReportData,
    SchemaViolation,
    SearchResult,
    WorkflowResult,
)
```

Append test classes:

```python
class TestAgentProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyAgent:
            name = "test-agent"

            async def execute(self, context: object) -> AgentResult:
                raise NotImplementedError

        agent: Agent = MyAgent()
        assert isinstance(agent, Agent)


class TestOrchestratorProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyOrchestrator:
            async def run_workflow(
                self, workflow_id: str, agents: list[Agent], context: object
            ) -> WorkflowResult:
                raise NotImplementedError

        orch: Orchestrator = MyOrchestrator()
        assert isinstance(orch, Orchestrator)


class TestReportRendererProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyRenderer:
            async def render(
                self, report_data: ReportData, template: str, config: RenderConfig
            ) -> bytes:
                raise NotImplementedError

        renderer: ReportRenderer = MyRenderer()
        assert isinstance(renderer, ReportRenderer)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_interfaces.py -v`
Expected: 14 tests PASS (11 existing + 3 new)

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 142 passed, 19 skipped

- [ ] **Step 5: Commit**

```bash
git add src/graphrag_core/interfaces.py tests/test_interfaces.py
git commit -m "feat: add Agent, Orchestrator, and ReportRenderer Protocols"
```

---

## Task 3: ToolLibrary

**Files:**
- Create: `tests/test_tools/__init__.py`
- Create: `tests/test_tools/test_library.py`
- Create: `src/graphrag_core/tools/__init__.py`
- Create: `src/graphrag_core/tools/library.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools/__init__.py` (empty).

Create `tests/test_tools/test_library.py`:

```python
"""Tests for ToolLibrary."""

from __future__ import annotations

import pytest

from graphrag_core.models import ToolParameter, ToolResult


class TestToolLibraryRegister:
    def test_register_and_get(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True, data="ok")

        library = ToolLibrary()
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"input": ToolParameter(name="input", type="string", description="test input")},
            handler=handler,
        )
        library.register(tool)

        retrieved = library.get("test_tool")
        assert retrieved.name == "test_tool"

    def test_duplicate_register_raises(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True)

        library = ToolLibrary()
        tool = Tool(name="dup", description="", parameters={}, handler=handler)
        library.register(tool)

        with pytest.raises(ValueError):
            library.register(tool)

    def test_get_unknown_raises(self):
        from graphrag_core.tools.library import ToolLibrary

        library = ToolLibrary()
        with pytest.raises(KeyError):
            library.get("nonexistent")

    def test_list_tools(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True)

        library = ToolLibrary()
        library.register(Tool(name="a", description="", parameters={}, handler=handler))
        library.register(Tool(name="b", description="", parameters={}, handler=handler))

        tools = library.list_tools()
        names = [t.name for t in tools]
        assert "a" in names
        assert "b" in names
        assert len(tools) == 2


class TestToolLibraryExecute:
    @pytest.mark.asyncio
    async def test_execute_calls_handler(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True, data=kwargs.get("value"))

        library = ToolLibrary()
        library.register(Tool(name="echo", description="", parameters={}, handler=handler))

        result = await library.execute("echo", value=42)
        assert result.success is True
        assert result.data == 42

    @pytest.mark.asyncio
    async def test_execute_catches_exceptions(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def failing_handler(**kwargs) -> ToolResult:
            raise RuntimeError("boom")

        library = ToolLibrary()
        library.register(Tool(name="fail", description="", parameters={}, handler=failing_handler))

        result = await library.execute("fail")
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        from graphrag_core.tools.library import ToolLibrary

        library = ToolLibrary()
        with pytest.raises(KeyError):
            await library.execute("nonexistent")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_tools/test_library.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/graphrag_core/tools/__init__.py`:

```python
"""BB7: Core tool library."""

__all__: list[str] = []
```

Create `src/graphrag_core/tools/library.py`:

```python
"""BB7: Tool registry and execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from graphrag_core.models import ToolParameter, ToolResult


@dataclass
class Tool:
    """A callable tool with metadata."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    handler: Callable[..., Awaitable[ToolResult]]


class ToolLibrary:
    """Registry of tested, schema-validated graph query tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        try:
            return await tool.handler(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools/test_library.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 149 passed, 19 skipped

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/tools/ tests/test_tools/
git commit -m "feat: add ToolLibrary with register, get, list, execute"
```

---

## Task 4: Core tools

**Files:**
- Create: `tests/test_tools/test_core_tools.py`
- Create: `src/graphrag_core/tools/core_tools.py`
- Modify: `src/graphrag_core/tools/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools/test_core_tools.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_tools/test_core_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement core tools**

Create `src/graphrag_core/tools/core_tools.py`:

```python
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
```

- [ ] **Step 4: Update tools __init__.py**

Replace `src/graphrag_core/tools/__init__.py`:

```python
"""BB7: Core tool library."""

from graphrag_core.tools.core_tools import register_core_tools
from graphrag_core.tools.library import Tool, ToolLibrary

__all__ = ["Tool", "ToolLibrary", "register_core_tools"]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tools/test_core_tools.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 155 passed, 19 skipped

- [ ] **Step 7: Commit**

```bash
git add src/graphrag_core/tools/ tests/test_tools/
git commit -m "feat: add 4 domain-agnostic core tools and register_core_tools"
```

---

## Task 5: AgentContext

**Files:**
- Create: `tests/test_agents/__init__.py`
- Create: `tests/test_agents/test_context.py`
- Create: `src/graphrag_core/agents/__init__.py`
- Create: `src/graphrag_core/agents/context.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents/__init__.py` (empty).

Create `tests/test_agents/test_context.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_agents/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/graphrag_core/agents/__init__.py`:

```python
"""BB8: Agent orchestration."""

__all__: list[str] = []
```

Create `src/graphrag_core/agents/context.py`:

```python
"""BB8: Shared context for agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """Runtime context passed between agents in a workflow.

    Fields are typed as Any to avoid circular imports — Protocol
    conformance is enforced at the call site.
    """

    graph_store: Any
    tool_library: Any
    search_engine: Any
    workflow_state: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_agents/test_context.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 158 passed, 19 skipped

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/agents/ tests/test_agents/
git commit -m "feat: add AgentContext dataclass"
```

---

## Task 6: SequentialOrchestrator

**Files:**
- Create: `tests/test_agents/test_orchestrator.py`
- Create: `src/graphrag_core/agents/orchestrator.py`
- Modify: `src/graphrag_core/agents/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents/test_orchestrator.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_agents/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `src/graphrag_core/agents/orchestrator.py`:

```python
"""BB8: Sequential agent orchestrator."""

from __future__ import annotations

from graphrag_core.agents.context import AgentContext
from graphrag_core.models import WorkflowResult


class SequentialOrchestrator:
    """Runs agents sequentially, stopping on first failure."""

    async def run_workflow(
        self,
        workflow_id: str,
        agents: list,
        context: AgentContext,
    ) -> WorkflowResult:
        agent_results = []

        for agent in agents:
            result = await agent.execute(context)
            agent_results.append(result)
            if not result.success:
                return WorkflowResult(
                    workflow_id=workflow_id,
                    success=False,
                    agent_results=agent_results,
                )

        return WorkflowResult(
            workflow_id=workflow_id,
            success=True,
            agent_results=agent_results,
        )
```

- [ ] **Step 4: Update agents __init__.py**

Replace `src/graphrag_core/agents/__init__.py`:

```python
"""BB8: Agent orchestration."""

from graphrag_core.agents.context import AgentContext
from graphrag_core.agents.orchestrator import SequentialOrchestrator

__all__ = ["AgentContext", "SequentialOrchestrator"]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_agents/test_orchestrator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 163 passed, 19 skipped

- [ ] **Step 7: Commit**

```bash
git add src/graphrag_core/agents/ tests/test_agents/
git commit -m "feat: add SequentialOrchestrator for agent workflows"
```

---

## Task 7: Update public API re-exports

**Files:**
- Modify: `src/graphrag_core/__init__.py`

- [ ] **Step 1: Update imports and __all__**

Read the file first, then make these changes:

Add to the interfaces import block:

```python
from graphrag_core.interfaces import (
    Agent,
    ApprovalGateway,
    Chunker,
    DetectionLayer,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    GraphStore,
    IngestionPipeline,
    LLMClient,
    LLMCurationLayer,
    Orchestrator,
    ReportRenderer,
    SearchEngine,
)
```

Add imports after the curation import:

```python
from graphrag_core.tools import Tool, ToolLibrary, register_core_tools
from graphrag_core.agents import AgentContext, SequentialOrchestrator
```

Add to the models import:

```python
from graphrag_core.models import (
    AgentResult,
    CurationIssue,
    CurationReport,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    ImportRun,
    KnownEntity,
    NodeTypeDefinition,
    OntologySchema,
    RegistryMatch,
    RenderConfig,
    ReportData,
    SearchResult,
    ToolParameter,
    ToolResult,
    WorkflowResult,
)
```

Add to `__all__`:

```python
    # Protocols (add new ones)
    "Agent",
    "Orchestrator",
    "ReportRenderer",
    # BB7 implementations
    "Tool",
    "ToolLibrary",
    "register_core_tools",
    # BB8 implementations
    "AgentContext",
    "SequentialOrchestrator",
    # Models (add new ones)
    "AgentResult",
    "RenderConfig",
    "ReportData",
    "ToolParameter",
    "ToolResult",
    "WorkflowResult",
```

- [ ] **Step 2: Verify imports**

Run: `uv run python -c "from graphrag_core import Agent, Orchestrator, ReportRenderer, Tool, ToolLibrary, register_core_tools, AgentContext, SequentialOrchestrator, ToolResult, AgentResult, WorkflowResult; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 163 passed, 19 skipped

- [ ] **Step 4: Commit**

```bash
git add src/graphrag_core/__init__.py
git commit -m "feat: add BB7 and BB8 re-exports to public API"
```

---

## Summary

| Task | What it builds | New tests |
|---|---|---|
| 1 | BB7 + BB8 models | 0 (models only) |
| 2 | Agent, Orchestrator, ReportRenderer Protocols | 3 conformance tests |
| 3 | ToolLibrary (register/get/list/execute) | 7 unit tests |
| 4 | 4 core tools + register_core_tools | 6 unit tests |
| 5 | AgentContext dataclass | 3 unit tests |
| 6 | SequentialOrchestrator | 5 unit tests |
| 7 | Public API re-exports | 0 (wiring) |

**Total new tests:** ~24
**Expected final count:** ~163 unit tests passing + 19 integration tests

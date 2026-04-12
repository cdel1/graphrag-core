# BB7 + BB8: Core Tool Library & Multi-Agent Orchestration

**Date:** 2026-04-12
**Status:** Approved
**Goal:** Implement BB7 (ToolLibrary + 4 domain-agnostic core tools) and BB8 (Agent/Orchestrator/ReportRenderer Protocols + SequentialOrchestrator), completing all 8 building blocks of graphrag-core.

---

## Decisions

| Decision | Rationale |
|---|---|
| Only 4 domain-agnostic core tools | `get_entity`, `search_entities`, `get_audit_trail`, `get_related`. The other 4 from the spec are domain-specific (quarters, topics, trends) and belong in Layer 2 repos. |
| BB8: Protocols + SequentialOrchestrator only | No LangGraph dependency. Domain repos bring their own orchestrator for branching/parallel workflows. |
| No ReportRenderer implementation | Rendering is format-specific (DOCX, PDF). No useful default without dependencies. Protocol only. |
| AgentContext is a dataclass, not Pydantic | Holds live service instances (GraphStore, ToolLibrary, SearchEngine). Not serializable. |
| AgentContext fields typed as `Any` | Avoids circular imports. Protocol conformance enforced at call site. |
| Tool is a dataclass, not Pydantic | Holds a callable handler which is not serializable. |
| ToolLibrary is a concrete class, not a Protocol | The spec defines it as a class. There's no reason to swap implementations — it's a simple dict-based registry. |
| Core tools use factory functions with closures | `make_get_entity_tool(graph_store)` returns a `Tool` with the handler bound. Avoids tools needing AgentContext at runtime. |

---

## Section 1: New Interfaces & Models

### New Protocols (added to `interfaces.py`)

```python
class Agent(Protocol):
    name: str
    async def execute(self, context: AgentContext) -> AgentResult: ...

class Orchestrator(Protocol):
    async def run_workflow(
        self, workflow_id: str, agents: list[Agent], context: AgentContext
    ) -> WorkflowResult: ...

class ReportRenderer(Protocol):
    async def render(
        self, report_data: ReportData, template: str, config: RenderConfig
    ) -> bytes: ...
```

### New Models (added to `models.py`)

**BB7:**

```python
class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True

class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None
```

**BB8:**

```python
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

**AgentContext** (dataclass in `agents/context.py`, not in `models.py`):

```python
@dataclass
class AgentContext:
    graph_store: Any       # GraphStore Protocol
    tool_library: Any      # ToolLibrary
    search_engine: Any     # SearchEngine Protocol
    workflow_state: dict[str, Any] = field(default_factory=dict)
```

---

## Section 2: BB7 — ToolLibrary and Core Tools

### `src/graphrag_core/tools/library.py`

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    handler: Callable[..., Awaitable[ToolResult]]

class ToolLibrary:
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def list_tools(self) -> list[Tool]: ...
    async def execute(self, name: str, **kwargs) -> ToolResult: ...
```

- `register()` — Stores tool by name. Raises `ValueError` on duplicate name.
- `get()` — Returns tool. Raises `KeyError` if not found.
- `list_tools()` — Returns all registered tools.
- `execute()` — Looks up tool, calls `handler(**kwargs)`, catches exceptions and returns `ToolResult(success=False, error=str(e))`.

### `src/graphrag_core/tools/core_tools.py`

4 factory functions, each returning a `Tool` with handler bound via closure:

1. **`make_get_entity_tool(graph_store)`** — Wraps `graph_store.get_node(entity_id)`. Returns node or error.
2. **`make_search_entities_tool(search_engine)`** — Wraps `search_engine.fulltext_search(query, node_types, top_k)`. Returns matches.
3. **`make_get_audit_trail_tool(graph_store)`** — Wraps `graph_store.get_audit_trail(node_id)`. Returns provenance chain.
4. **`make_get_related_tool(graph_store)`** — Wraps `graph_store.get_related(node_id, rel_type, depth)`. Returns connected nodes.

**`register_core_tools(library, graph_store, search_engine)`** — Convenience function that creates and registers all 4.

---

## Section 3: BB8 — SequentialOrchestrator

### `src/graphrag_core/agents/orchestrator.py`

```python
class SequentialOrchestrator:
    async def run_workflow(
        self, workflow_id: str, agents: list[Agent], context: AgentContext
    ) -> WorkflowResult:
```

Runs agents in order:
1. Call `agent.execute(context)`, collect `AgentResult`
2. If agent fails (`success=False`), stop workflow, return `WorkflowResult(success=False)`
3. After each agent, `context.workflow_state` is available to the next (shared state communication)
4. Return `WorkflowResult` with all results

~25 lines. No retry, no parallel, no branching.

### `src/graphrag_core/agents/context.py`

The `AgentContext` dataclass. Fields typed as `Any` to avoid circular imports.

---

## Testing

### BB7 Tests

**`tests/test_tools/test_library.py`:**
- register/get/list/execute
- Execute catches handler exceptions → `ToolResult(success=False)`
- Duplicate registration raises ValueError
- Get unknown tool raises KeyError

**`tests/test_tools/test_core_tools.py`:**
- `get_entity` returns node from InMemoryGraphStore
- `get_entity` returns error for missing node
- `search_entities` returns results from InMemorySearchEngine
- `get_audit_trail` returns provenance from InMemoryGraphStore
- `get_related` returns connected nodes
- `register_core_tools` registers all 4

### BB8 Tests

**`tests/test_agents/test_orchestrator.py`:**
- Runs agents in order
- Agents share workflow_state
- Stops on first failure
- WorkflowResult reflects success/failure
- Empty agent list returns success
- Protocol conformance

**`tests/test_agents/test_context.py`:**
- AgentContext holds live objects
- workflow_state defaults to empty dict

---

## Public API

**`src/graphrag_core/tools/__init__.py`:** Re-exports `Tool`, `ToolLibrary`, `register_core_tools`.

**`src/graphrag_core/agents/__init__.py`:** Re-exports `AgentContext`, `SequentialOrchestrator`.

**`src/graphrag_core/__init__.py`:** Add re-exports for new Protocols (`Agent`, `Orchestrator`, `ReportRenderer`), implementations (`ToolLibrary`, `SequentialOrchestrator`, `AgentContext`), and models (`ToolParameter`, `ToolResult`, `AgentResult`, `WorkflowResult`, `ReportData`, `RenderConfig`).

---

## New File Tree (additions only)

```
src/graphrag_core/
├── tools/
│   ├── __init__.py              # re-exports Tool, ToolLibrary, register_core_tools
│   ├── library.py               # Tool dataclass, ToolLibrary class
│   └── core_tools.py            # 4 core tool factories + register_core_tools
├── agents/
│   ├── __init__.py              # re-exports AgentContext, SequentialOrchestrator
│   ├── context.py               # AgentContext dataclass
│   └── orchestrator.py          # SequentialOrchestrator
tests/
├── test_tools/
│   ├── __init__.py
│   ├── test_library.py          # ToolLibrary unit tests
│   └── test_core_tools.py       # Core tool unit tests
├── test_agents/
│   ├── __init__.py
│   ├── test_orchestrator.py     # SequentialOrchestrator unit tests
│   └── test_context.py          # AgentContext unit tests
```

---

## Not Included (Deferred)

### Domain-specific tools (for Layer 2 repos)

These tools belong in `construction-intel` or `ey-graph-rag`, registered via `ToolLibrary.register()`:

| Tool | Signature | Where |
|---|---|---|
| `get_entity_history` | `(entity_id, from_q, to_q) -> Evolution` | Domain repo — requires quarter-based temporal model |
| `compare_quarters` | `(topic_id, q1, q2) -> Diff` | Domain repo — requires MonitoringTopic/quarter semantics |
| `find_unaddressed_topics` | `(quarter) -> list[Topic]` | Domain repo — requires domain-specific "addressed" definition |
| `find_trend` | `(topic_id) -> TrendVector` | Domain repo — requires temporal analysis over domain entities |

### Other deferrals

- `LangGraphOrchestrator` — Requires `langgraph` dependency. Domain repos implement when they need parallel/branching workflows.
- `ReportRenderer` concrete implementations — `DocxRenderer`, `PdfRenderer`, etc. Format-specific, requires `python-docx` or similar. Protocol defined this sprint.
- Retry/timeout logic in orchestrator — Comes with LangGraph or a custom implementation when needed.

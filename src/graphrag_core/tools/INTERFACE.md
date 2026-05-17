# `tools/` — INTERFACE (BB7)

**Classes:** `Tool`, `ToolLibrary` (data classes, not Protocols — but the *registration / invocation contract* is the interface here)
**Source:** [`graphrag_core/tools/library.py`](library.py), [`graphrag_core/tools/core_tools.py`](core_tools.py)
**Default implementations:** 4 of 8 core tools ship (`get_entity`, `search_entities`, `get_audit_trail`, `get_related`). 4 temporal tools (`get_entity_history`, `compare_periods`, `find_trend`, `find_unaddressed_topics`) live in Lacuna pending push-down (audit decision E1, before next PyPI release).
**Vocabulary:** `Tool`, `ToolParameter`, `ToolResult` — see `tessera/CONTEXT.md`

---

## `Tool`

A callable graph-query primitive exposed to agents (LLM tool-calling, MCP server, or programmatic). Each `Tool` is the unit of agent-callable capability.

### Shape

```python
class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    handler: Callable[..., Awaitable[ToolResult]]
```

### Contracts on a registered Tool

- **`name`** is unique within a `ToolLibrary`. Re-registering the same name overwrites.
- **`description`** is the *agent-facing* description. LLM tool-calling and MCP both surface this string verbatim; it must be precise and self-contained.
- **`parameters`** is a dict of `{param_name: ToolParameter}`. Each parameter has `name`, `type` (free-form string: `"string"`, `"integer"`, `"list[string]"`, etc.), `description`, `required: bool`.
- **`handler`** is async. Returns a `ToolResult` with `success: bool`, `data: Any` (must be JSON-serializable), optional `error: str`. **Never raises** — handler exceptions become `ToolResult(success=False, error=...)`. This is the contract that lets `ToolLibrary.execute()` be exception-safe.

### Tool-result data conventions

- On success: `data` is the Pydantic-dumped form of a model (e.g., `node.model_dump()`).
- On lists: `data` is `[model.model_dump() for model in result]`.
- On failure: `data is None`, `error` is a human-readable string.
- Never raw Pydantic instances in `data` — JSON-serializable only.

---

## `ToolLibrary`

Registry. Schema for agent discovery.

### Interface

```python
def register(self, tool: Tool) -> None: ...
def get(self, name: str) -> Tool: ...
def list_tools(self) -> list[Tool]: ...
async def execute(self, name: str, **kwargs) -> ToolResult: ...
```

### Contracts

- **`register`** — adds or overwrites. Idempotent on `name` (overwrite is intentional).
- **`get`** — raises `KeyError` on missing name.
- **`list_tools`** — returns all registered tools. Used by MCP server, LLM tool-calling clients, and CLI introspection.
- **`execute`** — looks up by `name`, validates `kwargs` against `tool.parameters` (missing required → `ToolResult(success=False, error=...)`, never raises), invokes `tool.handler(**kwargs)`. Wraps handler exceptions into `ToolResult`.

### Error contract

`execute()` is exception-safe by contract. The full failure surface is encoded in `ToolResult`:

| Failure | `success` | `data` | `error` |
|---|---|---|---|
| Tool not registered | False | None | `"tool 'X' not found"` |
| Missing required parameter | False | None | `"missing required parameter: 'foo'"` |
| Handler raised | False | None | `f"handler error: {exception}"` |
| Handler returned `ToolResult(success=False, ...)` | False | None | (forwarded from handler) |
| Success | True | (serialized model) | None |

This contract is **load-bearing for the MCP server**. The agent calling the tool must never see a raw Python exception; the failure must be a `ToolResult` it can reason about.

---

## Core tools (graphrag-core shipping today)

Registered by `register_core_tools(library, graph_store, search_engine)`:

| Name | Parameters | Returns |
|---|---|---|
| `get_entity` | `entity_id: string` | Node dump or error |
| `search_entities` | `query: string`, `node_types: list[string]` (optional), `top_k: int` (optional) | List of SearchResult dumps |
| `get_audit_trail` | `node_id: string` | AuditTrail dump |
| `get_related` | `node_id: string`, `rel_type: string` (optional), `depth: int` (optional) | List of node dumps |

## Pending push-down from Lacuna (audit E1)

Currently in `lacuna/intelligence/temporal.py` and `lacuna/intelligence/curation.py`; scheduled to move into graphrag-core BB7 before the next PyPI release:

| Name (Lacuna) | Push-down name (graphrag-core target) |
|---|---|
| `get_entity_history` | `get_entity_history` |
| `compare_periods` | `compare_periods` (renamed from spec's `compare_quarters` — generalization) |
| `find_trend` | `find_trend` |
| `find_unaddressed_topics` | `find_unaddressed_topics` |

When the push-down lands, this section becomes "Core tools (full set)" and the temporal-tool definitions move into a new `core_tools_temporal.py`.

## Domain tools (Lacuna-only, stay in Layer 2)

| Name | Where | Notes |
|---|---|---|
| `find_divergent_topics` | `lacuna/intelligence/curation.py` | Requires ASSERTS/SOURCED_FROM semantics — domain-specific |
| `generate_report_section` | `lacuna/report/lifecycle.py` | Calls `GenerationLLM`; domain templating |
| `generate_executive_summary` | `lacuna/report/lifecycle.py` | Same |
| `apply_lens` | `lacuna/report/lens.py` | Lens system is Phase 6c, domain rendering |

These do not push down; they're examples of legitimate Layer 2 tools.

---

## Implementation skeleton (new tool)

```python
def make_my_tool(some_dependency) -> Tool:
    async def handler(*, entity_id: str, optional_arg: int = 10) -> ToolResult:
        try:
            result = await some_dependency.do_thing(entity_id, optional_arg)
            if result is None:
                return ToolResult(success=False, error=f"nothing found for {entity_id}")
            return ToolResult(success=True, data=result.model_dump())
        except Exception as e:
            return ToolResult(success=False, error=f"handler error: {e}")

    return Tool(
        name="my_tool",
        description="Does X for entity Y. Returns Z.",
        parameters={
            "entity_id": ToolParameter(name="entity_id", type="string", description="The node ID", required=True),
            "optional_arg": ToolParameter(name="optional_arg", type="integer", description="...", required=False),
        },
        handler=handler,
    )

library.register(make_my_tool(some_dep))
```

### Test checklist

- Registered tool appears in `list_tools()`.
- `execute("missing_name")` → `ToolResult(success=False, error="tool 'missing_name' not found")`.
- `execute("known_name", **missing_required)` → `ToolResult(success=False, error="missing required parameter: ...")`.
- Handler exception → caught, wrapped as `ToolResult(success=False, error=...)`. Never propagated.
- `data` is JSON-serializable (round-trip through `json.dumps` succeeds).

# `tools/` — INTERFACE (BB7)

**Classes:** `Tool`, `ToolLibrary` (data classes, not Protocols — but the *registration / invocation contract* is the interface here)
**Source:** [`graphrag_core/tools/library.py`](library.py), [`graphrag_core/tools/core_tools.py`](core_tools.py)
**Default implementations:** 4 of 7 core tools ship (`get_entity`, `search_entities`, `get_audit_trail`, `get_related`). 3 temporal tools (`get_node_history`, `compare_periods`, `find_trend`) push down from Lacuna into graphrag-core BB7 in v0.6.0 (audit decision E1, scope-revised 2026-05-17 — see `tessera/docs/adr/0001-audit-trail-reaches-document-level.md`).
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
| `get_audit_trail` | `node_id: string` | ProvenanceTrail dump |
| `get_related` | `node_id: string`, `rel_type: string` (optional), `depth: int` (optional) | List of node dumps |

## Pending push-down from Lacuna (audit E1, scope-revised 2026-05-17)

Three temporal tools push down to graphrag-core BB7 in v0.6.0. They consume the document-level audit trail (`get_audit_trail(node_id) → ProvenanceTrail` with `level="document"` steps) for period resolution — no Lacuna-specific labels or edge names are hardcoded in graphrag-core. Implementation: new file `core_tools_temporal.py`, registered via `register_temporal_tools(library, graph_store)`.

| Name (graphrag-core v0.6.0) | Parameters | Returns | Lacuna origin |
|---|---|---|---|
| `get_node_history` | `node_id: string`, `rel_type: string?`, `from_period: string?`, `to_period: string?` | `dict[period → list[GraphNode]]` of the node's neighbors grouped by their resolved period | `lacuna/intelligence/temporal.get_entity_history` — renamed; `Entity` is a Lacuna Tier-1 label, graphrag-core operates on `GraphNode` |
| `compare_periods` | `node_id: string`, `period_from: string`, `period_to: string`, `rel_type: string?` | `PeriodDiff(added, removed)` — set-diff of neighbors by resolved period | `lacuna/intelligence/temporal.compare_periods` — already named correctly per audit's `compare_quarters → compare_periods` rename |
| `find_trend` | `node_id: string`, `rel_type: string?` | `TrendSignal(direction, claim_counts, periods_analyzed)` | `lacuna/intelligence/temporal.find_trend` |

The optional `rel_type` kwarg follows the existing BB7 pattern in `make_get_related_tool`. Domain consumers wanting "Claim-only history" pass `rel_type="ABOUT"` at the call site — the label name lives at Lacuna's call site, never in Layer 1.

### Lacuna-side aftermath

- `lacuna/intelligence/temporal.py` shrinks to a deprecation shim re-exporting the graphrag-core tools for one release, then deletes.
- `lacuna/intelligence/curation.py::find_unaddressed_topics` **stays in Lacuna** (originally part of audit E1; descoped 2026-05-17). It references `Topic` (Tier 3, human-curated) and `HAS_RECOMMENDATION` (Lacuna edge) — not domain-agnostic by the push-down test.

## Domain tools (Lacuna-only, stay in Layer 2)

| Name | Where | Notes |
|---|---|---|
| `find_unaddressed_topics` | `lacuna/intelligence/curation.py` | References Tier-3 `Topic` and Lacuna `HAS_RECOMMENDATION` edge — stays in Lacuna per 2026-05-17 scope revision |
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
- `register_temporal_tools(library, gs)` registers exactly three tools: `get_node_history`, `compare_periods`, `find_trend`.
- `get_node_history` groups a node's neighbors by their resolved source-document period.
- `compare_periods` returns `PeriodDiff(added, removed)` set-diff of neighbors between two periods.
- `find_trend` returns direction `∈ {"increasing", "decreasing", "stable", "insufficient_data"}` based on first-vs-last period neighbor counts.
- All BB7 temporal-tool handlers are exception-safe: internal errors are wrapped as `ToolResult(success=False, error=...)` instead of propagated.
- Missing-node case: temporal tools return `ToolResult(success=True, data={periods: {}})` or equivalent empty result — they do not raise.

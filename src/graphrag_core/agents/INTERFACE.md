# `agents/` — INTERFACE (BB8)

**Protocols:** `Agent`, `Orchestrator`, `ReportRenderer`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 244–268
**Default implementations:** [`SequentialOrchestrator`](orchestrator.py), [`AgentContext`](context.py). `ReportRenderer` is **Protocol-only** (no default impl shipped; v0.1.0 named `DocxRenderer` but not yet implemented).
**Vocabulary:** `Agent`, `Orchestrator`, `AgentResult`, `WorkflowResult`, `ReportData`, `RenderConfig` — see `tessera/CONTEXT.md`
**Important:** the v1.0 architecture spec's autonomous multi-agent workflow (Orchestrator/Analysis/Perspective/Report/Recommendation/QA Agents) was **superseded** by the TfT human-orchestrator model in `2026-05-01-lacuna-human-interface-design.md`. These Protocols remain in graphrag-core as a building block for future agent-orchestrated adapters (Phase 9+, see `2026-05-15-agentic-substrate-design.md`).

---

## `Agent`

A single agent with a defined role. Receives shared context, returns a result.

### Interface

```python
class Agent(Protocol):
    name: str
    async def execute(self, context: object) -> AgentResult: ...
```

### Contracts

- **`name`** is unique within an orchestrated workflow. The orchestrator records `name` in the `WorkflowResult` for traceability.
- **`execute`** returns `AgentResult` with `agent_name: str`, `success: bool`, `data: dict`, `error: str | None`. Never raises — failures become `success=False`. (Mirrors the `Tool` contract.)
- **`context`** is intentionally typed as `object` in the Protocol to allow different orchestrators to pass different context shapes. The default `Orchestrator` (Sequential) passes an `AgentContext` with `graph_store`, `tool_library`, `search_engine`, `period`, `workflow_state`. Concrete agents may rely on a specific context shape and document the dependency.

### Error mode

Failure surface is `AgentResult(success=False, error=...)`. The orchestrator decides whether to halt the workflow or continue. `SequentialOrchestrator` halts on first failure.

---

## `Orchestrator`

Coordinates a list of `Agent`s through a workflow.

### Interface

```python
class Orchestrator(Protocol):
    async def run_workflow(
        self,
        workflow_id: str,
        agents: list[Agent],
        context: object,
    ) -> WorkflowResult: ...
```

### Contracts

- **`workflow_id`** is caller-provided; recorded in `WorkflowResult` for audit.
- **`agents`** are run in caller-defined order (sequential) or implementation-defined order (e.g., DAG, parallel).
- **`WorkflowResult.success`** is `True` only if every agent's `AgentResult.success` is `True`.
- **Shared mutable context** (e.g., `AgentContext.workflow_state`) is the orchestrator's responsibility. The Sequential default exposes a shared dict; agents append to it.

### Reference impl

`SequentialOrchestrator` — runs agents in order, stops on first failure. 34 lines. The minimal orchestrator. Used today; richer orchestrators (LangGraph, parallel-DAG) named in v0.1.0 spec but not yet shipped.

---

## `ReportRenderer`

Renders structured `ReportData` into output bytes (DOCX, PDF, HTML, etc.).

### Interface

```python
class ReportRenderer(Protocol):
    async def render(
        self,
        report_data: ReportData,
        template: str,
        config: RenderConfig,
    ) -> bytes: ...
```

### Contracts

- **Returns raw bytes** of the target format. Caller writes / streams / serves.
- **`template`** is implementation-defined: a path, a template-engine string, a key into a template registry. The Protocol doesn't prescribe.
- **`config.format`** is the output format (`"docx"`, `"pdf"`, `"html"`, `"markdown"`). Implementations may support a subset.

### Status

Protocol shipped; no default implementation. v0.1.0 spec named `DocxRenderer` (Microsoft Word output); not yet implemented. Lacuna's Phase 6c report flow currently emits markdown directly without going through this Protocol (per `2026-05-02-phase6c-report-lifecycle-design.md` design decision — markdown templating is "good enough" for the current TfT adapter; full `ReportRenderer` lands when a customer needs DOCX or PDF).

---

## Implementation skeleton

### Custom Agent

```python
class MyAnalysisAgent:
    name = "my_analysis"

    async def execute(self, context):
        try:
            # 1. Pull from context (graph_store, tool_library, period, ...).
            # 2. Do analysis work.
            # 3. Return AgentResult.
            result_data = {"summary": "...", "findings": [...]}
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result_data,
                error=None,
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data={},
                error=str(e),
            )
```

### Custom Orchestrator (parallel)

```python
class ParallelOrchestrator:
    async def run_workflow(self, workflow_id, agents, context):
        results = await asyncio.gather(*[agent.execute(context) for agent in agents])
        return WorkflowResult(
            workflow_id=workflow_id,
            success=all(r.success for r in results),
            agent_results=results,
        )
```

### Test checklist

- Agent: `execute` never raises (always returns `AgentResult`).
- Agent: success path → `success=True`, `data` populated, `error=None`.
- Agent: failure path → `success=False`, `error` populated.
- Orchestrator: workflow_id preserved end-to-end.
- Orchestrator: any agent failure surfaces as `WorkflowResult.success=False`.
- `SequentialOrchestrator`: halts on first failure (subsequent agents not executed).

---

## Future direction (agent-orchestrated adapters, Phase 9+)

Per `2026-05-15-agentic-substrate-design.md` §6, the substrate already exposes everything needed for an autonomous adapter: tool library, audit chain, three-tier curation. When the TfT discipline matures and the MCP adapter is in production, the Phase 7 stage decomposition (`curate / outline / generate / editorial / lens / summary / finalize`) can be re-run as autonomous agents using these Protocols. The substrate stays put; the orchestrator changes from "human via CLI" to "autonomous + critic-agent loop with human approval at Tier 3."

# `curation/` — INTERFACE (BB5)

**Protocols:** `DetectionLayer`, `LLMCurationLayer`, `ApprovalGateway`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 196–220
**Default implementations:** Partial — `DetectionLayer` impl ([`detection.py`](detection.py)) ships, `LLMCurationLayer` and `ApprovalGateway` are **Protocol-only** (no default concrete class yet; see `repos/graphrag-core/CLAUDE.md` "Architecture" table).
**Vocabulary:** Tier 1/2/3 discipline, `CurationIssue`, `ApprovalBatch` — see `tessera/CONTEXT.md`

---

## Three-layer governance model

The curation pipeline is the quality gate between AI-generated graph content and trusted Tier-3 intelligence. Three layers, in this order:

1. **`DetectionLayer`** — deterministic, zero LLM cost. Catches duplicates, orphans, schema violations.
2. **`LLMCurationLayer`** — LLM-based suggestions. Entity resolution proposals, relevance scoring, candidate naming.
3. **`ApprovalGateway`** — human approval. The decisive filter. Mutations only land after a human signs off.

The pipeline is fully domain-agnostic — the same three Protocols apply to construction monitoring, transaction DD, compliance, anywhere.

---

## `DetectionLayer`

Deterministic, repeatable quality checks. Same input → same output. No LLM. No I/O outside the graph store.

### Interface

```python
async def detect(
    self,
    graph_store: GraphStore,
    schema: OntologySchema,
) -> list[CurationIssue]: ...
```

### Contracts

- **Deterministic.** Same graph state → same list of issues (with stable IDs).
- **Read-only.** Never mutates the graph. Proposes; never disposes.
- **Issue IDs are stable across runs.** A duplicate-detection issue between nodes A and B has the same ID on every run until A or B changes. Callers may dedupe issue lists by ID.
- Each `CurationIssue` carries: `issue_type` (`"duplicate"`, `"orphan"`, `"schema_violation"`, `"skipped_detection"`), `severity` (`"info"`, `"warning"`, `"error"`), `affected_nodes`, `suggested_action`, `auto_fixable`, `source_layer="deterministic"`.

### Standard detections

- **Duplicate detection** — node similarity (shared neighbors, name fuzzy match, embedding similarity).
- **Orphan detection** — Weakly Connected Components; flags nodes with no path to the main component.
- **Schema violations** — node has required property missing, edge has invalid source/target types.

### Error modes

- Empty graph → returns `[]`.
- Schema mismatch (graph doesn't conform to passed schema) → returns `CurationIssue`s of type `"schema_violation"`, does not raise.

### Performance invariants

- O(n) for orphan detection; O(n²) worst case for naive duplicate detection (use blocking / indexed search for production).
- Designed to run after every `ImportRun` *or* on demand. Implementations should support both.

---

## `LLMCurationLayer`

LLM-powered suggestions on top of deterministic issues. Optional; pipelines may run with `DetectionLayer + ApprovalGateway` only and skip this layer.

### Interface

```python
async def curate(self, issues: list[CurationIssue]) -> list[CurationIssue]: ...
```

### Contracts

- **Augments, doesn't replace.** Returns the input issues plus any new LLM-derived suggestions; or returns refined `suggested_action` on existing issues. Callers receive a single combined list.
- **`source_layer="llm"`** on LLM-added suggestions, distinct from `"deterministic"`.
- **Idempotent on stable issue IDs.** Calling `curate(curate(issues))` doesn't multiply suggestions.

### Typical augmentations

- Entity resolution: "Stakeholder 'J. Smith' and 'John Smith' look like the same person" — proposes merge.
- Topic naming: a `TopicCandidate` from community detection gets a proposed name.
- Relevance scoring: "this claim has no incoming relationships and looks like noise — propose dismiss."
- Merge/split proposals for `TopicCandidate`s.

### Error modes

- LLM provider failure → propagates; pipeline may continue with deterministic-only output.

### Performance invariants

- One LLM call per issue (or batched). Cost-bounded by the size of the deterministic output.

---

## `ApprovalGateway`

The decisive filter. Tier 3 mutations only land after a human approves.

### Interface

```python
async def submit_for_approval(self, issues: list[CurationIssue]) -> str: ...
async def get_approval_status(self, batch_id: str) -> ApprovalBatch: ...
async def apply_approved(self, batch_id: str) -> ApplyResult: ...
```

### Contracts

- **`submit_for_approval`** — accepts a list of curation issues, returns a `batch_id` for tracking. The implementation chooses the approval surface (CLI prompt, web UI, queue + webhook, file-based state).
- **`get_approval_status`** — returns an `ApprovalBatch` with `status: "pending" | "approved" | "rejected" | "partial"` and the per-issue decisions.
- **`apply_approved`** — applies all approved mutations to the graph. Returns `ApplyResult` (applied count, failed count, errors). Idempotent on `batch_id` — calling twice doesn't double-apply.
- **Audit trail required.** Every mutation must produce an immutable record (a `TopicEvent`, `ImportRun`, or equivalent) so the approval decision is reconstructable.

### Error modes

- Unknown `batch_id` → `get_approval_status` raises `KeyError`; `apply_approved` raises `KeyError`.
- Partial application failure → `ApplyResult.errors` is populated; does not raise.

### Performance invariants

- Implementation-specific. CLI gateway is per-issue interactive; webhook gateway is async.

### TfT integration

Lacuna's Phase 7 TfT workflow uses a file-based `ReportState` model (`repos/lacuna/src/lacuna/report/state.py`) as the operational layer of the approval pipeline for the *report flow* specifically. The `ApprovalGateway` Protocol covers a broader class of curation mutations (entity merges, topic creations, divergence confirmations) and remains unimplemented at the Protocol level — the TfT spec deliberately scoped down to file-based persistence for the report use case (`2026-05-03-phase7-tft-workflow-design.md` decision log).

---

## Implementation skeleton

```python
class MyDetector:
    async def detect(self, graph_store, schema):
        issues = []
        # 1. Run WCC on graph_store.list_nodes/list_relationships.
        # 2. For each orphan: append CurationIssue(issue_type="orphan", ...).
        # 3. Compute duplicate-candidate pairs.
        # 4. Validate schema.
        return issues

class MyLLMCurator:
    async def curate(self, issues):
        # 1. For each issue, ask LLM for suggested resolution.
        # 2. Append/refine, set source_layer="llm".
        return issues

class MyApprovalGateway:
    async def submit_for_approval(self, issues):
        batch_id = uuid()
        # Persist issues + batch_id somewhere durable.
        return batch_id
    async def get_approval_status(self, batch_id): ...
    async def apply_approved(self, batch_id): ...
```

### Test checklist

- Detector: determinism — same graph, same issues, same IDs.
- Detector: empty graph → `[]`.
- LLM curator: idempotency on rerun.
- Gateway: `apply_approved` is idempotent on batch_id (replay-safe).
- Gateway: rejected issues don't mutate the graph.
- Audit: every mutation has a corresponding TopicEvent or equivalent.

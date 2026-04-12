# BB5 + BB6: Governed Curation (Detection Layer) & Known Entity Registry

**Date:** 2026-04-12
**Status:** Approved
**Goal:** Implement the deterministic detection layer (BB5) and entity registry with fuzzy matching (BB6) to provide a quality gate for the knowledge graph.

---

## Decisions

| Decision | Rationale |
|---|---|
| BB5: Deterministic detection only this sprint | Foundation layer — finds the problems. LLM curation + human approval are Protocols only, plugged in later. |
| LLMCurationLayer + ApprovalGateway as Protocols | Defined in interfaces.py, no implementations. CurationPipeline accepts them optionally (None = skip). |
| BB6: Token normalization + SequenceMatcher for fuzzy matching | Stdlib only, no dependencies. Handles name reorderings, case differences, minor typos. |
| rapidfuzz deferred | Future upgrade: swap `fuzzy_score` implementation for `rapidfuzz.fuzz.token_sort_ratio`. One-function change, no interface impact. |
| Pairwise fallback capped at 1000 nodes per label | O(n^2) without registry. Cap prevents silent performance degradation; emits a warning CurationIssue when skipped. Registry path is O(n). |
| DetectionLayer.detect() takes graph_store + schema | More practical than import_run_id — layer needs to query graph and know schema directly. |

---

## Section 1: New Interfaces

### New Protocols (added to `interfaces.py`)

```python
class DetectionLayer(Protocol):
    async def detect(self, graph_store: GraphStore, schema: OntologySchema) -> list[CurationIssue]: ...

class LLMCurationLayer(Protocol):
    async def curate(self, issues: list[CurationIssue]) -> list[CurationIssue]: ...

class ApprovalGateway(Protocol):
    async def submit_for_approval(self, issues: list[CurationIssue]) -> str: ...
    async def get_approval_status(self, batch_id: str) -> ApprovalBatch: ...
    async def apply_approved(self, batch_id: str) -> ApplyResult: ...

class EntityRegistry(Protocol):
    async def register(self, entity: KnownEntity) -> str: ...
    async def lookup(self, name: str, entity_type: str, match_strategy: str = "fuzzy") -> list[RegistryMatch]: ...
    async def bulk_register(self, entities: list[KnownEntity]) -> int: ...
```

Note: `DetectionLayer.detect()` takes `graph_store` and `schema` as parameters, not `import_run_id`. The detection layer needs direct access to query the graph and validate against the schema.

### New Models (added to `models.py`)

```python
# BB5: Curation
class CurationIssue(BaseModel):
    id: str
    issue_type: str          # "duplicate", "orphan", "schema_violation", "skipped_detection"
    severity: str            # "info", "warning", "error"
    affected_nodes: list[str]
    suggested_action: str
    auto_fixable: bool
    source_layer: str        # "deterministic", "llm"

class CurationReport(BaseModel):
    issues: list[CurationIssue]
    nodes_scanned: int
    relationships_scanned: int

# BB5: Approval (models for Protocol, not implemented this sprint)
class ApprovalBatch(BaseModel):
    batch_id: str
    status: str              # "pending", "approved", "rejected", "partial"
    issues: list[CurationIssue]

class ApplyResult(BaseModel):
    batch_id: str
    applied: int
    failed: int
    errors: list[str]

# BB6: Entity Registry
class KnownEntity(BaseModel):
    name: str
    entity_type: str
    aliases: list[str] = []
    properties: dict[str, Any] = {}

class RegistryMatch(BaseModel):
    entity_id: str
    name: str
    score: float           # 0.0-1.0
    match_method: str      # "exact", "fuzzy", "embedding"
```

---

## Section 2: BB6 — InMemoryEntityRegistry

### File: `src/graphrag_core/registry/memory.py`

```python
class InMemoryEntityRegistry:
    def __init__(self) -> None:
```

Stores entities in a dict keyed by generated ID (`{entity_type}-{normalized_name}`).

**Methods:**

- `register()` — Stores a `KnownEntity`, returns its ID. If entity with same ID exists, merges aliases.
- `bulk_register()` — Calls `register()` for each entity, returns count of newly registered.
- `lookup()` — Three strategies:
  - `"exact"` — Direct name + alias match. Score 1.0.
  - `"fuzzy"` — Token-normalized `SequenceMatcher.ratio()`. Returns matches above 0.7 threshold. Checks name and all aliases.
  - `"embedding"` — Returns empty list (no embedding support in in-memory implementation).

### Fuzzy matching helper: `src/graphrag_core/registry/matching.py`

```python
def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, sort tokens."""

def fuzzy_score(a: str, b: str) -> float:
    """Token-normalized SequenceMatcher ratio."""
```

Extracted to own file so both registry and detection layer can import it.

### Testing

- `register` stores and returns ID
- `register` merges aliases on duplicate
- `bulk_register` returns correct count
- `lookup` exact match on name
- `lookup` exact match on alias
- `lookup` fuzzy catches reorderings ("Alice Smith" vs "Smith, Alice")
- `lookup` fuzzy catches case differences
- `lookup` fuzzy below threshold returns empty
- `lookup` embedding returns empty
- Protocol conformance

---

## Section 3: BB5 — DeterministicDetectionLayer

### File: `src/graphrag_core/curation/detection.py`

```python
class DeterministicDetectionLayer:
    def __init__(self, entity_registry: EntityRegistry | None = None) -> None:
```

Optional registry — works with or without BB6.

**Three detectors:**

1. **Duplicate detection** — With registry: for each node, lookup its name via registry. If multiple graph nodes match the same registry entity, flag as duplicates. Without registry: pairwise fuzzy comparison within each label group, capped at 1000 nodes per group. If cap exceeded, emit a `skipped_detection` warning issue.

2. **Orphan detection** — Finds nodes with no relationships in the graph.

3. **Schema violation detection** — Delegates to `GraphStore.validate_schema()`. Wraps `SchemaViolation` objects as `CurationIssue` objects.

**Performance characteristics:**
- With registry: O(n) per label group (single lookup per node)
- Without registry: O(n^2) per label group, capped at 1000 nodes. Groups exceeding cap emit a warning.

### Testing

- Detects duplicate nodes (same name, same label)
- Detects fuzzy duplicates ("Acme Corp" vs "ACME Corporation")
- Detects orphan nodes (no relationships)
- Detects schema violations (missing required properties)
- No false positives on valid graph
- Works without entity registry (pairwise fallback)
- Skips pairwise detection when label group exceeds 1000, emits warning
- Issues have correct `issue_type`, `severity`, `source_layer`

---

## Section 4: CurationPipeline

### File: `src/graphrag_core/curation/pipeline.py`

```python
class CurationPipeline:
    def __init__(
        self,
        detection: DetectionLayer,
        llm_curation: LLMCurationLayer | None = None,
        approval: ApprovalGateway | None = None,
    ) -> None:
```

**Flow:**
1. Call `detection.detect(graph_store, schema)` — always runs, returns `list[CurationIssue]`
2. If `llm_curation` is provided, pass issues to `llm_curation.curate(issues)` — skipped this sprint
3. If `approval` is provided, submit non-auto-fixable issues — skipped this sprint
4. Count nodes and relationships in graph_store for the report (iterating the store)
5. Return `CurationReport` with issues + scan counts

### Testing

- Pipeline runs detection and returns report
- Pipeline works without LLM/approval layers (None)
- Report contains correct issue counts

---

## Public API

**`src/graphrag_core/registry/__init__.py`:** Re-exports `InMemoryEntityRegistry`.

**`src/graphrag_core/curation/__init__.py`:** Re-exports `DeterministicDetectionLayer`, `CurationPipeline`.

**`src/graphrag_core/__init__.py`:** Add re-exports for `InMemoryEntityRegistry`, `DeterministicDetectionLayer`, `CurationPipeline`, plus new Protocols and models.

---

## New File Tree (additions only)

```
src/graphrag_core/
├── registry/
│   ├── __init__.py              # re-exports InMemoryEntityRegistry
│   ├── memory.py                # InMemoryEntityRegistry
│   └── matching.py              # normalize_name(), fuzzy_score()
├── curation/
│   ├── __init__.py              # re-exports DeterministicDetectionLayer, CurationPipeline
│   ├── detection.py             # DeterministicDetectionLayer
│   └── pipeline.py              # CurationPipeline
tests/
├── test_registry/
│   ├── __init__.py
│   ├── test_matching.py         # normalize_name, fuzzy_score unit tests
│   └── test_memory_registry.py  # InMemoryEntityRegistry unit tests
├── test_curation/
│   ├── __init__.py
│   ├── test_detection.py        # DeterministicDetectionLayer unit tests
│   └── test_pipeline.py         # CurationPipeline unit tests
```

---

## Not Included (Deferred)

- `LLMCurationLayer` concrete implementation (LLM-based entity resolution, merge suggestions) — Protocol defined this sprint
- `ApprovalGateway` concrete implementation (CLI or webhook-based review) — Protocol defined this sprint
- `rapidfuzz` for improved fuzzy matching — swap `fuzzy_score` in `matching.py` for `rapidfuzz.fuzz.token_sort_ratio`. Single-function change, no interface impact. Track in BB6 backlog.
- Embedding-based entity matching in EntityRegistry — requires default EmbeddingModel implementation
- Neo4j-backed EntityRegistry — in-memory is sufficient for now; Neo4j version follows when persistence is needed

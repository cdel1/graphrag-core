# `graph/` — INTERFACE (BB3 + Tier 2 detector)

**Protocols:** `GraphStore`, `CommunityDetector`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 131–164
**Default implementations:** [`Neo4jGraphStore`](neo4j.py), [`MemoryGraphStore`](memory.py) (for tests)
**Vocabulary:** Tier 1/2 ontology, `EXTRACTED_FROM`, `ASSERTS`, `ABOUT`, audit trail — see `tessera/CONTEXT.md`

---

## `GraphStore`

The provenance-native graph backend. Every node and every edge must be traceable to a source `ImportRun`.

### Interface

```python
async def merge_node(self, node: GraphNode, import_run_id: str) -> str: ...
async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str: ...
async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None: ...

async def get_node(self, node_id: str) -> GraphNode | None: ...
async def get_audit_trail(self, node_id: str) -> AuditTrail: ...
async def get_related(self, node_id: str, rel_type: str | None = None, depth: int = 1) -> list[GraphNode]: ...

async def apply_schema(self, schema: OntologySchema) -> None: ...
async def validate_schema(self) -> list[SchemaViolation]: ...

async def list_nodes(self) -> list[GraphNode]: ...
async def count_relationships(self) -> int: ...
async def list_relationships(self) -> list[GraphRelationship]: ...
```

### Contracts

- **`merge_node`** — Idempotent on `node.id`. If a node with this ID exists, properties are merged (last-write-wins on conflicts). Returns the canonical node ID (may differ if the implementation canonicalizes IDs). The `import_run_id` is recorded as provenance regardless of whether the node was newly created or merged.
- **`merge_relationship`** — Idempotent on `(source_id, target_id, type)`. If an edge with the same triple exists, properties are merged. `import_run_id` is recorded.
- **`record_provenance`** — Idempotent on `(node_id, chunk_id, import_run_id)`. Records the lineage edge `(:GraphNode)-[:EXTRACTED_FROM]->(:DocumentChunk)`.
- **`get_audit_trail`** — Must return the full provenance chain reaching from the node through its chunks to their source documents. Emits ordered `ProvenanceStep`s with `level ∈ {"node", "chunk", "document"}`. **The `level="node"` step is always first. Ordering of `chunk` and `document` steps within the chain is implementation-defined and varies by backend; consumers must filter by `step.level`, not by position.** Consumers (e.g., BB7 temporal tools) must filter the chain by `step.level`, not by position. The `level="document"` step carries `DocumentMetadata` fields in `metadata` (`title`, `source`, `doc_type`, `date`, `period`, `sha256`). If the node has no provenance, returns an `AuditTrail` with `provenance_chain=[]`, not `None`.
- **`get_related`** — `depth=1` returns immediate neighbors. `depth=2` includes neighbors-of-neighbors. The caller pays for traversal cost — depth >3 is a code smell.
- **`list_nodes` / `list_relationships`** — May be expensive on large graphs. Used by Tier 2 computations (community detection, divergence detection). Implementations should stream or paginate if backing store supports it; current Protocol returns full list — callers must accept O(n) memory.

### Error modes

- All methods may raise `ConnectionError` (transient) or `RuntimeError` (logic bug in caller).
- `apply_schema` raises `SchemaError` if the underlying store cannot enforce the constraints.
- `validate_schema` *never raises* — returns `list[SchemaViolation]` (empty if clean).
- `get_node` returns `None` for a missing ID — does not raise.

### Performance invariants

- `merge_node` / `merge_relationship`: O(1) amortized (with backing index).
- `get_node`: O(1).
- `get_related(depth=1)`: O(degree).
- `list_nodes` / `list_relationships`: O(n). Avoid in hot paths. Tier 2 computations call these once per detection run and cache the snapshot.
- `count_relationships`: O(1) if the backing store maintains a counter; O(n) otherwise. Used by tests; not a hot path.

### Idempotency / atomicity

- Per-call atomicity: a single `merge_node` call either fully writes or fully fails.
- Cross-call atomicity is **not** guaranteed — a multi-step pipeline that fails halfway leaves a partial graph. Idempotent re-runs are the recovery model.
- `ImportRun` records track which run produced which nodes/edges; recovery scripts can scope rollback to a single `import_run_id`.

### Reference impls

- `Neo4jGraphStore` — Cypher-based; uses `MERGE` semantics for idempotency; vector + fulltext indexes per `apply_schema`.
- `MemoryGraphStore` — dict-of-dicts; tests only; no schema enforcement.

---

## `CommunityDetector`

Tier 2 computation. Produces `Community` nodes from the Tier-1 graph for promotion to `TopicCandidate` (and ultimately Tier 3 `Topic`).

### Interface

```python
async def detect(self, graph_store: GraphStore) -> list[Community]: ...
```

### Contracts

- Takes a `GraphStore` (not a graph snapshot) — the detector pulls what it needs.
- Returns `list[Community]` — each `Community` has `id`, `node_ids`, `size`, `modularity_score`, `metadata`.
- **Pure function over the current graph state.** Two calls back-to-back with no graph mutations between them must produce equivalent communities (modulo non-determinism in the algorithm — see below).
- Implementations should support **hierarchical** detection (`level=0` leaves through `level=N` aggregates) by populating `metadata.level` and `metadata.parent_id`. Single-level implementations are valid but limit downstream UX.

### Error modes

- Empty graph → returns `[]`, does not raise.
- Disconnected components → each becomes its own community; that's expected, not an error.
- Algorithm failure (e.g., underlying library raises) → propagates the exception. Caller responsible for retry / fallback.

### Performance invariants

- O(E log V) for Leiden via graspologic.
- Allowed to materialize the full edge list in memory (call `graph_store.list_relationships()`).
- Caller may cache the result and recompute only on demand or after batch ingest.

### Non-determinism

Community detection algorithms (Leiden in particular) have stochastic tie-breaking. Implementations **should** seed their RNG and document the seed. Two calls in the same process should be deterministic; calls across processes may differ in community IDs but should agree on community *membership* up to relabeling.

### Reference impl

`LeidenCommunityDetector` lives in **Lacuna** (`lacuna/intelligence/communities.py`), not graphrag-core, because graspologic is a heavy dependency we don't want in Layer 1. The Protocol is in graphrag-core; the implementation is in Layer 2.

---

## Implementation skeleton

A minimal new `GraphStore` backend (sketch — adapt to backing store):

```python
class MyGraphStore:
    async def merge_node(self, node, import_run_id):
        # 1. Look up by node.id; if exists, merge properties; else create.
        # 2. Tag with import_run_id (back-edge or property).
        # 3. Return canonical ID.
        ...
    async def merge_relationship(self, rel, import_run_id): ...
    async def record_provenance(self, node_id, chunk_id, import_run_id): ...
    async def get_node(self, node_id): ...
    async def get_audit_trail(self, node_id):
        # Traverse: node -> chunks -> documents -> data sources.
        # Return AuditTrail(node_id, provenance_chain=[ProvenanceStep, ...]).
        ...
    async def get_related(self, node_id, rel_type=None, depth=1): ...
    async def apply_schema(self, schema): ...
    async def validate_schema(self): ...
    async def list_nodes(self): ...
    async def count_relationships(self): ...
    async def list_relationships(self): ...
```

### Test checklist

- `merge_node` is idempotent (call twice with same node, count = 1).
- `merge_relationship` is idempotent (call twice with same triple, count = 1).
- `get_audit_trail` returns the full chain for a node ingested via `IngestionPipeline`. The chain has at least one `level="document"` step; that step's `metadata` carries `period` (when set on the source document).
- `get_related(depth=1)` returns only direct neighbors.
- `list_nodes` returns every node ever merged (no filtering by import_run_id).
- `validate_schema` is idempotent and never raises.
- Empty store: `count_relationships() == 0`, `list_nodes() == []`.

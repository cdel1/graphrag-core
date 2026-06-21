# `search/` — INTERFACE (BB4)

**Protocol:** `SearchEngine`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 172–189
**Default implementations:** [`Neo4jHybridSearch`](neo4j.py), [`InMemorySearchEngine`](memory.py), [`fusion.py`](fusion.py) (RRF helper)
**Vocabulary:** `Chunk`, `SearchResult` — see `tessera/CONTEXT.md`
**Doctrine:** [`tessera/docs/adr/0032-bb4-hybrid-search-rrf-and-protocol-shape.md`](../../../../../tessera/docs/adr/0032-bb4-hybrid-search-rrf-and-protocol-shape.md) — RRF as canonical fusion, bi-modal hybrid, dual L1 impls.

---

## `SearchEngine`

Multi-modal search over the knowledge graph: vector similarity, fulltext (BM25 / Lucene), graph traversal, and a **bi-modal hybrid** combining vector + fulltext via Reciprocal Rank Fusion.

### Interface

```python
async def vector_search(
    self,
    query_embedding: list[float],
    top_k: int = 10,
    filters: dict | None = None,
) -> list[SearchResult]: ...

async def fulltext_search(
    self,
    query: str,
    node_types: list[str] | None = None,
    top_k: int = 10,
) -> list[SearchResult]: ...

async def graph_search(
    self,
    start_node_id: str,
    pattern: str,
    depth: int = 2,
) -> list[SearchResult]: ...

async def hybrid_search(
    self,
    query: str,
    embedding: list[float],
    top_k: int = 10,
    *,
    rrf_k: int = 60,
) -> list[SearchResult]: ...
```

### Contracts

- **`vector_search`** — Cosine similarity against the configured embedding index. `query_embedding` must match the stored vectors' dimensionality; **MUST raise `ValueError` on dim mismatch** (InMemorySearchEngine enforces this; Neo4j relies on the underlying index, which rejects mismatched dims with its own driver exception — see "Error modes" below). `filters` is a backend-specific dict; per-impl supported keys are documented below.
- **`fulltext_search`** — Keyword search. Implementations choose their own ranker (BM25 in InMemory; Lucene-backed in Neo4j). `node_types` restricts to specific labels. Filter is applied **before** the result-set limit (no post-LIMIT filtering).
- **`graph_search`** — Traversal from `start_node_id` following relationships matching `pattern`. `pattern` is a single relationship-type identifier (e.g., `"ABOUT"`, `"RELATES_TO"`) — not a full Cypher fragment. `depth` is the maximum traversal hops. Caller responsible for `pattern` being a valid identifier in the backend.
- **`hybrid_search`** — Combines **vector + fulltext** (only) via Reciprocal Rank Fusion. `graph_search` is intentionally **NOT** fused into hybrid: graph traversal needs a `start_node_id` that hybrid_search doesn't have, and graph scores (inverse hops) don't compose meaningfully with similarity/BM25 scores in RRF. Callers wanting tri-modal search should call `graph_search` separately and fuse results explicitly via `fusion.reciprocal_rank_fusion(...)`. The `rrf_k` parameter exposes the RRF k constant (default 60 per Cormack et al. 2009); raise it to flatten the rank-weighting curve.

### Contracts on `SearchResult`

- `node_id: str`, `label: str`, `score: float`, `source: str ("vector" | "fulltext" | "graph" | "hybrid")`, `properties: dict`.
- `score` semantics depend on `source`:
  - `"vector"`: cosine similarity in `[-1, 1]`. Implementations clamp negative scores to `0` only if documented (current impls do not).
  - `"fulltext"`: BM25 score (InMemory) or Lucene relevance score (Neo4j). **Unbounded above**, non-negative. Comparable within a single result set; **not comparable across engines or queries**.
  - `"graph"`: `1.0 / (hop_distance + 1)` style — inverse rank. Bounded `(0, 1]`.
  - `"hybrid"`: RRF score = `sum(1/(rrf_k + rank))` across input lists. **Unbounded above; not normalized to `[0, 1]`**. Only the *ordering* is meaningful; the absolute score is not.
- Results sorted by `score` descending. Ties broken by insertion order (implementation-defined).

### Filters per impl

| Impl | `vector_search.filters` keys | Notes |
|---|---|---|
| `Neo4jHybridSearch` | `label` (str) | Post-filters results by node label. Unknown keys silently ignored. |
| `InMemorySearchEngine` | `label` (str) | Same semantics. |

`fulltext_search.node_types` is honoured uniformly by both impls (in the Cypher `WHERE` clause for Neo4j; pre-filter for InMemory).

### Error modes

- Empty graph → all methods return `[]`, do not raise.
- `query_embedding` with wrong dimensionality → `InMemorySearchEngine` raises `ValueError`; `Neo4jHybridSearch` propagates the driver-level exception (Neo4j vector index rejects mismatched dims).
- `start_node_id` doesn't exist → `graph_search` returns `[]`, does not raise.
- Malformed `pattern` → backend-specific. Neo4j: `_cypher.validate_identifier` rejects non-identifiers with `ValueError`. InMemory: passes `pattern` through to `GraphStore.get_related` as `rel_type`.
- `InMemorySearchEngine.graph_search` without a `graph_store` constructor kwarg → returns `[]` (no error). To enable graph_search on InMemory, pass `graph_store=...` at construction.

### Performance invariants

- `vector_search`: O(log n) with HNSW or similar (Neo4j); O(n) linear scan (InMemory).
- `fulltext_search`: O(matched terms) with index (Neo4j); O(N_docs × |query_tokens|) BM25 scan (InMemory). BM25 corpus stats built once at construction; per-query work is linear in matched docs.
- `graph_search`: O(depth × avg degree).
- `hybrid_search`: `vector_search` and `fulltext_search` run **concurrently** via `asyncio.gather` (both impls). End-to-end latency is `max(vector, fulltext) + RRF`. RRF receives `top_k * 2` candidates from each engine to give fusion a wider pool.

### Reference impls

- **`Neo4jHybridSearch`** — Neo4j native vector index (HNSW) + fulltext index (Lucene) + Cypher traversal. RRF in [`fusion.py`](fusion.py). Index creation via `ensure_indexes(...)`.
- **`InMemorySearchEngine`** — Linear-scan cosine similarity + BM25 over property-value tokens + (optional) graph traversal via a held `GraphStore` reference. **Real BM25**, not substring matching — comparable to `rank-bm25` semantically. Use `graph_store=...` constructor kwarg to enable `graph_search`. Intended for tests AND lightweight demos / pilot deployments on small graphs (per ADR-0024's "default install resolves to InMemoryGraphStore" promise).

---

## Implementation skeleton

```python
class MySearchEngine:
    def __init__(self, store: MyGraphStore):
        self._store = store

    async def vector_search(self, query_embedding, top_k=10, filters=None):
        # 1. Validate query_embedding dimensionality → ValueError on mismatch.
        # 2. Query backing vector index.
        # 3. Map results to SearchResult, source="vector".
        ...

    async def fulltext_search(self, query, node_types=None, top_k=10):
        # 1. Apply node_types filter BEFORE top_k limit.
        # 2. Return SearchResult with source="fulltext".
        ...

    async def graph_search(self, start_node_id, pattern, depth=2): ...

    async def hybrid_search(self, query, embedding, top_k=10, *, rrf_k=60):
        import asyncio
        vec, ft = await asyncio.gather(
            self.vector_search(embedding, top_k=top_k * 2),
            self.fulltext_search(query, top_k=top_k * 2),
        )
        return reciprocal_rank_fusion([vec, ft], top_k=top_k, k=rrf_k)
```

### Test checklist

- All methods return `[]` on empty graph.
- `vector_search` raises `ValueError` on wrong-dimensionality embeddings (or propagates a driver exception with documented behaviour).
- `fulltext_search` with `node_types` filter does not return fewer than `top_k` results when matching results exist (filter must run before LIMIT).
- `hybrid_search` returns at most `top_k` results.
- `hybrid_search` runs vector and fulltext concurrently (observable by mock-instrumenting per-engine latency).
- RRF: a result appearing in *both* vector and fulltext outranks one appearing in only one.
- Hybrid scores are NOT in `[0, 1]` — INTERFACE consumers must use ordering, not absolute scores.

# `search/` — INTERFACE (BB4)

**Protocol:** `SearchEngine`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 171–189
**Default implementations:** [`Neo4jHybridSearch`](neo4j.py), [`MemorySearch`](memory.py) (tests), [`fusion.py`](fusion.py) (RRF helper)
**Vocabulary:** `DocumentChunk`, `SearchResult` — see `tessera/CONTEXT.md`

---

## `SearchEngine`

Multi-modal search over the knowledge graph: vector similarity, fulltext (keyword), graph traversal, and a hybrid combining all three.

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
) -> list[SearchResult]: ...
```

### Contracts

- **`vector_search`** — Cosine similarity against the configured embedding index. `query_embedding` must have the same dimensionality as the stored vectors. `filters` is a backend-specific dict (e.g. `{"node_type": "Claim", "period": "2026-Q2"}`); implementations document supported keys.
- **`fulltext_search`** — Keyword search over fulltext indexes (`claim_text`, `chunk_text`, `topic_name`, `entity_name`). `node_types` restricts to specific labels; `None` searches all indexed labels.
- **`graph_search`** — Cypher-pattern traversal from `start_node_id`. `pattern` is a backend-specific query fragment (e.g. Cypher `-[:ASSERTS]->`). Caller responsible for `pattern` validity; engine validates start node existence.
- **`hybrid_search`** — Combines vector + fulltext + (optional) graph results via Reciprocal Rank Fusion. `query` and `embedding` must reference the same underlying query (caller responsible).

### Contracts on `SearchResult`

- `node_id: str`, `label: str`, `score: float`, `source: str ("vector" | "fulltext" | "graph" | "hybrid")`, `properties: dict`.
- `score` is normalized to [0, 1] within each method's result set. *Cross-method comparisons are not meaningful* without going through `hybrid_search`'s RRF.
- Results sorted by `score` descending. Ties broken by insertion order (implementation-defined).

### Error modes

- Empty graph → all methods return `[]`, do not raise.
- `query_embedding` with wrong dimensionality → raise `ValueError`.
- `start_node_id` doesn't exist → `graph_search` returns `[]`, does not raise.
- Malformed `pattern` → backend-specific exception (e.g., Cypher syntax error from Neo4j) — propagated to caller.

### Performance invariants

- `vector_search`: O(log n) with HNSW or similar index; O(n) with linear scan.
- `fulltext_search`: O(matched terms); leverages backend fulltext index.
- `graph_search`: O(depth × avg degree); avoid `depth > 3` unless explicitly required.
- `hybrid_search`: sum of the three above; runs them concurrently when the backend allows.

### Reference impls

- `Neo4jHybridSearch` — uses Neo4j's native vector index + fulltext index + Cypher traversal. RRF in `fusion.py::reciprocal_rank_fusion`.
- `MemorySearch` — tests only. Linear scan; no real fulltext or vector index.

---

## Implementation skeleton

```python
class MySearchEngine:
    def __init__(self, store: MyGraphStore):
        self._store = store

    async def vector_search(self, query_embedding, top_k=10, filters=None):
        # 1. Validate query_embedding dimensionality.
        # 2. Query backing vector index.
        # 3. Map results to SearchResult, source="vector".
        ...

    async def fulltext_search(self, query, node_types=None, top_k=10): ...
    async def graph_search(self, start_node_id, pattern, depth=2): ...

    async def hybrid_search(self, query, embedding, top_k=10):
        vec = await self.vector_search(embedding, top_k=top_k*2)
        ft  = await self.fulltext_search(query, top_k=top_k*2)
        return reciprocal_rank_fusion([vec, ft], top_k=top_k)
```

### Test checklist

- All methods return `[]` on empty graph.
- `vector_search` rejects wrong-dimensionality embeddings.
- `hybrid_search` returns at most `top_k` results.
- RRF: a result appearing in *both* vector and fulltext outranks one appearing in only one.
- Result `score` always in [0, 1].

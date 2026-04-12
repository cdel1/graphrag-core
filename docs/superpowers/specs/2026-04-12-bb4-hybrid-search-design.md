# BB4: Hybrid Search Engine

**Date:** 2026-04-12
**Status:** Approved
**Goal:** Implement the SearchEngine Protocol with Neo4jHybridSearch, completing the core data path (ingest -> extract -> store -> search).

---

## Decisions

| Decision | Rationale |
|---|---|
| No interface or model changes | SearchEngine Protocol and SearchResult model already defined |
| Embeddings: Protocol only, no default | Tests use FakeEmbeddingModel. Search takes pre-computed embeddings as input. Real EmbeddingModel implementations deferred to when a domain repo needs one. |
| Reciprocal Rank Fusion (RRF) for hybrid search | Parameter-free, industry standard, no score normalization needed across different Neo4j index types |
| `ensure_indexes()` on Neo4jHybridSearch, not on GraphStore | Index lifecycle is a search concern. Avoids changing OntologySchema or bleeding BB4 into BB2/BB3. |
| InMemorySearchEngine for unit tests | Same pattern as InMemoryGraphStore. Satisfies Protocol without Neo4j. |
| RRF logic shared in fusion.py | Both Neo4j and in-memory implementations use the same fusion function |
| `hybrid_search` does NOT include `graph_search` | Graph traversal requires a start node, which hybrid search doesn't have |

---

## Section 1: No Interface Changes

The existing `SearchEngine` Protocol in `interfaces.py` already defines all 4 methods:

```python
class SearchEngine(Protocol):
    async def vector_search(self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None) -> list[SearchResult]: ...
    async def fulltext_search(self, query: str, node_types: list[str] | None = None, top_k: int = 10) -> list[SearchResult]: ...
    async def graph_search(self, start_node_id: str, pattern: str, depth: int = 2) -> list[SearchResult]: ...
    async def hybrid_search(self, query: str, embedding: list[float], top_k: int = 10) -> list[SearchResult]: ...
```

The `SearchResult` model already exists: `node_id`, `label`, `score`, `source`, `properties`.

No changes to `interfaces.py` or `models.py`.

---

## Section 2: Neo4jHybridSearch

### File: `src/graphrag_core/search/neo4j.py`

**Constructor:**

```python
class Neo4jHybridSearch:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "development"),
        database: str = "neo4j",
        vector_index_name: str = "chunk_embeddings",
        fulltext_index_name: str = "node_fulltext",
    ) -> None:
```

Same Neo4j connection params as `Neo4jGraphStore`, plus configurable index names.

**Methods:**

- `vector_search()` — Calls `db.index.vector.queryNodes` on the vector index. Returns results with `source="vector"`. Applies optional `filters` dict as `WHERE` clauses on node properties.
- `fulltext_search()` — Calls `db.index.fulltext.queryNodes` on the fulltext index. Optionally filters results by `node_types` labels. Returns results with `source="fulltext"`.
- `graph_search()` — Starts from `start_node_id`, traverses up to `depth` hops matching the Cypher `pattern`. The `pattern` parameter is interpreted as a relationship type name (e.g., `"WORKS_AT"`, `"KNOWS"`). Returns connected nodes with `source="graph"`. Scores based on inverse distance (1/hop_count, where hop_count is the shortest path length). Depth clamped to max 10. Pattern validated with `_validate_identifier`.
- `hybrid_search()` — Runs `vector_search` and `fulltext_search` sequentially, fuses results using RRF via `reciprocal_rank_fusion()`. Returns results with `source="hybrid"`.

Same Cypher safety pattern as `Neo4jGraphStore`: uses `_validate_identifier` for interpolated values, depth clamped to 10.

### `ensure_indexes()` method

```python
async def ensure_indexes(
    self,
    vector_dimensions: int = 1536,
    vector_node_label: str = "Chunk",
    vector_property: str = "embedding",
    fulltext_node_labels: list[str] | None = None,
    fulltext_properties: list[str] | None = None,
) -> None:
```

Creates:
1. Vector index on `(:<vector_node_label>).<vector_property>` with given dimensions
2. Fulltext index across specified node labels and properties (defaults to all provided labels on `name` property)

Both use `CREATE ... INDEX IF NOT EXISTS` — idempotent. Index names match the constructor's `vector_index_name` and `fulltext_index_name`.

---

## Section 3: RRF Fusion

### File: `src/graphrag_core/search/fusion.py`

Shared module-level function used by both `Neo4jHybridSearch` and `InMemorySearchEngine`:

```python
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 10,
    k: int = 60,
) -> list[SearchResult]:
```

For each result list, assigns rank 1..N. For each unique `node_id` across all lists, computes `score = sum(1 / (k + rank))`. Sorts by fused score descending, returns top_k results with `source="hybrid"`.

The `k=60` constant is the standard RRF parameter (same as used by Elasticsearch).

---

## Section 4: InMemorySearchEngine

### File: `src/graphrag_core/search/memory.py`

```python
class InMemorySearchEngine:
    def __init__(
        self,
        nodes: list[GraphNode],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
```

Dict-based implementation for unit tests:

- `vector_search()` — Cosine similarity between `query_embedding` and stored embeddings. Returns top-k sorted by score.
- `fulltext_search()` — Case-insensitive substring match on node property values. Scored by match quality.
- `graph_search()` — Returns empty list. Graph traversal requires real topology; the in-memory version satisfies the Protocol shape only.
- `hybrid_search()` — Runs `vector_search` and `fulltext_search`, fuses with the same `reciprocal_rank_fusion()` function.

---

## Section 5: Testing

### Unit tests (no Neo4j)

**`tests/test_search/test_fusion.py`:**
- RRF with two result lists — correct score ordering
- RRF with overlapping results — scores combine
- RRF with single list — passthrough
- RRF with empty lists — returns empty
- RRF respects top_k

**`tests/test_search/test_memory.py`:**
- `vector_search` returns nearest neighbors by cosine similarity
- `vector_search` respects top_k
- `fulltext_search` matches on node properties (case-insensitive)
- `fulltext_search` filters by node_types
- `graph_search` returns empty
- `hybrid_search` fuses vector + fulltext results
- Protocol conformance

### Integration tests (`@pytest.mark.integration`)

**`tests/test_search/test_neo4j_search.py`:**
- Fixture: creates nodes with embeddings, calls `ensure_indexes()`
- `vector_search` returns nearest neighbors from real vector index
- `fulltext_search` returns matches from real fulltext index
- `fulltext_search` filters by node_types
- `graph_search` traverses from a start node
- `hybrid_search` fuses real results with RRF
- `ensure_indexes` is idempotent
- Protocol conformance

---

## Public API Updates

**`src/graphrag_core/search/__init__.py`:** Re-exports `InMemorySearchEngine`. Conditional import for `Neo4jHybridSearch` (try/except, same as `Neo4jGraphStore`).

**`src/graphrag_core/__init__.py`:** Add `InMemorySearchEngine` to direct imports. Add `Neo4jHybridSearch` as optional (try/except).

---

## New File Tree (additions only)

```
src/graphrag_core/
├── search/
│   ├── __init__.py              # re-exports
│   ├── neo4j.py                 # Neo4jHybridSearch
│   ├── memory.py                # InMemorySearchEngine
│   └── fusion.py                # reciprocal_rank_fusion()
tests/
├── test_search/
│   ├── __init__.py
│   ├── test_fusion.py           # RRF unit tests
│   ├── test_memory.py           # InMemorySearchEngine unit tests
│   └── test_neo4j_search.py     # @integration, real Neo4j
```

---

## Not Included (Deferred)

- Default `EmbeddingModel` implementation (Voyage, OpenAI, etc.)
- Configurable fusion strategies (weighted linear, etc.)
- Semantic/re-ranking on search results
- BB5-BB8 interfaces and implementations

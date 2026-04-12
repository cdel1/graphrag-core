# BB4: Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the SearchEngine Protocol with Neo4jHybridSearch, InMemorySearchEngine, and RRF fusion, completing the core data path (ingest -> extract -> store -> search).

**Architecture:** No interface/model changes needed — SearchEngine Protocol and SearchResult model already exist. RRF fusion is a shared module. Neo4jHybridSearch queries vector, fulltext, and graph indexes. InMemorySearchEngine provides fast unit-test support. `_validate_identifier` is extracted to a shared utility to avoid duplication.

**Tech Stack:** Python 3.12+, Pydantic v2, neo4j async driver, pytest + pytest-asyncio

**Design spec:** `docs/superpowers/specs/2026-04-12-bb4-hybrid-search-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `src/graphrag_core/_cypher.py` | Shared Cypher safety utilities (`_validate_identifier`, `_MAX_DEPTH`) |
| `src/graphrag_core/search/__init__.py` | Re-exports `InMemorySearchEngine`, conditional `Neo4jHybridSearch` |
| `src/graphrag_core/search/fusion.py` | `reciprocal_rank_fusion()` — shared RRF implementation |
| `src/graphrag_core/search/memory.py` | `InMemorySearchEngine` — dict-based SearchEngine for tests |
| `src/graphrag_core/search/neo4j.py` | `Neo4jHybridSearch` — Neo4j vector/fulltext/graph search + ensure_indexes |
| `tests/test_search/__init__.py` | Package marker |
| `tests/test_search/test_fusion.py` | RRF unit tests |
| `tests/test_search/test_memory.py` | InMemorySearchEngine unit tests |
| `tests/test_search/test_neo4j_search.py` | Neo4jHybridSearch integration tests |

### Modified files

| File | Change |
|---|---|
| `src/graphrag_core/graph/neo4j.py` | Import `_validate_identifier`, `_MAX_DEPTH` from `_cypher.py` instead of defining locally |
| `src/graphrag_core/__init__.py` | Add search re-exports |

---

## Task 1: Extract Cypher safety utilities to shared module

**Files:**
- Create: `src/graphrag_core/_cypher.py`
- Modify: `src/graphrag_core/graph/neo4j.py`

- [ ] **Step 1: Create the shared module**

Create `src/graphrag_core/_cypher.py`:

```python
"""Shared Cypher safety utilities."""

from __future__ import annotations

import re

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_DEPTH = 10


def validate_identifier(value: str, kind: str) -> str:
    """Reject identifiers that could cause Cypher injection."""
    if not SAFE_IDENTIFIER.match(value):
        raise ValueError(f"Invalid {kind}: {value!r}")
    return value
```

- [ ] **Step 2: Update graph/neo4j.py to import from shared module**

In `src/graphrag_core/graph/neo4j.py`, replace the local definitions:

Remove these lines (lines 5-18):
```python
import re
...
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_DEPTH = 10


def _validate_identifier(value: str, kind: str) -> str:
    """Reject identifiers that could cause Cypher injection."""
    if not _SAFE_IDENTIFIER.match(value):
        raise ValueError(f"Invalid {kind}: {value!r}")
    return value
```

Add this import after the `from __future__` line:
```python
from graphrag_core._cypher import MAX_DEPTH, validate_identifier
```

Then replace all occurrences in the file:
- `_validate_identifier(` → `validate_identifier(`
- `_MAX_DEPTH` → `MAX_DEPTH`

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: 83 passed, 11 skipped

- [ ] **Step 4: Commit**

```bash
git add src/graphrag_core/_cypher.py src/graphrag_core/graph/neo4j.py
git commit -m "refactor: extract Cypher safety utilities to shared _cypher module"
```

---

## Task 2: RRF fusion function

**Files:**
- Create: `tests/test_search/__init__.py`
- Create: `tests/test_search/test_fusion.py`
- Create: `src/graphrag_core/search/__init__.py`
- Create: `src/graphrag_core/search/fusion.py`

- [ ] **Step 1: Write failing tests for RRF**

Create `tests/test_search/__init__.py` (empty).

Create `tests/test_search/test_fusion.py`:

```python
"""Tests for reciprocal rank fusion."""

from __future__ import annotations

from graphrag_core.models import SearchResult


def _result(node_id: str, score: float = 0.0, source: str = "test") -> SearchResult:
    return SearchResult(node_id=node_id, label=node_id, score=score, source=source)


class TestReciprocalRankFusion:
    def test_fuses_two_lists(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        list_a = [_result("n1"), _result("n2"), _result("n3")]
        list_b = [_result("n2"), _result("n1"), _result("n4")]

        results = reciprocal_rank_fusion([list_a, list_b], top_k=10)

        # n1 is rank 1 in list_a (1/61) and rank 2 in list_b (1/62) = 0.03248
        # n2 is rank 2 in list_a (1/62) and rank 1 in list_b (1/61) = 0.03248
        # Both n1 and n2 should have the same score (appear in both lists)
        ids = [r.node_id for r in results]
        assert "n1" in ids
        assert "n2" in ids
        assert "n3" in ids
        assert "n4" in ids
        # n1 and n2 should be ranked above n3 and n4 (appear in both lists vs one)
        n1_idx = ids.index("n1")
        n3_idx = ids.index("n3")
        assert n1_idx < n3_idx

    def test_overlapping_results_score_higher(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        list_a = [_result("n1"), _result("n2")]
        list_b = [_result("n1"), _result("n3")]

        results = reciprocal_rank_fusion([list_a, list_b], top_k=10)

        # n1 appears in both lists, should be ranked first
        assert results[0].node_id == "n1"

    def test_single_list_passthrough(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        items = [_result("n1"), _result("n2"), _result("n3")]
        results = reciprocal_rank_fusion([items], top_k=10)

        assert len(results) == 3
        assert results[0].node_id == "n1"
        assert results[1].node_id == "n2"
        assert results[2].node_id == "n3"

    def test_empty_lists_returns_empty(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion([], top_k=10)
        assert results == []

        results = reciprocal_rank_fusion([[], []], top_k=10)
        assert results == []

    def test_respects_top_k(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        items = [_result(f"n{i}") for i in range(20)]
        results = reciprocal_rank_fusion([items], top_k=5)

        assert len(results) == 5

    def test_result_source_is_hybrid(self):
        from graphrag_core.search.fusion import reciprocal_rank_fusion

        results = reciprocal_rank_fusion(
            [[_result("n1", source="vector")], [_result("n1", source="fulltext")]],
            top_k=10,
        )
        assert results[0].source == "hybrid"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graphrag_core.search'`

- [ ] **Step 3: Implement RRF fusion**

Create `src/graphrag_core/search/__init__.py`:

```python
"""BB4: Search engine implementations."""

__all__: list[str] = []
```

Create `src/graphrag_core/search/fusion.py`:

```python
"""Reciprocal Rank Fusion for combining search results."""

from __future__ import annotations

from graphrag_core.models import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 10,
    k: int = 60,
) -> list[SearchResult]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Lists of SearchResult, each ordered by relevance (best first).
        top_k: Maximum number of results to return.
        k: RRF constant (default 60, standard value).

    Returns:
        Fused results sorted by combined RRF score, with source="hybrid".
    """
    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            scores[result.node_id] = scores.get(result.node_id, 0.0) + 1.0 / (k + rank)
            if result.node_id not in best_result:
                best_result[result.node_id] = result

    sorted_ids = sorted(scores, key=lambda nid: scores[nid], reverse=True)[:top_k]

    return [
        SearchResult(
            node_id=nid,
            label=best_result[nid].label,
            score=scores[nid],
            source="hybrid",
            properties=best_result[nid].properties,
        )
        for nid in sorted_ids
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search/test_fusion.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 89 passed, 11 skipped

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/search/ tests/test_search/
git commit -m "feat: add reciprocal rank fusion for hybrid search"
```

---

## Task 3: InMemorySearchEngine — vector search

**Files:**
- Create: `tests/test_search/test_memory.py`
- Create: `src/graphrag_core/search/memory.py`

- [ ] **Step 1: Write failing tests for vector search**

Create `tests/test_search/test_memory.py`:

```python
"""Tests for InMemorySearchEngine."""

from __future__ import annotations

import math

import pytest

from graphrag_core.models import GraphNode


def _nodes() -> list[GraphNode]:
    return [
        GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}),
        GraphNode(id="n2", label="Company", properties={"name": "Globex Inc"}),
        GraphNode(id="n3", label="Person", properties={"name": "Alice"}),
    ]


def _embeddings() -> dict[str, list[float]]:
    return {
        "n1": [1.0, 0.0, 0.0],
        "n2": [0.0, 1.0, 0.0],
        "n3": [0.7, 0.7, 0.0],
    }


class TestInMemoryVectorSearch:
    @pytest.mark.asyncio
    async def test_returns_nearest_by_cosine(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.vector_search(query_embedding=[1.0, 0.0, 0.0], top_k=3)

        assert len(results) == 3
        assert results[0].node_id == "n1"
        assert results[0].source == "vector"
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.vector_search(query_embedding=[1.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0].node_id == "n1"

    @pytest.mark.asyncio
    async def test_no_embeddings_returns_empty(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.vector_search(query_embedding=[1.0, 0.0, 0.0])

        assert results == []

    @pytest.mark.asyncio
    async def test_filters_by_property(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.vector_search(
            query_embedding=[0.7, 0.7, 0.0], top_k=10, filters={"label": "Company"}
        )

        assert all(r.label == "Company" for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search/test_memory.py::TestInMemoryVectorSearch -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graphrag_core.search.memory'`

- [ ] **Step 3: Implement InMemorySearchEngine with vector search**

Create `src/graphrag_core/search/memory.py`:

```python
"""In-memory SearchEngine implementation for testing."""

from __future__ import annotations

import math

from graphrag_core.models import GraphNode, SearchResult
from graphrag_core.search.fusion import reciprocal_rank_fusion


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemorySearchEngine:
    """Dict-based SearchEngine for unit tests."""

    def __init__(
        self,
        nodes: list[GraphNode],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self._nodes = {n.id: n for n in nodes}
        self._embeddings = embeddings or {}

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if not self._embeddings:
            return []

        scored: list[tuple[str, float]] = []
        for node_id, emb in self._embeddings.items():
            node = self._nodes.get(node_id)
            if node is None:
                continue
            if filters and filters.get("label") and node.label != filters["label"]:
                continue
            score = _cosine_similarity(query_embedding, emb)
            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                node_id=nid,
                label=self._nodes[nid].label,
                score=score,
                source="vector",
                properties=self._nodes[nid].properties,
            )
            for nid, score in scored[:top_k]
        ]

    async def fulltext_search(
        self, query: str, node_types: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        return []

    async def graph_search(
        self, start_node_id: str, pattern: str, depth: int = 2
    ) -> list[SearchResult]:
        return []

    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search/test_memory.py::TestInMemoryVectorSearch -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 93 passed, 11 skipped

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/search/memory.py tests/test_search/test_memory.py
git commit -m "feat: add InMemorySearchEngine with vector search"
```

---

## Task 4: InMemorySearchEngine — fulltext and hybrid search

**Files:**
- Modify: `tests/test_search/test_memory.py`
- Modify: `src/graphrag_core/search/memory.py`

- [ ] **Step 1: Write tests for fulltext, graph, and hybrid search**

Add to `tests/test_search/test_memory.py`:

```python
class TestInMemoryFulltextSearch:
    @pytest.mark.asyncio
    async def test_matches_on_property_values(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="acme", top_k=10)

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "fulltext"

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="ALICE", top_k=10)

        assert len(results) >= 1
        assert results[0].node_id == "n3"

    @pytest.mark.asyncio
    async def test_filters_by_node_types(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        nodes = _nodes()
        engine = InMemorySearchEngine(nodes=nodes)
        results = await engine.fulltext_search(query="a", node_types=["Person"], top_k=10)

        assert all(r.label == "Person" for r in results)

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="zzzznonexistent", top_k=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.fulltext_search(query="a", top_k=1)

        assert len(results) == 1


class TestInMemoryGraphSearch:
    @pytest.mark.asyncio
    async def test_returns_empty(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes())
        results = await engine.graph_search(start_node_id="n1", pattern="WORKS_AT")

        assert results == []


class TestInMemoryHybridSearch:
    @pytest.mark.asyncio
    async def test_fuses_vector_and_fulltext(self):
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=_nodes(), embeddings=_embeddings())
        results = await engine.hybrid_search(
            query="Acme", embedding=[1.0, 0.0, 0.0], top_k=10
        )

        assert len(results) >= 1
        assert results[0].source == "hybrid"
        # n1 should be top result: best vector match AND fulltext match for "Acme"
        assert results[0].node_id == "n1"
```

- [ ] **Step 2: Implement fulltext and hybrid search**

Update `src/graphrag_core/search/memory.py`, replacing the stub methods:

Replace the `fulltext_search` method:

```python
    async def fulltext_search(
        self, query: str, node_types: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        query_lower = query.lower()
        scored: list[tuple[str, float]] = []

        for node in self._nodes.values():
            if node_types and node.label not in node_types:
                continue
            searchable = " ".join(str(v) for v in node.properties.values()).lower()
            if query_lower in searchable:
                # Score: exact match on a value > substring match
                best_score = 0.0
                for val in node.properties.values():
                    val_lower = str(val).lower()
                    if val_lower == query_lower:
                        best_score = max(best_score, 1.0)
                    elif query_lower in val_lower:
                        best_score = max(best_score, len(query_lower) / len(val_lower))
                scored.append((node.id, best_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                node_id=nid,
                label=self._nodes[nid].label,
                score=score,
                source="fulltext",
                properties=self._nodes[nid].properties,
            )
            for nid, score in scored[:top_k]
        ]
```

Replace the `hybrid_search` method:

```python
    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]:
        vector_results = await self.vector_search(query_embedding=embedding, top_k=top_k)
        fulltext_results = await self.fulltext_search(query=query, top_k=top_k)

        return reciprocal_rank_fusion([vector_results, fulltext_results], top_k=top_k)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_search/test_memory.py -v`
Expected: All 10 tests PASS (4 vector + 5 fulltext + 1 graph + 0... wait, let me count: 4 vector + 5 fulltext + 1 graph + 1 hybrid = 11)

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 100 passed, 11 skipped

- [ ] **Step 5: Commit**

```bash
git add src/graphrag_core/search/memory.py tests/test_search/test_memory.py
git commit -m "feat: add fulltext and hybrid search to InMemorySearchEngine"
```

---

## Task 5: InMemorySearchEngine Protocol conformance

**Files:**
- Modify: `tests/test_search/test_memory.py`

- [ ] **Step 1: Write Protocol conformance test**

Add to `tests/test_search/test_memory.py`:

```python
class TestInMemorySearchEngineProtocol:
    def test_satisfies_search_engine_protocol(self):
        from graphrag_core.interfaces import SearchEngine
        from graphrag_core.search.memory import InMemorySearchEngine

        engine = InMemorySearchEngine(nodes=[])
        assert isinstance(engine, SearchEngine)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_search/test_memory.py::TestInMemorySearchEngineProtocol -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_search/test_memory.py
git commit -m "test: add InMemorySearchEngine Protocol conformance test"
```

---

## Task 6: Neo4jHybridSearch implementation

**Files:**
- Create: `src/graphrag_core/search/neo4j.py`
- Modify: `src/graphrag_core/search/__init__.py`

- [ ] **Step 1: Implement Neo4jHybridSearch**

Create `src/graphrag_core/search/neo4j.py`:

```python
"""BB4: Neo4j-backed hybrid search engine."""

from __future__ import annotations

from neo4j import AsyncGraphDatabase

from graphrag_core._cypher import MAX_DEPTH, validate_identifier
from graphrag_core.models import SearchResult
from graphrag_core.search.fusion import reciprocal_rank_fusion


class Neo4jHybridSearch:
    """Neo4j async implementation of the SearchEngine Protocol."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "development"),
        database: str = "neo4j",
        vector_index_name: str = "chunk_embeddings",
        fulltext_index_name: str = "node_fulltext",
    ) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self._database = database
        self._vector_index_name = vector_index_name
        self._fulltext_index_name = fulltext_index_name

    async def close(self) -> None:
        await self._driver.close()

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        query = (
            "CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node, score, labels(node) AS labels "
            "ORDER BY score DESC"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                index_name=self._vector_index_name,
                top_k=top_k,
                embedding=query_embedding,
            )
            results = []
            async for record in result:
                props = dict(record["node"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                props.pop("embedding", None)

                if filters:
                    if filters.get("label") and label != filters["label"]:
                        continue

                results.append(SearchResult(
                    node_id=node_id,
                    label=label,
                    score=record["score"],
                    source="vector",
                    properties=props,
                ))
            return results[:top_k]

    async def fulltext_search(
        self, query: str, node_types: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        cypher = (
            "CALL db.index.fulltext.queryNodes($index_name, $query) "
            "YIELD node, score "
            "RETURN node, score, labels(node) AS labels "
            "ORDER BY score DESC "
            "LIMIT $top_k"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                index_name=self._fulltext_index_name,
                query=query,
                top_k=top_k,
            )
            results = []
            async for record in result:
                props = dict(record["node"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                props.pop("embedding", None)

                if node_types and label not in node_types:
                    continue

                results.append(SearchResult(
                    node_id=node_id,
                    label=label,
                    score=record["score"],
                    source="fulltext",
                    properties=props,
                ))
            return results

    async def graph_search(
        self, start_node_id: str, pattern: str, depth: int = 2
    ) -> list[SearchResult]:
        depth = min(max(depth, 1), MAX_DEPTH)
        validate_identifier(pattern, "relationship type")
        cypher = (
            f"MATCH (start {{id: $start_id}})-[:{pattern}*1..{depth}]-(m) "
            "WHERE m.id <> $start_id "
            "WITH DISTINCT m, labels(m) AS labels, "
            f"  length(shortestPath((start)-[:{pattern}*]-(m))) AS hops "
            "MATCH (start {id: $start_id}) "
            "RETURN m, labels, hops "
            "ORDER BY hops ASC"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, start_id=start_node_id)
            results = []
            async for record in result:
                props = dict(record["m"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                hops = record["hops"]
                score = 1.0 / hops if hops > 0 else 1.0
                results.append(SearchResult(
                    node_id=node_id,
                    label=label,
                    score=score,
                    source="graph",
                    properties=props,
                ))
            return results

    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]:
        vector_results = await self.vector_search(query_embedding=embedding, top_k=top_k)
        fulltext_results = await self.fulltext_search(query=query, top_k=top_k)

        return reciprocal_rank_fusion([vector_results, fulltext_results], top_k=top_k)

    async def ensure_indexes(
        self,
        vector_dimensions: int = 1536,
        vector_node_label: str = "Chunk",
        vector_property: str = "embedding",
        fulltext_node_labels: list[str] | None = None,
        fulltext_properties: list[str] | None = None,
    ) -> None:
        validate_identifier(vector_node_label, "node label")
        validate_identifier(vector_property, "property name")

        fulltext_labels = fulltext_node_labels or [vector_node_label]
        fulltext_props = fulltext_properties or ["name"]

        for lbl in fulltext_labels:
            validate_identifier(lbl, "node label")
        for prop in fulltext_props:
            validate_identifier(prop, "property name")

        async with self._driver.session(database=self._database) as session:
            vector_cypher = (
                f"CREATE VECTOR INDEX {self._vector_index_name} IF NOT EXISTS "
                f"FOR (n:{vector_node_label}) ON (n.{vector_property}) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dimensions, "
                "`vector.similarity_function`: 'cosine'}}"
            )
            await session.run(vector_cypher, dimensions=vector_dimensions)

            labels_str = "|".join(fulltext_labels)
            props_str = ", ".join(f"n.{p}" for p in fulltext_props)
            fulltext_cypher = (
                f"CREATE FULLTEXT INDEX {self._fulltext_index_name} IF NOT EXISTS "
                f"FOR (n:{labels_str}) ON EACH [{props_str}]"
            )
            await session.run(fulltext_cypher)
```

- [ ] **Step 2: Update search __init__.py**

Replace `src/graphrag_core/search/__init__.py` with:

```python
"""BB4: Search engine implementations."""

from graphrag_core.search.memory import InMemorySearchEngine

__all__ = ["InMemorySearchEngine"]

try:
    from graphrag_core.search.neo4j import Neo4jHybridSearch
    __all__.append("Neo4jHybridSearch")
except ImportError:
    pass
```

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from graphrag_core.search import Neo4jHybridSearch, InMemorySearchEngine; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 101 passed, 11 skipped

- [ ] **Step 5: Commit**

```bash
git add src/graphrag_core/search/
git commit -m "feat: add Neo4jHybridSearch implementing SearchEngine Protocol"
```

---

## Task 7: Neo4jHybridSearch integration tests

**Files:**
- Create: `tests/test_search/test_neo4j_search.py`

- [ ] **Step 1: Write integration tests**

Create `tests/test_search/test_neo4j_search.py`:

```python
"""Integration tests for Neo4jHybridSearch. Requires running Neo4j."""

from __future__ import annotations

import os

import pytest

from graphrag_core.models import GraphNode

pytestmark = pytest.mark.integration

NEO4J_TEST_DB = os.environ.get("NEO4J_TEST_DATABASE", "neo4j")


@pytest.fixture
async def search_engine():
    from graphrag_core.graph.neo4j import Neo4jGraphStore
    from graphrag_core.search.neo4j import Neo4jHybridSearch

    store = Neo4jGraphStore(database=NEO4J_TEST_DB)
    engine = Neo4jHybridSearch(
        database=NEO4J_TEST_DB,
        vector_index_name="test_vector_idx",
        fulltext_index_name="test_fulltext_idx",
    )

    # Wipe database
    async with store._driver.session(database=NEO4J_TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")
        # Drop indexes if they exist
        try:
            await session.run("DROP INDEX test_vector_idx IF EXISTS")
        except Exception:
            pass
        try:
            await session.run("DROP INDEX test_fulltext_idx IF EXISTS")
        except Exception:
            pass

    # Create test nodes with embeddings
    await store.merge_node(
        GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1"
    )
    await store.merge_node(
        GraphNode(id="n2", label="Company", properties={"name": "Globex Industries"}), "run-1"
    )
    await store.merge_node(
        GraphNode(id="n3", label="Person", properties={"name": "Alice Smith"}), "run-1"
    )

    # Add embeddings directly to nodes
    async with store._driver.session(database=NEO4J_TEST_DB) as session:
        await session.run(
            "MATCH (n {id: 'n1'}) SET n.embedding = $emb",
            emb=[1.0, 0.0, 0.0],
        )
        await session.run(
            "MATCH (n {id: 'n2'}) SET n.embedding = $emb",
            emb=[0.0, 1.0, 0.0],
        )
        await session.run(
            "MATCH (n {id: 'n3'}) SET n.embedding = $emb",
            emb=[0.7, 0.7, 0.0],
        )

    # Add a relationship for graph search
    from graphrag_core.models import GraphRelationship
    await store.merge_relationship(
        GraphRelationship(source_id="n3", target_id="n1", type="WORKS_AT"), "run-1"
    )

    # Create indexes
    await engine.ensure_indexes(
        vector_dimensions=3,
        vector_node_label="Company",
        vector_property="embedding",
        fulltext_node_labels=["Company", "Person"],
        fulltext_properties=["name"],
    )

    # Wait for indexes to come online
    import asyncio
    await asyncio.sleep(1)

    await store.close()
    yield engine
    await engine.close()


class TestNeo4jVectorSearch:
    @pytest.mark.asyncio
    async def test_returns_nearest_neighbors(self, search_engine):
        results = await search_engine.vector_search(
            query_embedding=[1.0, 0.0, 0.0], top_k=3
        )

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "vector"
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_respects_top_k(self, search_engine):
        results = await search_engine.vector_search(
            query_embedding=[1.0, 0.0, 0.0], top_k=1
        )
        assert len(results) == 1


class TestNeo4jFulltextSearch:
    @pytest.mark.asyncio
    async def test_matches_on_name(self, search_engine):
        results = await search_engine.fulltext_search(query="Acme", top_k=10)

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "fulltext"

    @pytest.mark.asyncio
    async def test_filters_by_node_types(self, search_engine):
        results = await search_engine.fulltext_search(
            query="A", node_types=["Person"], top_k=10
        )

        assert all(r.label == "Person" for r in results)


class TestNeo4jGraphSearch:
    @pytest.mark.asyncio
    async def test_traverses_from_start_node(self, search_engine):
        results = await search_engine.graph_search(
            start_node_id="n3", pattern="WORKS_AT", depth=1
        )

        assert len(results) >= 1
        assert results[0].node_id == "n1"
        assert results[0].source == "graph"
        assert results[0].score == 1.0  # 1 hop away -> score = 1/1


class TestNeo4jHybridSearch:
    @pytest.mark.asyncio
    async def test_fuses_vector_and_fulltext(self, search_engine):
        results = await search_engine.hybrid_search(
            query="Acme", embedding=[1.0, 0.0, 0.0], top_k=10
        )

        assert len(results) >= 1
        assert results[0].source == "hybrid"
        # n1 should rank highest: best vector match AND fulltext match
        assert results[0].node_id == "n1"


class TestNeo4jEnsureIndexes:
    @pytest.mark.asyncio
    async def test_idempotent(self, search_engine):
        # Calling ensure_indexes again should not error
        await search_engine.ensure_indexes(
            vector_dimensions=3,
            vector_node_label="Company",
            vector_property="embedding",
            fulltext_node_labels=["Company", "Person"],
            fulltext_properties=["name"],
        )


class TestNeo4jHybridSearchProtocol:
    def test_satisfies_search_engine_protocol(self):
        from graphrag_core.interfaces import SearchEngine
        from graphrag_core.search.neo4j import Neo4jHybridSearch

        engine = Neo4jHybridSearch()
        assert isinstance(engine, SearchEngine)
```

- [ ] **Step 2: Verify integration tests are skipped by default**

Run: `uv run pytest tests/test_search/test_neo4j_search.py -v`
Expected: All tests SKIPPED

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 101 passed, N skipped (11 previous + 8 new integration)

- [ ] **Step 4: Commit**

```bash
git add tests/test_search/test_neo4j_search.py
git commit -m "test: add Neo4jHybridSearch integration tests"
```

---

## Task 8: Update public API re-exports

**Files:**
- Modify: `src/graphrag_core/__init__.py`

- [ ] **Step 1: Add search re-exports**

In `src/graphrag_core/__init__.py`, add the following import after the `from graphrag_core.graph import InMemoryGraphStore` line:

```python
from graphrag_core.search import InMemorySearchEngine
```

Add `"InMemorySearchEngine"` to the `__all__` list under a new `# BB4 implementations` comment:

```python
    # BB4 implementations
    "InMemorySearchEngine",
```

Add a try/except block after the existing `Neo4jGraphStore` try/except:

```python
try:
    from graphrag_core.search import Neo4jHybridSearch
    __all__.append("Neo4jHybridSearch")
except ImportError:
    pass
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from graphrag_core import InMemorySearchEngine, Neo4jHybridSearch; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 101 passed, N skipped

- [ ] **Step 4: Commit**

```bash
git add src/graphrag_core/__init__.py
git commit -m "feat: add BB4 search re-exports to public API"
```

---

## Summary

| Task | What it builds | New tests |
|---|---|---|
| 1 | Extract Cypher safety to shared module | 0 (refactor, existing tests verify) |
| 2 | RRF fusion function | 6 unit tests |
| 3 | InMemorySearchEngine — vector search | 4 unit tests |
| 4 | InMemorySearchEngine — fulltext + hybrid | 7 unit tests |
| 5 | InMemorySearchEngine Protocol conformance | 1 conformance |
| 6 | Neo4jHybridSearch implementation | 0 (tested in Task 7) |
| 7 | Neo4jHybridSearch integration tests | 8 integration tests |
| 8 | Public API re-exports | 0 (wiring) |

**Total new tests:** ~18 unit + 8 integration = ~26
**Expected final count:** ~101 unit tests passing + ~19 integration tests (when Neo4j is available)

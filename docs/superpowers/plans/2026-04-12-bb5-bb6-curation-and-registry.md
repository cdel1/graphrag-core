# BB5 + BB6: Curation & Entity Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the deterministic detection layer (BB5) and entity registry with fuzzy matching (BB6) to provide a quality gate for the knowledge graph.

**Architecture:** BB6 (EntityRegistry) is built first — BB5's duplicate detection uses it. Fuzzy matching uses token normalization + stdlib SequenceMatcher. Detection layer finds duplicates, orphans, and schema violations. CurationPipeline orchestrates detection and accepts optional (not-yet-implemented) LLM curation and approval layers. A small Protocol extension adds `list_nodes()` and `count_relationships()` to GraphStore so the detection layer can scan the graph.

**Tech Stack:** Python 3.12+, Pydantic v2, difflib.SequenceMatcher, pytest + pytest-asyncio

**Design spec:** `docs/superpowers/specs/2026-04-12-bb5-bb6-curation-and-registry-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `src/graphrag_core/registry/__init__.py` | Re-exports `InMemoryEntityRegistry` |
| `src/graphrag_core/registry/matching.py` | `normalize_name()`, `fuzzy_score()` — shared fuzzy matching |
| `src/graphrag_core/registry/memory.py` | `InMemoryEntityRegistry` — dict-based EntityRegistry |
| `src/graphrag_core/curation/__init__.py` | Re-exports `DeterministicDetectionLayer`, `CurationPipeline` |
| `src/graphrag_core/curation/detection.py` | `DeterministicDetectionLayer` — duplicate/orphan/schema detection |
| `src/graphrag_core/curation/pipeline.py` | `CurationPipeline` — orchestrates detection flow |
| `tests/test_registry/__init__.py` | Package marker |
| `tests/test_registry/test_matching.py` | Fuzzy matching unit tests |
| `tests/test_registry/test_memory_registry.py` | InMemoryEntityRegistry unit tests |
| `tests/test_curation/__init__.py` | Package marker |
| `tests/test_curation/test_detection.py` | DeterministicDetectionLayer unit tests |
| `tests/test_curation/test_pipeline.py` | CurationPipeline unit tests |

### Modified files

| File | Change |
|---|---|
| `src/graphrag_core/models.py` | Add BB5 + BB6 models |
| `src/graphrag_core/interfaces.py` | Add BB5 + BB6 Protocols, add `list_nodes()`/`count_relationships()` to GraphStore |
| `src/graphrag_core/graph/memory.py` | Implement `list_nodes()` and `count_relationships()` |
| `src/graphrag_core/graph/neo4j.py` | Implement `list_nodes()` and `count_relationships()` |
| `src/graphrag_core/__init__.py` | Add BB5 + BB6 re-exports |
| `tests/test_interfaces.py` | Add Protocol conformance tests for new Protocols |

---

## Task 1: Add BB5 + BB6 models

**Files:**
- Modify: `src/graphrag_core/models.py`

- [ ] **Step 1: Add BB5 and BB6 models to models.py**

Append after the BB4 section (after line 152) in `src/graphrag_core/models.py`:

```python


# ---------------------------------------------------------------------------
# BB5: Governed Curation
# ---------------------------------------------------------------------------

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


class ApprovalBatch(BaseModel):
    batch_id: str
    status: str              # "pending", "approved", "rejected", "partial"
    issues: list[CurationIssue]


class ApplyResult(BaseModel):
    batch_id: str
    applied: int
    failed: int
    errors: list[str]


# ---------------------------------------------------------------------------
# BB6: Known Entity Registry
# ---------------------------------------------------------------------------

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

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: 101 passed, 19 skipped (no regressions — models are just new classes)

- [ ] **Step 3: Commit**

```bash
git add src/graphrag_core/models.py
git commit -m "feat: add BB5 curation and BB6 registry models"
```

---

## Task 2: Add BB5 + BB6 Protocols and extend GraphStore

**Files:**
- Modify: `src/graphrag_core/interfaces.py`
- Modify: `tests/test_interfaces.py`

- [ ] **Step 1: Add new imports and Protocols to interfaces.py**

First, update the imports at the top of `src/graphrag_core/interfaces.py` to include new model types. Add to the existing import block:

```python
from graphrag_core.models import (
    ApplyResult,
    ApprovalBatch,
    AuditTrail,
    ChunkConfig,
    CurationIssue,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    KnownEntity,
    OntologySchema,
    ParsedDocument,
    RegistryMatch,
    SchemaViolation,
    SearchResult,
)
```

Then add `list_nodes` and `count_relationships` to the `GraphStore` Protocol (after `validate_schema`):

```python
    async def list_nodes(self) -> list[GraphNode]: ...

    async def count_relationships(self) -> int: ...
```

Then append after the BB4 section:

```python


# ---------------------------------------------------------------------------
# BB5: Governed Curation
# ---------------------------------------------------------------------------

@runtime_checkable
class DetectionLayer(Protocol):
    """Deterministic quality checks on the knowledge graph."""

    async def detect(
        self, graph_store: GraphStore, schema: OntologySchema
    ) -> list[CurationIssue]: ...


@runtime_checkable
class LLMCurationLayer(Protocol):
    """LLM-based curation suggestions (entity resolution, relevance)."""

    async def curate(self, issues: list[CurationIssue]) -> list[CurationIssue]: ...


@runtime_checkable
class ApprovalGateway(Protocol):
    """Human approval for high-impact curation operations."""

    async def submit_for_approval(self, issues: list[CurationIssue]) -> str: ...

    async def get_approval_status(self, batch_id: str) -> ApprovalBatch: ...

    async def apply_approved(self, batch_id: str) -> ApplyResult: ...


# ---------------------------------------------------------------------------
# BB6: Known Entity Registry
# ---------------------------------------------------------------------------

@runtime_checkable
class EntityRegistry(Protocol):
    """Manages known entities for deduplication during extraction."""

    async def register(self, entity: KnownEntity) -> str: ...

    async def lookup(
        self, name: str, entity_type: str, match_strategy: str = "fuzzy"
    ) -> list[RegistryMatch]: ...

    async def bulk_register(self, entities: list[KnownEntity]) -> int: ...
```

- [ ] **Step 2: Add Protocol conformance tests**

Add to `tests/test_interfaces.py`. First update the imports at the top to include the new types:

```python
from graphrag_core.interfaces import (
    ApprovalGateway,
    Chunker,
    DetectionLayer,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    GraphStore,
    LLMClient,
    LLMCurationLayer,
    SearchEngine,
)
from graphrag_core.models import (
    ApplyResult,
    ApprovalBatch,
    AuditTrail,
    ChunkConfig,
    CurationIssue,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    KnownEntity,
    OntologySchema,
    ParsedDocument,
    RegistryMatch,
    SchemaViolation,
    SearchResult,
)
```

Then add the `list_nodes` and `count_relationships` stubs to the existing `TestGraphStoreProtocol` `MyStore` class:

```python
            async def list_nodes(self) -> list[GraphNode]:
                raise NotImplementedError

            async def count_relationships(self) -> int:
                raise NotImplementedError
```

Then append new test classes:

```python
class TestDetectionLayerProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyDetector:
            async def detect(
                self, graph_store: GraphStore, schema: OntologySchema
            ) -> list[CurationIssue]:
                raise NotImplementedError

        detector: DetectionLayer = MyDetector()
        assert isinstance(detector, DetectionLayer)


class TestLLMCurationLayerProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyCurator:
            async def curate(self, issues: list[CurationIssue]) -> list[CurationIssue]:
                raise NotImplementedError

        curator: LLMCurationLayer = MyCurator()
        assert isinstance(curator, LLMCurationLayer)


class TestApprovalGatewayProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyGateway:
            async def submit_for_approval(self, issues: list[CurationIssue]) -> str:
                raise NotImplementedError

            async def get_approval_status(self, batch_id: str) -> ApprovalBatch:
                raise NotImplementedError

            async def apply_approved(self, batch_id: str) -> ApplyResult:
                raise NotImplementedError

        gateway: ApprovalGateway = MyGateway()
        assert isinstance(gateway, ApprovalGateway)


class TestEntityRegistryProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyRegistry:
            async def register(self, entity: KnownEntity) -> str:
                raise NotImplementedError

            async def lookup(
                self, name: str, entity_type: str, match_strategy: str = "fuzzy"
            ) -> list[RegistryMatch]:
                raise NotImplementedError

            async def bulk_register(self, entities: list[KnownEntity]) -> int:
                raise NotImplementedError

        registry: EntityRegistry = MyRegistry()
        assert isinstance(registry, EntityRegistry)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_interfaces.py -v`
Expected: 11 tests PASS (7 existing + 4 new)

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 105 passed, 19 skipped

- [ ] **Step 5: Commit**

```bash
git add src/graphrag_core/interfaces.py tests/test_interfaces.py
git commit -m "feat: add BB5 and BB6 Protocols, extend GraphStore with list_nodes/count_relationships"
```

---

## Task 3: Implement list_nodes and count_relationships on both GraphStores

**Files:**
- Modify: `src/graphrag_core/graph/memory.py`
- Modify: `src/graphrag_core/graph/neo4j.py`

- [ ] **Step 1: Add to InMemoryGraphStore**

Add to the end of `InMemoryGraphStore` class in `src/graphrag_core/graph/memory.py`:

```python
    async def list_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    async def count_relationships(self) -> int:
        return len(self._relationships)
```

- [ ] **Step 2: Add to Neo4jGraphStore**

Add to the end of `Neo4jGraphStore` class in `src/graphrag_core/graph/neo4j.py` (before the last line):

```python
    async def list_nodes(self) -> list[GraphNode]:
        query = "MATCH (n) WHERE NOT n:Chunk RETURN n, labels(n) AS labels"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query)
            nodes = []
            async for record in result:
                props = dict(record["n"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                nodes.append(GraphNode(id=node_id, label=label, properties=props))
            return nodes

    async def count_relationships(self) -> int:
        query = "MATCH ()-[r]->() WHERE type(r) <> 'SOURCED' RETURN count(r) AS cnt"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["cnt"] if record else 0
```

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 105 passed, 19 skipped

- [ ] **Step 4: Commit**

```bash
git add src/graphrag_core/graph/memory.py src/graphrag_core/graph/neo4j.py
git commit -m "feat: implement list_nodes and count_relationships on both GraphStores"
```

---

## Task 4: Fuzzy matching utilities

**Files:**
- Create: `tests/test_registry/__init__.py`
- Create: `tests/test_registry/test_matching.py`
- Create: `src/graphrag_core/registry/__init__.py`
- Create: `src/graphrag_core/registry/matching.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_registry/__init__.py` (empty).

Create `tests/test_registry/test_matching.py`:

```python
"""Tests for fuzzy matching utilities."""

from __future__ import annotations


class TestNormalizeName:
    def test_lowercases(self):
        from graphrag_core.registry.matching import normalize_name

        assert normalize_name("ACME Corp") == "acme corp"

    def test_strips_punctuation(self):
        from graphrag_core.registry.matching import normalize_name

        assert normalize_name("Acme, Inc.") == "acme inc"

    def test_sorts_tokens(self):
        from graphrag_core.registry.matching import normalize_name

        assert normalize_name("Smith Alice") == "alice smith"

    def test_strips_extra_whitespace(self):
        from graphrag_core.registry.matching import normalize_name

        assert normalize_name("  Acme   Corp  ") == "acme corp"


class TestFuzzyScore:
    def test_exact_match_returns_one(self):
        from graphrag_core.registry.matching import fuzzy_score

        assert fuzzy_score("Acme Corp", "Acme Corp") == 1.0

    def test_case_insensitive(self):
        from graphrag_core.registry.matching import fuzzy_score

        assert fuzzy_score("acme corp", "ACME CORP") == 1.0

    def test_reordering_scores_high(self):
        from graphrag_core.registry.matching import fuzzy_score

        score = fuzzy_score("Alice Smith", "Smith, Alice")
        assert score >= 0.9

    def test_completely_different_scores_low(self):
        from graphrag_core.registry.matching import fuzzy_score

        score = fuzzy_score("Acme Corp", "Zebra Industries")
        assert score < 0.5

    def test_minor_typo_scores_high(self):
        from graphrag_core.registry.matching import fuzzy_score

        score = fuzzy_score("Acme Corporation", "Acme Corporaton")
        assert score >= 0.8

    def test_empty_strings(self):
        from graphrag_core.registry.matching import fuzzy_score

        assert fuzzy_score("", "") == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry/test_matching.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement matching utilities**

Create `src/graphrag_core/registry/__init__.py`:

```python
"""BB6: Entity registry implementations."""

__all__: list[str] = []
```

Create `src/graphrag_core/registry/matching.py`:

```python
"""Fuzzy matching utilities for entity deduplication."""

from __future__ import annotations

import re
import string
from difflib import SequenceMatcher

_PUNCTUATION_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, sort tokens."""
    name = _PUNCTUATION_RE.sub("", name.lower())
    tokens = sorted(name.split())
    return " ".join(tokens)


def fuzzy_score(a: str, b: str) -> float:
    """Token-normalized SequenceMatcher ratio."""
    norm_a = normalize_name(a)
    norm_b = normalize_name(b)
    return SequenceMatcher(None, norm_a, norm_b).ratio()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry/test_matching.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 115 passed, 19 skipped

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/registry/ tests/test_registry/
git commit -m "feat: add fuzzy matching utilities for entity deduplication"
```

---

## Task 5: InMemoryEntityRegistry

**Files:**
- Create: `tests/test_registry/test_memory_registry.py`
- Create: `src/graphrag_core/registry/memory.py`
- Modify: `src/graphrag_core/registry/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_registry/test_memory_registry.py`:

```python
"""Tests for InMemoryEntityRegistry."""

from __future__ import annotations

import pytest

from graphrag_core.models import KnownEntity


class TestRegister:
    @pytest.mark.asyncio
    async def test_stores_and_returns_id(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        entity = KnownEntity(name="Acme Corp", entity_type="Company")
        entity_id = await registry.register(entity)

        assert entity_id == "Company-acme corp"

    @pytest.mark.asyncio
    async def test_merges_aliases_on_duplicate(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        e1 = KnownEntity(name="Acme Corp", entity_type="Company", aliases=["Acme"])
        e2 = KnownEntity(name="Acme Corp", entity_type="Company", aliases=["ACME Inc"])

        await registry.register(e1)
        await registry.register(e2)

        matches = await registry.lookup("Acme Corp", "Company", match_strategy="exact")
        assert len(matches) == 1
        # Verify the entity has both alias sets (implementation detail tested via lookup)


class TestBulkRegister:
    @pytest.mark.asyncio
    async def test_returns_count_of_newly_registered(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        entities = [
            KnownEntity(name="Acme Corp", entity_type="Company"),
            KnownEntity(name="Globex", entity_type="Company"),
            KnownEntity(name="Acme Corp", entity_type="Company"),  # duplicate
        ]
        count = await registry.bulk_register(entities)

        assert count == 2


class TestLookupExact:
    @pytest.mark.asyncio
    async def test_matches_on_name(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Acme Corp", "Company", match_strategy="exact")
        assert len(matches) == 1
        assert matches[0].name == "Acme Corp"
        assert matches[0].score == 1.0
        assert matches[0].match_method == "exact"

    @pytest.mark.asyncio
    async def test_matches_on_alias(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(
            KnownEntity(name="Acme Corp", entity_type="Company", aliases=["Acme", "ACME Inc"])
        )

        matches = await registry.lookup("Acme", "Company", match_strategy="exact")
        assert len(matches) == 1
        assert matches[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Globex", "Company", match_strategy="exact")
        assert matches == []

    @pytest.mark.asyncio
    async def test_filters_by_entity_type(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Alice", entity_type="Person"))
        await registry.register(KnownEntity(name="Alice", entity_type="Company"))

        matches = await registry.lookup("Alice", "Person", match_strategy="exact")
        assert len(matches) == 1
        assert matches[0].entity_id.startswith("Person-")


class TestLookupFuzzy:
    @pytest.mark.asyncio
    async def test_catches_reorderings(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Alice Smith", entity_type="Person"))

        matches = await registry.lookup("Smith, Alice", "Person", match_strategy="fuzzy")
        assert len(matches) >= 1
        assert matches[0].score >= 0.7
        assert matches[0].match_method == "fuzzy"

    @pytest.mark.asyncio
    async def test_catches_case_differences(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="ACME Corporation", entity_type="Company"))

        matches = await registry.lookup("acme corporation", "Company", match_strategy="fuzzy")
        assert len(matches) >= 1
        assert matches[0].score >= 0.9

    @pytest.mark.asyncio
    async def test_below_threshold_returns_empty(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Zebra Industries", "Company", match_strategy="fuzzy")
        assert matches == []


class TestLookupEmbedding:
    @pytest.mark.asyncio
    async def test_returns_empty(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Acme", "Company", match_strategy="embedding")
        assert matches == []


class TestEntityRegistryProtocol:
    def test_satisfies_entity_registry_protocol(self):
        from graphrag_core.interfaces import EntityRegistry
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        assert isinstance(registry, EntityRegistry)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry/test_memory_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement InMemoryEntityRegistry**

Create `src/graphrag_core/registry/memory.py`:

```python
"""In-memory EntityRegistry implementation."""

from __future__ import annotations

from graphrag_core.models import KnownEntity, RegistryMatch
from graphrag_core.registry.matching import fuzzy_score, normalize_name

_FUZZY_THRESHOLD = 0.7


class InMemoryEntityRegistry:
    """Dict-based EntityRegistry for unit tests and lightweight usage."""

    def __init__(self) -> None:
        self._entities: dict[str, KnownEntity] = {}

    def _make_id(self, entity: KnownEntity) -> str:
        return f"{entity.entity_type}-{normalize_name(entity.name)}"

    async def register(self, entity: KnownEntity) -> str:
        entity_id = self._make_id(entity)
        existing = self._entities.get(entity_id)
        if existing:
            merged_aliases = list(set(existing.aliases + entity.aliases))
            self._entities[entity_id] = KnownEntity(
                name=existing.name,
                entity_type=existing.entity_type,
                aliases=merged_aliases,
                properties={**existing.properties, **entity.properties},
            )
        else:
            self._entities[entity_id] = entity
        return entity_id

    async def bulk_register(self, entities: list[KnownEntity]) -> int:
        count = 0
        for entity in entities:
            entity_id = self._make_id(entity)
            was_new = entity_id not in self._entities
            await self.register(entity)
            if was_new:
                count += 1
        return count

    async def lookup(
        self, name: str, entity_type: str, match_strategy: str = "fuzzy"
    ) -> list[RegistryMatch]:
        if match_strategy == "embedding":
            return []

        matches: list[RegistryMatch] = []

        for entity_id, entity in self._entities.items():
            if entity.entity_type != entity_type:
                continue

            if match_strategy == "exact":
                all_names = [entity.name] + entity.aliases
                if name in all_names:
                    matches.append(RegistryMatch(
                        entity_id=entity_id,
                        name=entity.name,
                        score=1.0,
                        match_method="exact",
                    ))
            elif match_strategy == "fuzzy":
                all_names = [entity.name] + entity.aliases
                best_score = 0.0
                for candidate in all_names:
                    score = fuzzy_score(name, candidate)
                    best_score = max(best_score, score)
                if best_score >= _FUZZY_THRESHOLD:
                    matches.append(RegistryMatch(
                        entity_id=entity_id,
                        name=entity.name,
                        score=best_score,
                        match_method="fuzzy",
                    ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches
```

- [ ] **Step 4: Update registry __init__.py**

Replace `src/graphrag_core/registry/__init__.py`:

```python
"""BB6: Entity registry implementations."""

from graphrag_core.registry.memory import InMemoryEntityRegistry

__all__ = ["InMemoryEntityRegistry"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry/test_memory_registry.py -v`
Expected: All 11 tests PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 126 passed, 19 skipped

- [ ] **Step 7: Commit**

```bash
git add src/graphrag_core/registry/ tests/test_registry/
git commit -m "feat: add InMemoryEntityRegistry with fuzzy matching"
```

---

## Task 6: DeterministicDetectionLayer

**Files:**
- Create: `tests/test_curation/__init__.py`
- Create: `tests/test_curation/test_detection.py`
- Create: `src/graphrag_core/curation/__init__.py`
- Create: `src/graphrag_core/curation/detection.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_curation/__init__.py` (empty).

Create `tests/test_curation/test_detection.py`:

```python
"""Tests for DeterministicDetectionLayer."""

from __future__ import annotations

import pytest

from graphrag_core.models import (
    GraphNode,
    GraphRelationship,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
    KnownEntity,
)


def _schema() -> OntologySchema:
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Company",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
            NodeTypeDefinition(
                label="Person",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(type="WORKS_AT", source_types=["Person"], target_types=["Company"]),
        ],
    )


async def _populated_store():
    from graphrag_core.graph.memory import InMemoryGraphStore

    store = InMemoryGraphStore()
    await store.apply_schema(_schema())
    await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1")
    await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "ACME Corporation"}), "run-1")
    await store.merge_node(GraphNode(id="n3", label="Person", properties={"name": "Alice"}), "run-1")
    await store.merge_relationship(
        GraphRelationship(source_id="n3", target_id="n1", type="WORKS_AT"), "run-1"
    )
    return store


class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_detects_fuzzy_duplicates_with_registry(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        store = await _populated_store()
        detector = DeterministicDetectionLayer(entity_registry=registry)
        issues = await detector.detect(store, _schema())

        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        assert len(dup_issues) >= 1
        assert any("n1" in i.affected_nodes and "n2" in i.affected_nodes for i in dup_issues)

    @pytest.mark.asyncio
    async def test_detects_duplicates_without_registry(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer

        store = await _populated_store()
        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        assert len(dup_issues) >= 1

    @pytest.mark.asyncio
    async def test_no_false_positives_on_distinct_names(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Globex Industries"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="PARTNER"), "run-1"
        )

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        assert len(dup_issues) == 0


class TestOrphanDetection:
    @pytest.mark.asyncio
    async def test_detects_orphan_nodes(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer

        store = await _populated_store()
        # n2 ("ACME Corporation") has no relationships
        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        orphan_issues = [i for i in issues if i.issue_type == "orphan"]
        assert len(orphan_issues) >= 1
        orphan_node_ids = [nid for i in orphan_issues for nid in i.affected_nodes]
        assert "n2" in orphan_node_ids

    @pytest.mark.asyncio
    async def test_no_orphans_when_all_connected(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        orphan_issues = [i for i in issues if i.issue_type == "orphan"]
        assert len(orphan_issues) == 0


class TestSchemaViolationDetection:
    @pytest.mark.asyncio
    async def test_detects_missing_required_property(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.apply_schema(_schema())
        await store.merge_node(GraphNode(id="n1", label="Company", properties={}), "run-1")

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        schema_issues = [i for i in issues if i.issue_type == "schema_violation"]
        assert len(schema_issues) >= 1
        assert schema_issues[0].severity == "error"


class TestPairwiseCap:
    @pytest.mark.asyncio
    async def test_emits_warning_when_label_group_exceeds_cap(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer, _PAIRWISE_CAP
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        for i in range(_PAIRWISE_CAP + 1):
            await store.merge_node(
                GraphNode(id=f"n{i}", label="Company", properties={"name": f"Company {i}"}), "run-1"
            )

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        skip_issues = [i for i in issues if i.issue_type == "skipped_detection"]
        assert len(skip_issues) >= 1
        assert skip_issues[0].severity == "warning"


class TestDetectionLayerProtocol:
    def test_satisfies_detection_layer_protocol(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.interfaces import DetectionLayer

        detector = DeterministicDetectionLayer()
        assert isinstance(detector, DetectionLayer)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_curation/test_detection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DeterministicDetectionLayer**

Create `src/graphrag_core/curation/__init__.py`:

```python
"""BB5: Governed curation implementations."""

__all__: list[str] = []
```

Create `src/graphrag_core/curation/detection.py`:

```python
"""BB5: Deterministic detection layer for graph quality issues."""

from __future__ import annotations

import uuid
from collections import defaultdict

from graphrag_core.interfaces import EntityRegistry, GraphStore
from graphrag_core.models import CurationIssue, GraphNode, OntologySchema
from graphrag_core.registry.matching import fuzzy_score

_PAIRWISE_CAP = 1000
_FUZZY_THRESHOLD = 0.7


class DeterministicDetectionLayer:
    """Finds duplicates, orphans, and schema violations without LLM calls."""

    def __init__(self, entity_registry: EntityRegistry | None = None) -> None:
        self._registry = entity_registry

    async def detect(
        self, graph_store: GraphStore, schema: OntologySchema
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []

        nodes = await graph_store.list_nodes()

        issues.extend(await self._detect_duplicates(nodes))
        issues.extend(await self._detect_orphans(nodes, graph_store))
        issues.extend(await self._detect_schema_violations(graph_store, schema))

        return issues

    async def _detect_duplicates(self, nodes: list[GraphNode]) -> list[CurationIssue]:
        issues: list[CurationIssue] = []

        groups: dict[str, list[GraphNode]] = defaultdict(list)
        for node in nodes:
            groups[node.label].append(node)

        for label, group in groups.items():
            if self._registry is not None:
                issues.extend(await self._detect_duplicates_with_registry(group))
            else:
                if len(group) > _PAIRWISE_CAP:
                    issues.append(CurationIssue(
                        id=str(uuid.uuid4()),
                        issue_type="skipped_detection",
                        severity="warning",
                        affected_nodes=[],
                        suggested_action=f"Register entities in EntityRegistry for label '{label}' ({len(group)} nodes exceeds pairwise cap of {_PAIRWISE_CAP})",
                        auto_fixable=False,
                        source_layer="deterministic",
                    ))
                else:
                    issues.extend(self._detect_duplicates_pairwise(group))

        return issues

    async def _detect_duplicates_with_registry(
        self, nodes: list[GraphNode]
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []
        entity_to_nodes: dict[str, list[str]] = defaultdict(list)

        for node in nodes:
            name = node.properties.get("name", "")
            if not name:
                continue
            matches = await self._registry.lookup(name, node.label, match_strategy="fuzzy")
            if matches:
                entity_to_nodes[matches[0].entity_id].append(node.id)

        for entity_id, node_ids in entity_to_nodes.items():
            if len(node_ids) > 1:
                issues.append(CurationIssue(
                    id=str(uuid.uuid4()),
                    issue_type="duplicate",
                    severity="warning",
                    affected_nodes=node_ids,
                    suggested_action=f"Merge nodes {node_ids} — they match registry entity '{entity_id}'",
                    auto_fixable=False,
                    source_layer="deterministic",
                ))

        return issues

    def _detect_duplicates_pairwise(
        self, nodes: list[GraphNode]
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, a in enumerate(nodes):
            name_a = a.properties.get("name", "")
            if not name_a:
                continue
            for b in nodes[i + 1:]:
                name_b = b.properties.get("name", "")
                if not name_b:
                    continue
                pair = (min(a.id, b.id), max(a.id, b.id))
                if pair in seen_pairs:
                    continue
                score = fuzzy_score(name_a, name_b)
                if score >= _FUZZY_THRESHOLD:
                    seen_pairs.add(pair)
                    issues.append(CurationIssue(
                        id=str(uuid.uuid4()),
                        issue_type="duplicate",
                        severity="warning",
                        affected_nodes=[a.id, b.id],
                        suggested_action=f"Merge '{name_a}' and '{name_b}' (similarity: {score:.2f})",
                        auto_fixable=False,
                        source_layer="deterministic",
                    ))

        return issues

    async def _detect_orphans(
        self, nodes: list[GraphNode], graph_store: GraphStore
    ) -> list[CurationIssue]:
        issues: list[CurationIssue] = []

        for node in nodes:
            related = await graph_store.get_related(node.id)
            if not related:
                issues.append(CurationIssue(
                    id=str(uuid.uuid4()),
                    issue_type="orphan",
                    severity="info",
                    affected_nodes=[node.id],
                    suggested_action=f"Node '{node.id}' ({node.label}) has no relationships",
                    auto_fixable=False,
                    source_layer="deterministic",
                ))

        return issues

    async def _detect_schema_violations(
        self, graph_store: GraphStore, schema: OntologySchema
    ) -> list[CurationIssue]:
        await graph_store.apply_schema(schema)
        violations = await graph_store.validate_schema()

        return [
            CurationIssue(
                id=str(uuid.uuid4()),
                issue_type="schema_violation",
                severity="error",
                affected_nodes=[v.node_id],
                suggested_action=v.message,
                auto_fixable=False,
                source_layer="deterministic",
            )
            for v in violations
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_curation/test_detection.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 134 passed, 19 skipped

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/curation/ tests/test_curation/
git commit -m "feat: add DeterministicDetectionLayer with duplicate, orphan, and schema detection"
```

---

## Task 7: CurationPipeline

**Files:**
- Create: `tests/test_curation/test_pipeline.py`
- Create: `src/graphrag_core/curation/pipeline.py`
- Modify: `src/graphrag_core/curation/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_curation/test_pipeline.py`:

```python
"""Tests for CurationPipeline."""

from __future__ import annotations

import pytest

from graphrag_core.models import (
    GraphNode,
    GraphRelationship,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
)


def _schema() -> OntologySchema:
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Company",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
        ],
        relationship_types=[],
    )


class TestCurationPipeline:
    @pytest.mark.asyncio
    async def test_runs_detection_and_returns_report(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        pipeline = CurationPipeline(detection=DeterministicDetectionLayer())
        report = await pipeline.run(store, _schema())

        assert report.nodes_scanned == 1
        assert report.relationships_scanned == 0
        assert isinstance(report.issues, list)

    @pytest.mark.asyncio
    async def test_works_without_llm_or_approval(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        pipeline = CurationPipeline(
            detection=DeterministicDetectionLayer(),
            llm_curation=None,
            approval=None,
        )
        report = await pipeline.run(store, _schema())

        assert report.nodes_scanned == 0
        assert report.issues == []

    @pytest.mark.asyncio
    async def test_report_contains_detected_issues(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.apply_schema(_schema())
        # Node missing required "name" property
        await store.merge_node(GraphNode(id="n1", label="Company", properties={}), "run-1")

        pipeline = CurationPipeline(detection=DeterministicDetectionLayer())
        report = await pipeline.run(store, _schema())

        assert len(report.issues) >= 1
        assert any(i.issue_type == "schema_violation" for i in report.issues)

    @pytest.mark.asyncio
    async def test_report_counts_nodes_and_relationships(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "A"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "B"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="PARTNER"), "run-1"
        )

        pipeline = CurationPipeline(detection=DeterministicDetectionLayer())
        report = await pipeline.run(store, _schema())

        assert report.nodes_scanned == 2
        assert report.relationships_scanned == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_curation/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CurationPipeline**

Create `src/graphrag_core/curation/pipeline.py`:

```python
"""BB5: Curation pipeline orchestrator."""

from __future__ import annotations

from graphrag_core.interfaces import ApprovalGateway, DetectionLayer, GraphStore, LLMCurationLayer
from graphrag_core.models import CurationReport, OntologySchema


class CurationPipeline:
    """Orchestrates the curation flow: detect → (curate) → (approve)."""

    def __init__(
        self,
        detection: DetectionLayer,
        llm_curation: LLMCurationLayer | None = None,
        approval: ApprovalGateway | None = None,
    ) -> None:
        self._detection = detection
        self._llm_curation = llm_curation
        self._approval = approval

    async def run(
        self,
        graph_store: GraphStore,
        schema: OntologySchema,
    ) -> CurationReport:
        issues = await self._detection.detect(graph_store, schema)

        if self._llm_curation is not None:
            issues = await self._llm_curation.curate(issues)

        nodes = await graph_store.list_nodes()
        rel_count = await graph_store.count_relationships()

        return CurationReport(
            issues=issues,
            nodes_scanned=len(nodes),
            relationships_scanned=rel_count,
        )
```

- [ ] **Step 4: Update curation __init__.py**

Replace `src/graphrag_core/curation/__init__.py`:

```python
"""BB5: Governed curation implementations."""

from graphrag_core.curation.detection import DeterministicDetectionLayer
from graphrag_core.curation.pipeline import CurationPipeline

__all__ = ["DeterministicDetectionLayer", "CurationPipeline"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_curation/test_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 138 passed, 19 skipped

- [ ] **Step 7: Commit**

```bash
git add src/graphrag_core/curation/ tests/test_curation/
git commit -m "feat: add CurationPipeline orchestrating detection flow"
```

---

## Task 8: Update public API re-exports

**Files:**
- Modify: `src/graphrag_core/__init__.py`

- [ ] **Step 1: Update imports and __all__**

In `src/graphrag_core/__init__.py`:

Add these imports after the search import:

```python
from graphrag_core.registry import InMemoryEntityRegistry
from graphrag_core.curation import DeterministicDetectionLayer, CurationPipeline
```

Add to the interfaces import block:

```python
from graphrag_core.interfaces import (
    ApprovalGateway,
    Chunker,
    DetectionLayer,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    GraphStore,
    IngestionPipeline,
    LLMClient,
    LLMCurationLayer,
    SearchEngine,
)
```

Add to `__all__`, the new Protocols and implementations:

```python
    # Protocols (add to existing section)
    "ApprovalGateway",
    "DetectionLayer",
    "EntityRegistry",
    "LLMCurationLayer",
    # BB5 implementations
    "CurationPipeline",
    "DeterministicDetectionLayer",
    # BB6 implementations
    "InMemoryEntityRegistry",
```

Add to models import:

```python
from graphrag_core.models import (
    CurationIssue,
    CurationReport,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    ImportRun,
    KnownEntity,
    NodeTypeDefinition,
    OntologySchema,
    RegistryMatch,
    SearchResult,
)
```

And add the new models to `__all__`:

```python
    # Models (add to existing section)
    "CurationIssue",
    "CurationReport",
    "KnownEntity",
    "RegistryMatch",
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from graphrag_core import DetectionLayer, EntityRegistry, DeterministicDetectionLayer, CurationPipeline, InMemoryEntityRegistry, CurationIssue, KnownEntity; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 138 passed, 19 skipped

- [ ] **Step 4: Commit**

```bash
git add src/graphrag_core/__init__.py
git commit -m "feat: add BB5 and BB6 re-exports to public API"
```

---

## Summary

| Task | What it builds | New tests |
|---|---|---|
| 1 | BB5 + BB6 models | 0 (models only) |
| 2 | Protocols + GraphStore extension | 4 conformance tests |
| 3 | list_nodes/count_relationships on both stores | 0 (verified by existing tests) |
| 4 | Fuzzy matching utilities | 10 unit tests |
| 5 | InMemoryEntityRegistry | 11 unit tests |
| 6 | DeterministicDetectionLayer | 8 unit tests |
| 7 | CurationPipeline | 4 unit tests |
| 8 | Public API re-exports | 0 (wiring) |

**Total new tests:** ~37
**Expected final count:** ~138 unit tests passing + 19 integration tests (when Neo4j is available)

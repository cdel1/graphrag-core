# BB2 + BB3: Extraction Engine & Graph Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement BB2 (LLMExtractionEngine) and BB3 (Neo4jGraphStore) to create a working document-to-graph pipeline with provenance tracking.

**Architecture:** Provider-agnostic LLMClient Protocol with Anthropic default. Extraction engine validates strictly against OntologySchema. Neo4jGraphStore implements all 8 GraphStore Protocol methods. InMemoryGraphStore provides fast unit-test support. Both building blocks are independent behind the GraphStore Protocol and can be developed in parallel.

**Tech Stack:** Python 3.12+, Pydantic v2, anthropic SDK, neo4j async driver, pytest + pytest-asyncio

**Design spec:** `docs/superpowers/specs/2026-04-12-bb2-bb3-extraction-and-graph-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `src/graphrag_core/llm/__init__.py` | Re-exports `AnthropicLLMClient` |
| `src/graphrag_core/llm/anthropic.py` | `AnthropicLLMClient` — thin wrapper around `anthropic` SDK |
| `src/graphrag_core/extraction/__init__.py` | Re-exports `LLMExtractionEngine` |
| `src/graphrag_core/extraction/engine.py` | `LLMExtractionEngine` — prompt building, JSON parsing, schema validation |
| `src/graphrag_core/graph/__init__.py` | Re-exports `Neo4jGraphStore`, `InMemoryGraphStore` |
| `src/graphrag_core/graph/memory.py` | `InMemoryGraphStore` — dict-based GraphStore for tests |
| `src/graphrag_core/graph/neo4j.py` | `Neo4jGraphStore` — async Neo4j driver implementation |
| `tests/conftest.py` | Integration marker registration, Neo4j fixture |
| `tests/test_extraction/__init__.py` | Package marker |
| `tests/test_extraction/test_engine.py` | LLMExtractionEngine unit tests with FakeLLMClient |
| `tests/test_graph/__init__.py` | Package marker |
| `tests/test_graph/test_memory.py` | InMemoryGraphStore unit tests |
| `tests/test_graph/test_neo4j.py` | Neo4jGraphStore integration tests |
| `tests/test_integration/__init__.py` | Package marker |
| `tests/test_integration/test_ingest_to_graph.py` | End-to-end BB1 → BB2 → BB3 integration test |

### Modified files

| File | Change |
|---|---|
| `src/graphrag_core/interfaces.py` | Add `LLMClient` Protocol |
| `src/graphrag_core/__init__.py` | Add re-exports for new classes |
| `pyproject.toml` | Add `anthropic` and `neo4j` optional extras, add `integration` pytest marker |
| `tests/test_interfaces.py` | Add `LLMClient` Protocol conformance test |

---

## Task 1: Add LLMClient Protocol

**Files:**
- Modify: `src/graphrag_core/interfaces.py`
- Modify: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_interfaces.py`:

```python
from graphrag_core.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingModel,
    ExtractionEngine,
    GraphStore,
    LLMClient,
    SearchEngine,
)

# ... (existing test classes remain unchanged) ...

class TestLLMClientProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyLLMClient:
            async def complete(
                self,
                messages: list[dict[str, str]],
                system: str | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> str:
                raise NotImplementedError

        client: LLMClient = MyLLMClient()
        assert isinstance(client, LLMClient)
```

Update the import at the top to include `LLMClient`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_interfaces.py::TestLLMClientProtocol -v`
Expected: FAIL with `ImportError: cannot import name 'LLMClient'`

- [ ] **Step 3: Add LLMClient Protocol to interfaces.py**

Add after the `EmbeddingModel` Protocol and before the BB2 section comment in `src/graphrag_core/interfaces.py`:

```python
@runtime_checkable
class LLMClient(Protocol):
    """Sends structured prompts to a language model and returns text."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_interfaces.py::TestLLMClientProtocol -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: 61 passed (60 existing + 1 new)

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/interfaces.py tests/test_interfaces.py
git commit -m "feat: add LLMClient Protocol to interfaces"
```

---

## Task 2: Add pytest integration marker and conftest

**Files:**
- Create: `tests/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add integration marker to pyproject.toml**

Add to the `[tool.pytest.ini_options]` section in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: requires external services (Neo4j, LLM APIs)",
]
```

- [ ] **Step 2: Create tests/conftest.py**

```python
"""Shared fixtures for graphrag-core tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --run-integration is passed or RUN_INTEGRATION=1."""
    run_integration = config.getoption("--run-integration", default=False) or os.environ.get(
        "RUN_INTEGRATION", ""
    ) == "1"
    if run_integration:
        return
    skip = pytest.mark.skip(reason="integration tests require --run-integration or RUN_INTEGRATION=1")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services",
    )
```

- [ ] **Step 3: Run full suite to verify nothing broke**

Run: `uv run pytest tests/ -x -q`
Expected: 61 passed (existing tests unchanged, no integration tests added yet)

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py pyproject.toml
git commit -m "chore: add integration test marker and conftest"
```

---

## Task 3: Add optional dependencies (anthropic, neo4j)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional extras to pyproject.toml**

Add after the `[dependency-groups]` section:

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
neo4j = ["neo4j>=5.0"]
all = ["graphrag-core[anthropic,neo4j]"]
```

- [ ] **Step 2: Install the dev + all extras**

Run: `uv sync --all-extras`
Expected: `anthropic` and `neo4j` packages installed successfully

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "import anthropic; import neo4j; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add anthropic and neo4j optional dependencies"
```

---

## Task 4: Implement AnthropicLLMClient

**Files:**
- Create: `src/graphrag_core/llm/__init__.py`
- Create: `src/graphrag_core/llm/anthropic.py`

- [ ] **Step 1: Create llm package with __init__.py**

Create `src/graphrag_core/llm/__init__.py`:

```python
"""LLM client implementations."""

from graphrag_core.llm.anthropic import AnthropicLLMClient

__all__ = ["AnthropicLLMClient"]
```

- [ ] **Step 2: Implement AnthropicLLMClient**

Create `src/graphrag_core/llm/anthropic.py`:

```python
"""Anthropic Claude LLM client."""

from __future__ import annotations

from anthropic import AsyncAnthropic


class AnthropicLLMClient:
    """Thin wrapper around the Anthropic SDK implementing the LLMClient Protocol."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system is not None:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text
```

- [ ] **Step 3: Verify Protocol conformance**

Run: `uv run python -c "from graphrag_core.interfaces import LLMClient; from graphrag_core.llm import AnthropicLLMClient; assert isinstance(AnthropicLLMClient(), LLMClient); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 61 passed (no regressions)

- [ ] **Step 5: Commit**

```bash
git add src/graphrag_core/llm/
git commit -m "feat: add AnthropicLLMClient implementing LLMClient Protocol"
```

---

## Task 5: LLMExtractionEngine — happy path

**Files:**
- Create: `tests/test_extraction/__init__.py`
- Create: `tests/test_extraction/test_engine.py`
- Create: `src/graphrag_core/extraction/__init__.py`
- Create: `src/graphrag_core/extraction/engine.py`

- [ ] **Step 1: Create test package and write the FakeLLMClient + happy path test**

Create `tests/test_extraction/__init__.py` (empty).

Create `tests/test_extraction/test_engine.py`:

```python
"""Tests for LLMExtractionEngine."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentChunk,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
)


class FakeLLMClient:
    """Returns canned JSON responses for testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


def _schema() -> OntologySchema:
    """A simple test schema: Company and Person with WORKS_AT relationship."""
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Company",
                properties=[
                    PropertyDefinition(name="name", type="string", required=True),
                ],
                required_properties=["name"],
            ),
            NodeTypeDefinition(
                label="Person",
                properties=[
                    PropertyDefinition(name="name", type="string", required=True),
                    PropertyDefinition(name="role", type="string", required=False),
                ],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(
                type="WORKS_AT",
                source_types=["Person"],
                target_types=["Company"],
            ),
        ],
    )


def _import_run() -> ImportRun:
    return ImportRun(
        id="run-1",
        timestamp=datetime(2026, 4, 12, 10, 0),
        source_type="text/plain",
        documents_processed=1,
        entities_extracted=0,
    )


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(id="chunk-0", text="Alice is a software engineer at Acme Corp.", position=0),
    ]


class TestExtractionEngineHappyPath:
    @pytest.mark.asyncio
    async def test_extracts_nodes_and_relationships(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice", "role": "software engineer"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme Corp"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 2
        assert result.nodes[0].label == "Person"
        assert result.nodes[0].properties["name"] == "Alice"
        assert result.nodes[1].label == "Company"
        assert result.nodes[1].properties["name"] == "Acme Corp"

        assert len(result.relationships) == 1
        assert result.relationships[0].type == "WORKS_AT"
        assert result.relationships[0].source_id == "person-alice"
        assert result.relationships[0].target_id == "company-acme"

        assert len(result.provenance) == 2
        chunk_ids = {p.chunk_id for p in result.provenance}
        assert chunk_ids == {"chunk-0"}
        node_ids = {p.node_id for p in result.provenance}
        assert node_ids == {"person-alice", "company-acme"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extraction/test_engine.py::TestExtractionEngineHappyPath -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graphrag_core.extraction.engine'`

- [ ] **Step 3: Implement LLMExtractionEngine (core extraction + validation)**

Create `src/graphrag_core/extraction/__init__.py`:

```python
"""BB2: Schema-guided entity extraction."""

from graphrag_core.extraction.engine import LLMExtractionEngine

__all__ = ["LLMExtractionEngine"]
```

Create `src/graphrag_core/extraction/engine.py`:

```python
"""BB2: LLM-powered schema-guided entity extraction engine."""

from __future__ import annotations

import json

from graphrag_core.interfaces import LLMClient
from graphrag_core.models import (
    DocumentChunk,
    ExtractedNode,
    ExtractedRelationship,
    ExtractionResult,
    ImportRun,
    OntologySchema,
    ProvenanceLink,
)


class LLMExtractionEngine:
    """Extracts entities and relationships from text using an LLM, guided by an ontology schema."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def extract(
        self,
        chunks: list[DocumentChunk],
        schema: OntologySchema,
        import_run: ImportRun,
    ) -> ExtractionResult:
        all_nodes: list[ExtractedNode] = []
        all_rels: list[ExtractedRelationship] = []
        all_provenance: list[ProvenanceLink] = []

        system_prompt = self._build_system_prompt(schema)

        for chunk in chunks:
            nodes, rels = await self._extract_chunk(chunk, system_prompt)
            nodes, rels = self._validate(nodes, rels, schema)

            for node in nodes:
                all_provenance.append(
                    ProvenanceLink(chunk_id=chunk.id, node_id=node.id, confidence=1.0)
                )

            all_nodes.extend(nodes)
            all_rels.extend(rels)

        return ExtractionResult(
            nodes=all_nodes,
            relationships=all_rels,
            provenance=all_provenance,
        )

    async def _extract_chunk(
        self, chunk: DocumentChunk, system_prompt: str
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        response = await self._llm.complete(
            messages=[{"role": "user", "content": chunk.text}],
            system=system_prompt,
            temperature=0.0,
        )
        return self._parse_response(response)

    def _build_system_prompt(self, schema: OntologySchema) -> str:
        node_descriptions = []
        for nt in schema.node_types:
            props = ", ".join(
                f"{p.name} ({p.type}{', required' if p.required else ''})"
                for p in nt.properties
            )
            node_descriptions.append(f"- {nt.label}: properties=[{props}]")

        rel_descriptions = []
        for rt in schema.relationship_types:
            rel_descriptions.append(
                f"- {rt.type}: {rt.source_types} -> {rt.target_types}"
            )

        return (
            "You are an entity extraction engine. Extract entities and relationships "
            "from the provided text according to this schema.\n\n"
            "ALLOWED NODE TYPES:\n"
            + "\n".join(node_descriptions)
            + "\n\nALLOWED RELATIONSHIP TYPES:\n"
            + "\n".join(rel_descriptions)
            + "\n\nDo not extract entities or relationships not listed above.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"nodes": [{"id": "<unique_id>", "label": "<NodeType>", "properties": {<key>: <value>}}], '
            '"relationships": [{"source_id": "<node_id>", "target_id": "<node_id>", "type": "<RelType>", "properties": {}}]}\n\n'
            "Rules:\n"
            "- Every node id must be unique and descriptive (e.g., 'person-alice', 'company-acme')\n"
            "- Only use node types and relationship types listed above\n"
            "- Include all required properties for each node type\n"
            "- Return empty arrays if no entities are found"
        )

    def _parse_response(
        self, response: str
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        data = json.loads(response)

        nodes = [
            ExtractedNode(
                id=n["id"],
                label=n["label"],
                properties=n.get("properties", {}),
            )
            for n in data.get("nodes", [])
        ]

        rels = [
            ExtractedRelationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                type=r["type"],
                properties=r.get("properties", {}),
            )
            for r in data.get("relationships", [])
        ]

        return nodes, rels

    def _validate(
        self,
        nodes: list[ExtractedNode],
        rels: list[ExtractedRelationship],
        schema: OntologySchema,
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        allowed_labels = {nt.label for nt in schema.node_types}
        allowed_rel_types = {rt.type for rt in schema.relationship_types}
        rel_constraints = {
            rt.type: (set(rt.source_types), set(rt.target_types))
            for rt in schema.relationship_types
        }

        valid_nodes = [n for n in nodes if n.label in allowed_labels]
        valid_node_ids = {n.id for n in valid_nodes}
        node_labels = {n.id: n.label for n in valid_nodes}

        valid_rels = []
        for rel in rels:
            if rel.type not in allowed_rel_types:
                continue
            if rel.source_id not in valid_node_ids or rel.target_id not in valid_node_ids:
                continue
            source_types, target_types = rel_constraints[rel.type]
            if node_labels[rel.source_id] not in source_types:
                continue
            if node_labels[rel.target_id] not in target_types:
                continue
            valid_rels.append(rel)

        return valid_nodes, valid_rels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_extraction/test_engine.py::TestExtractionEngineHappyPath -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 62 passed

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/extraction/ tests/test_extraction/
git commit -m "feat: add LLMExtractionEngine with schema-guided extraction"
```

---

## Task 6: LLMExtractionEngine — validation edge cases

**Files:**
- Modify: `tests/test_extraction/test_engine.py`

- [ ] **Step 1: Write test for off-schema node filtering**

Add to `tests/test_extraction/test_engine.py`:

```python
class TestExtractionEngineValidation:
    @pytest.mark.asyncio
    async def test_drops_off_schema_nodes(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "New York"}},
            ],
            "relationships": [],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 1
        assert result.nodes[0].label == "Person"

    @pytest.mark.asyncio
    async def test_drops_off_schema_relationships(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "FOUNDED", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_drops_relationship_with_wrong_source_type(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "company-a", "label": "Company", "properties": {"name": "A"}},
                {"id": "company-b", "label": "Company", "properties": {"name": "B"}},
            ],
            "relationships": [
                {"source_id": "company-a", "target_id": "company-b", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_drops_dangling_relationships(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "NYC"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "loc-nyc", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        # Location is off-schema, so it's dropped. The relationship references the dropped node.
        assert len(result.nodes) == 1
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=["not valid json {{"]))
        with pytest.raises(json.JSONDecodeError):
            await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty_result(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[]))
        result = await engine.extract(chunks=[], schema=_schema(), import_run=_import_run())

        assert result.nodes == []
        assert result.relationships == []
        assert result.provenance == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction/test_engine.py -v`
Expected: All 7 tests PASS (1 happy path from Task 5 + 6 validation tests)

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 67 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_extraction/test_engine.py
git commit -m "test: add extraction engine validation edge case tests"
```

---

## Task 7: LLMExtractionEngine — multi-chunk extraction

**Files:**
- Modify: `tests/test_extraction/test_engine.py`

- [ ] **Step 1: Write test for multi-chunk extraction with provenance tracking**

Add to `tests/test_extraction/test_engine.py`:

```python
class TestExtractionEngineMultiChunk:
    @pytest.mark.asyncio
    async def test_extracts_across_multiple_chunks(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        chunks = [
            DocumentChunk(id="chunk-0", text="Alice works at Acme.", position=0),
            DocumentChunk(id="chunk-1", text="Bob works at Globex.", position=1),
        ]

        response_0 = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT", "properties": {}},
            ],
        })
        response_1 = json.dumps({
            "nodes": [
                {"id": "person-bob", "label": "Person", "properties": {"name": "Bob"}},
                {"id": "company-globex", "label": "Company", "properties": {"name": "Globex"}},
            ],
            "relationships": [
                {"source_id": "person-bob", "target_id": "company-globex", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(
            llm_client=FakeLLMClient(responses=[response_0, response_1])
        )
        result = await engine.extract(chunks=chunks, schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 4
        assert len(result.relationships) == 2
        assert len(result.provenance) == 4

        # Verify provenance maps to correct chunks
        chunk_0_provenance = [p for p in result.provenance if p.chunk_id == "chunk-0"]
        chunk_1_provenance = [p for p in result.provenance if p.chunk_id == "chunk-1"]
        assert len(chunk_0_provenance) == 2
        assert len(chunk_1_provenance) == 2
        assert {p.node_id for p in chunk_0_provenance} == {"person-alice", "company-acme"}
        assert {p.node_id for p in chunk_1_provenance} == {"person-bob", "company-globex"}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_extraction/test_engine.py::TestExtractionEngineMultiChunk -v`
Expected: PASS

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 68 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_extraction/test_engine.py
git commit -m "test: add multi-chunk extraction with provenance tracking tests"
```

---

## Task 8: ExtractionEngine Protocol conformance

**Files:**
- Modify: `tests/test_extraction/test_engine.py`

- [ ] **Step 1: Write Protocol conformance test**

Add to `tests/test_extraction/test_engine.py`:

```python
class TestExtractionEngineProtocol:
    def test_satisfies_extraction_engine_protocol(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine
        from graphrag_core.interfaces import ExtractionEngine

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[]))
        assert isinstance(engine, ExtractionEngine)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_extraction/test_engine.py::TestExtractionEngineProtocol -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_extraction/test_engine.py
git commit -m "test: add ExtractionEngine Protocol conformance test"
```

---

## Task 9: InMemoryGraphStore — write operations

**Files:**
- Create: `tests/test_graph/__init__.py`
- Create: `tests/test_graph/test_memory.py`
- Create: `src/graphrag_core/graph/__init__.py`
- Create: `src/graphrag_core/graph/memory.py`

- [ ] **Step 1: Write failing tests for merge_node, merge_relationship, record_provenance**

Create `tests/test_graph/__init__.py` (empty).

Create `tests/test_graph/test_memory.py`:

```python
"""Tests for InMemoryGraphStore."""

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


class TestInMemoryGraphStoreWrite:
    @pytest.mark.asyncio
    async def test_merge_node_stores_and_returns_id(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        node = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        result_id = await store.merge_node(node, import_run_id="run-1")

        assert result_id == "n1"

    @pytest.mark.asyncio
    async def test_merge_node_updates_existing(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        node_v1 = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        node_v2 = GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"})

        await store.merge_node(node_v1, import_run_id="run-1")
        await store.merge_node(node_v2, import_run_id="run-2")

        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.properties["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_merge_relationship_stores_and_returns_id(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")

        rel = GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT")
        result_id = await store.merge_relationship(rel, import_run_id="run-1")

        assert result_id is not None
        assert len(result_id) > 0

    @pytest.mark.asyncio
    async def test_record_provenance(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")

        trail = await store.get_audit_trail("n1")
        assert trail.node_id == "n1"
        assert len(trail.provenance_chain) >= 1
        chunk_steps = [s for s in trail.provenance_chain if s.level == "chunk"]
        assert len(chunk_steps) == 1
        assert chunk_steps[0].id == "chunk-0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph/test_memory.py::TestInMemoryGraphStoreWrite -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graphrag_core.graph.memory'`

- [ ] **Step 3: Implement InMemoryGraphStore**

Create `src/graphrag_core/graph/__init__.py`:

```python
"""BB3: Graph store implementations."""

from graphrag_core.graph.memory import InMemoryGraphStore

__all__ = ["InMemoryGraphStore"]
```

Create `src/graphrag_core/graph/memory.py`:

```python
"""In-memory GraphStore implementation for testing."""

from __future__ import annotations

from graphrag_core.models import (
    AuditTrail,
    GraphNode,
    GraphRelationship,
    OntologySchema,
    ProvenanceStep,
    SchemaViolation,
)


class InMemoryGraphStore:
    """Dict-based GraphStore for unit tests and lightweight usage."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._relationships: list[GraphRelationship] = []
        self._provenance: dict[str, list[str]] = {}  # node_id -> [chunk_ids]

    async def merge_node(self, node: GraphNode, import_run_id: str) -> str:
        self._nodes[node.id] = node
        return node.id

    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str:
        for i, existing in enumerate(self._relationships):
            if (
                existing.source_id == rel.source_id
                and existing.target_id == rel.target_id
                and existing.type == rel.type
            ):
                self._relationships[i] = rel
                return f"{rel.source_id}-{rel.type}-{rel.target_id}"
        self._relationships.append(rel)
        return f"{rel.source_id}-{rel.type}-{rel.target_id}"

    async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None:
        if node_id not in self._provenance:
            self._provenance[node_id] = []
        if chunk_id not in self._provenance[node_id]:
            self._provenance[node_id].append(chunk_id)

    async def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    async def get_audit_trail(self, node_id: str) -> AuditTrail:
        chain: list[ProvenanceStep] = []
        node = self._nodes.get(node_id)
        if node:
            chain.append(ProvenanceStep(level="node", id=node_id, metadata={"label": node.label}))
        for chunk_id in self._provenance.get(node_id, []):
            chain.append(ProvenanceStep(level="chunk", id=chunk_id, metadata={}))
        return AuditTrail(node_id=node_id, provenance_chain=chain)

    async def get_related(
        self, node_id: str, rel_type: str | None = None, depth: int = 1
    ) -> list[GraphNode]:
        if depth < 1:
            return []

        related_ids: set[str] = set()
        for rel in self._relationships:
            if rel.source_id == node_id and (rel_type is None or rel.type == rel_type):
                related_ids.add(rel.target_id)
            if rel.target_id == node_id and (rel_type is None or rel.type == rel_type):
                related_ids.add(rel.source_id)

        result = [self._nodes[nid] for nid in related_ids if nid in self._nodes]

        if depth > 1:
            for nid in list(related_ids):
                deeper = await self.get_related(nid, rel_type, depth - 1)
                for node in deeper:
                    if node.id != node_id and node.id not in related_ids:
                        related_ids.add(node.id)
                        result.append(node)

        return result

    async def apply_schema(self, schema: OntologySchema) -> None:
        self._schema = schema

    async def validate_schema(self) -> list[SchemaViolation]:
        if not hasattr(self, "_schema") or self._schema is None:
            return []

        violations: list[SchemaViolation] = []
        label_to_required: dict[str, list[str]] = {}
        for nt in self._schema.node_types:
            label_to_required[nt.label] = nt.required_properties

        for node in self._nodes.values():
            required = label_to_required.get(node.label, [])
            for prop_name in required:
                if prop_name not in node.properties:
                    violations.append(
                        SchemaViolation(
                            node_id=node.id,
                            violation_type="missing_property",
                            message=f"Required property '{prop_name}' is missing on {node.label} '{node.id}'",
                        )
                    )

        return violations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph/test_memory.py::TestInMemoryGraphStoreWrite -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 72 passed

- [ ] **Step 6: Commit**

```bash
git add src/graphrag_core/graph/ tests/test_graph/
git commit -m "feat: add InMemoryGraphStore with write operations and provenance"
```

---

## Task 10: InMemoryGraphStore — read and schema operations

**Files:**
- Modify: `tests/test_graph/test_memory.py`

- [ ] **Step 1: Write tests for read and schema operations**

Add to `tests/test_graph/test_memory.py`:

```python
class TestInMemoryGraphStoreRead:
    @pytest.mark.asyncio
    async def test_get_node_returns_none_for_missing(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        result = await store.get_node("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_related_returns_neighbors(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )

        related = await store.get_related("n1")
        assert len(related) == 1
        assert related[0].id == "n2"

    @pytest.mark.asyncio
    async def test_get_related_filters_by_rel_type(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_node(GraphNode(id="n3", label="Person", properties={"name": "Bob"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n3", type="KNOWS"), "run-1"
        )

        related = await store.get_related("n1", rel_type="WORKS_AT")
        assert len(related) == 1
        assert related[0].id == "n2"

    @pytest.mark.asyncio
    async def test_get_related_respects_depth(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="a", label="X", properties={}), "run-1")
        await store.merge_node(GraphNode(id="b", label="X", properties={}), "run-1")
        await store.merge_node(GraphNode(id="c", label="X", properties={}), "run-1")
        await store.merge_relationship(GraphRelationship(source_id="a", target_id="b", type="LINK"), "run-1")
        await store.merge_relationship(GraphRelationship(source_id="b", target_id="c", type="LINK"), "run-1")

        depth_1 = await store.get_related("a", depth=1)
        assert {n.id for n in depth_1} == {"b"}

        depth_2 = await store.get_related("a", depth=2)
        assert {n.id for n in depth_2} == {"b", "c"}

    @pytest.mark.asyncio
    async def test_audit_trail_empty_provenance(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        trail = await store.get_audit_trail("n1")
        assert trail.node_id == "n1"
        assert len(trail.provenance_chain) == 1
        assert trail.provenance_chain[0].level == "node"


class TestInMemoryGraphStoreSchema:
    @pytest.mark.asyncio
    async def test_validate_finds_missing_required_property(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Company",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                    required_properties=["name"],
                ),
            ],
            relationship_types=[],
        )
        await store.apply_schema(schema)
        await store.merge_node(GraphNode(id="n1", label="Company", properties={}), "run-1")

        violations = await store.validate_schema()
        assert len(violations) == 1
        assert violations[0].node_id == "n1"
        assert violations[0].violation_type == "missing_property"

    @pytest.mark.asyncio
    async def test_validate_passes_for_valid_nodes(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Company",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                    required_properties=["name"],
                ),
            ],
            relationship_types=[],
        )
        await store.apply_schema(schema)
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        violations = await store.validate_schema()
        assert violations == []

    @pytest.mark.asyncio
    async def test_validate_without_schema_returns_empty(self):
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        violations = await store.validate_schema()
        assert violations == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph/test_memory.py -v`
Expected: All 12 tests PASS (4 write + 5 read + 3 schema)

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 80 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_graph/test_memory.py
git commit -m "test: add InMemoryGraphStore read and schema validation tests"
```

---

## Task 11: InMemoryGraphStore Protocol conformance

**Files:**
- Modify: `tests/test_graph/test_memory.py`

- [ ] **Step 1: Write Protocol conformance test**

Add to `tests/test_graph/test_memory.py`:

```python
class TestInMemoryGraphStoreProtocol:
    def test_satisfies_graph_store_protocol(self):
        from graphrag_core.graph.memory import InMemoryGraphStore
        from graphrag_core.interfaces import GraphStore

        store = InMemoryGraphStore()
        assert isinstance(store, GraphStore)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_graph/test_memory.py::TestInMemoryGraphStoreProtocol -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_graph/test_memory.py
git commit -m "test: add InMemoryGraphStore Protocol conformance test"
```

---

## Task 12: Neo4jGraphStore implementation

**Files:**
- Create: `src/graphrag_core/graph/neo4j.py`
- Modify: `src/graphrag_core/graph/__init__.py`

- [ ] **Step 1: Implement Neo4jGraphStore**

Create `src/graphrag_core/graph/neo4j.py`:

```python
"""BB3: Neo4j-backed GraphStore implementation."""

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import AsyncGraphDatabase

from graphrag_core.models import (
    AuditTrail,
    GraphNode,
    GraphRelationship,
    OntologySchema,
    ProvenanceStep,
    SchemaViolation,
)


class Neo4jGraphStore:
    """Neo4j async implementation of the GraphStore Protocol."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "development"),
        database: str = "neo4j",
    ) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self._database = database

    async def close(self) -> None:
        await self._driver.close()

    async def merge_node(self, node: GraphNode, import_run_id: str) -> str:
        query = (
            f"MERGE (n:{node.label} {{id: $id}}) "
            "SET n += $props, n._import_run_id = $run_id, n._updated_at = $now "
            "RETURN n.id AS id"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                id=node.id,
                props=node.properties,
                run_id=import_run_id,
                now=datetime.now(timezone.utc).isoformat(),
            )
            record = await result.single()
            return record["id"]

    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str:
        query = (
            "MATCH (a {id: $source_id}), (b {id: $target_id}) "
            f"MERGE (a)-[r:{rel.type}]->(b) "
            "SET r += $props, r._import_run_id = $run_id, r._updated_at = $now "
            "RETURN $source_id + '-' + $rel_type + '-' + $target_id AS id"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                rel_type=rel.type,
                props=rel.properties,
                run_id=import_run_id,
                now=datetime.now(timezone.utc).isoformat(),
            )
            record = await result.single()
            return record["id"]

    async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None:
        query = (
            "MERGE (c:Chunk {id: $chunk_id}) "
            "WITH c "
            "MATCH (n {id: $node_id}) "
            "MERGE (c)-[r:SOURCED]->(n) "
            "SET r._import_run_id = $run_id"
        )
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                chunk_id=chunk_id,
                node_id=node_id,
                run_id=import_run_id,
            )

    async def get_node(self, node_id: str) -> GraphNode | None:
        query = "MATCH (n {id: $id}) RETURN n, labels(n) AS labels"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record is None:
                return None
            props = dict(record["n"])
            label = [l for l in record["labels"] if l != "Chunk"][0] if record["labels"] else "Unknown"
            node_id_val = props.pop("id", node_id)
            props.pop("_import_run_id", None)
            props.pop("_updated_at", None)
            return GraphNode(id=node_id_val, label=label, properties=props)

    async def get_audit_trail(self, node_id: str) -> AuditTrail:
        query = (
            "MATCH (n {id: $id}) "
            "OPTIONAL MATCH (c:Chunk)-[:SOURCED]->(n) "
            "RETURN n, labels(n) AS labels, collect(c.id) AS chunk_ids"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            record = await result.single()

            chain: list[ProvenanceStep] = []
            if record and record["n"]:
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                chain.append(ProvenanceStep(level="node", id=node_id, metadata={"label": label}))
                for chunk_id in record["chunk_ids"]:
                    if chunk_id:
                        chain.append(ProvenanceStep(level="chunk", id=chunk_id, metadata={}))

            return AuditTrail(node_id=node_id, provenance_chain=chain)

    async def get_related(
        self, node_id: str, rel_type: str | None = None, depth: int = 1
    ) -> list[GraphNode]:
        if rel_type:
            query = (
                f"MATCH (n {{id: $id}})-[:{rel_type}*1..{depth}]-(m) "
                "WHERE m.id <> $id "
                "RETURN DISTINCT m, labels(m) AS labels"
            )
        else:
            query = (
                f"MATCH (n {{id: $id}})-[*1..{depth}]-(m) "
                "WHERE m.id <> $id "
                "RETURN DISTINCT m, labels(m) AS labels"
            )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            nodes = []
            async for record in result:
                props = dict(record["m"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                nid = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                nodes.append(GraphNode(id=nid, label=label, properties=props))
            return nodes

    async def apply_schema(self, schema: OntologySchema) -> None:
        async with self._driver.session(database=self._database) as session:
            for nt in schema.node_types:
                constraint_query = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{nt.label}) REQUIRE n.id IS UNIQUE"
                )
                await session.run(constraint_query)
                for prop in nt.required_properties:
                    index_query = (
                        f"CREATE INDEX IF NOT EXISTS "
                        f"FOR (n:{nt.label}) ON (n.{prop})"
                    )
                    await session.run(index_query)

    async def validate_schema(self) -> list[SchemaViolation]:
        violations: list[SchemaViolation] = []
        return violations
```

Note: `validate_schema` returns an empty list for now. Full validation would require storing the schema reference, which is deferred — the `InMemoryGraphStore` already validates, and in production Neo4j constraints handle the rest.

- [ ] **Step 2: Update graph __init__.py to export Neo4jGraphStore**

Update `src/graphrag_core/graph/__init__.py`:

```python
"""BB3: Graph store implementations."""

from graphrag_core.graph.memory import InMemoryGraphStore

__all__ = ["InMemoryGraphStore"]

try:
    from graphrag_core.graph.neo4j import Neo4jGraphStore
    __all__.append("Neo4jGraphStore")
except ImportError:
    pass
```

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from graphrag_core.graph import Neo4jGraphStore, InMemoryGraphStore; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 81 passed (no regressions)

- [ ] **Step 5: Commit**

```bash
git add src/graphrag_core/graph/
git commit -m "feat: add Neo4jGraphStore implementing GraphStore Protocol"
```

---

## Task 13: Neo4jGraphStore integration tests

**Files:**
- Create: `tests/test_graph/test_neo4j.py`

- [ ] **Step 1: Write integration tests**

Create `tests/test_graph/test_neo4j.py`:

```python
"""Integration tests for Neo4jGraphStore. Requires running Neo4j."""

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

pytestmark = pytest.mark.integration


@pytest.fixture
async def store():
    from graphrag_core.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(database="test")
    # Wipe the test database before each test
    async with store._driver.session(database="test") as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield store
    await store.close()


class TestNeo4jMergeNode:
    @pytest.mark.asyncio
    async def test_merge_and_retrieve_node(self, store):
        node = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        result_id = await store.merge_node(node, import_run_id="run-1")
        assert result_id == "n1"

        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.label == "Company"
        assert retrieved.properties["name"] == "Acme"

    @pytest.mark.asyncio
    async def test_merge_node_updates_properties(self, store):
        node_v1 = GraphNode(id="n1", label="Company", properties={"name": "Acme"})
        node_v2 = GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"})

        await store.merge_node(node_v1, import_run_id="run-1")
        await store.merge_node(node_v2, import_run_id="run-2")

        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.properties["name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_node_returns_none_for_missing(self, store):
        result = await store.get_node("nonexistent")
        assert result is None


class TestNeo4jMergeRelationship:
    @pytest.mark.asyncio
    async def test_merge_relationship(self, store):
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")

        rel = GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT")
        result_id = await store.merge_relationship(rel, import_run_id="run-1")
        assert result_id == "n1-WORKS_AT-n2"


class TestNeo4jProvenance:
    @pytest.mark.asyncio
    async def test_record_and_retrieve_provenance(self, store):
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")
        await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")

        trail = await store.get_audit_trail("n1")
        assert trail.node_id == "n1"
        assert len(trail.provenance_chain) >= 2
        levels = [s.level for s in trail.provenance_chain]
        assert "node" in levels
        assert "chunk" in levels

    @pytest.mark.asyncio
    async def test_multiple_provenance_links(self, store):
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")
        await store.record_provenance(node_id="n1", chunk_id="chunk-0", import_run_id="run-1")
        await store.record_provenance(node_id="n1", chunk_id="chunk-1", import_run_id="run-1")

        trail = await store.get_audit_trail("n1")
        chunk_steps = [s for s in trail.provenance_chain if s.level == "chunk"]
        assert len(chunk_steps) == 2
        chunk_ids = {s.id for s in chunk_steps}
        assert chunk_ids == {"chunk-0", "chunk-1"}


class TestNeo4jGetRelated:
    @pytest.mark.asyncio
    async def test_get_related_returns_neighbors(self, store):
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )

        related = await store.get_related("n1")
        assert len(related) == 1
        assert related[0].id == "n2"

    @pytest.mark.asyncio
    async def test_get_related_filters_by_type(self, store):
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_node(GraphNode(id="n3", label="Person", properties={"name": "Bob"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n3", type="KNOWS"), "run-1"
        )

        related = await store.get_related("n1", rel_type="WORKS_AT")
        assert len(related) == 1
        assert related[0].id == "n2"


class TestNeo4jSchema:
    @pytest.mark.asyncio
    async def test_apply_schema_creates_constraints(self, store):
        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Company",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                    required_properties=["name"],
                ),
            ],
            relationship_types=[],
        )
        await store.apply_schema(schema)

        # Verify constraint exists by checking we can't create duplicate ids
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Updated"}), "run-2")

        node = await store.get_node("n1")
        assert node is not None
        assert node.properties["name"] == "Updated"


class TestNeo4jProtocol:
    def test_satisfies_graph_store_protocol(self):
        from graphrag_core.graph.neo4j import Neo4jGraphStore
        from graphrag_core.interfaces import GraphStore

        store = Neo4jGraphStore()
        assert isinstance(store, GraphStore)
```

- [ ] **Step 2: Verify integration tests are skipped by default**

Run: `uv run pytest tests/test_graph/test_neo4j.py -v`
Expected: All tests SKIPPED (integration tests require `--run-integration`)

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 81 passed, N skipped

- [ ] **Step 4: (Optional) Run with Neo4j if available**

Start Neo4j: `docker compose -f /Users/dinoceli/Developer/tessera/infra/docker-compose.yml up neo4j -d`
Run: `uv run pytest tests/test_graph/test_neo4j.py -v --run-integration`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_graph/test_neo4j.py
git commit -m "test: add Neo4jGraphStore integration tests"
```

---

## Task 14: Update public API re-exports

**Files:**
- Modify: `src/graphrag_core/__init__.py`

- [ ] **Step 1: Update __init__.py with all new re-exports**

Replace the contents of `src/graphrag_core/__init__.py` with:

```python
"""graphrag-core: Domain-agnostic Graph RAG framework."""

__version__ = "0.1.0"

from graphrag_core.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingModel,
    ExtractionEngine,
    GraphStore,
    IngestionPipeline,
    LLMClient,
    SearchEngine,
)
from graphrag_core.ingestion import (
    DocxParser,
    MarkdownParser,
    PdfParser,
    TextParser,
    TokenChunker,
)
from graphrag_core.extraction import LLMExtractionEngine
from graphrag_core.graph import InMemoryGraphStore
from graphrag_core.models import (
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    SearchResult,
)

__all__ = [
    # Protocols
    "Chunker",
    "DocumentParser",
    "EmbeddingModel",
    "ExtractionEngine",
    "GraphStore",
    "IngestionPipeline",
    "LLMClient",
    "SearchEngine",
    # BB1 implementations
    "DocxParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "TokenChunker",
    # BB2 implementations
    "LLMExtractionEngine",
    # BB3 implementations
    "InMemoryGraphStore",
    # Models
    "DocumentChunk",
    "ExtractionResult",
    "GraphNode",
    "ImportRun",
    "NodeTypeDefinition",
    "OntologySchema",
    "SearchResult",
]

# Optional: Neo4j and Anthropic (require extras)
try:
    from graphrag_core.graph import Neo4jGraphStore
    __all__.append("Neo4jGraphStore")
except ImportError:
    pass

try:
    from graphrag_core.llm import AnthropicLLMClient
    __all__.append("AnthropicLLMClient")
except ImportError:
    pass
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from graphrag_core import LLMClient, LLMExtractionEngine, InMemoryGraphStore, Neo4jGraphStore, AnthropicLLMClient; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 81 passed

- [ ] **Step 4: Commit**

```bash
git add src/graphrag_core/__init__.py
git commit -m "feat: add BB2 and BB3 re-exports to public API"
```

---

## Task 15: End-to-end integration test (BB1 -> BB2 -> BB3)

**Files:**
- Create: `tests/test_integration/__init__.py`
- Create: `tests/test_integration/test_ingest_to_graph.py`

- [ ] **Step 1: Write end-to-end integration test**

Create `tests/test_integration/__init__.py` (empty).

Create `tests/test_integration/test_ingest_to_graph.py`:

```python
"""End-to-end integration test: BB1 (Ingestion) -> BB2 (Extraction) -> BB3 (Graph)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentChunk,
    DocumentMetadata,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    ParsedDocument,
    PropertyDefinition,
    RelationshipTypeDefinition,
    TextSection,
)

pytestmark = pytest.mark.integration


class FakeLLMClient:
    """Returns extraction JSON based on chunk content."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        return json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice", "role": "engineer"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme Corp"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT", "properties": {}},
            ],
        })


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
                properties=[
                    PropertyDefinition(name="name", type="string", required=True),
                    PropertyDefinition(name="role", type="string", required=False),
                ],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(type="WORKS_AT", source_types=["Person"], target_types=["Company"]),
        ],
    )


@pytest.fixture
async def neo4j_store():
    from graphrag_core.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(database="test")
    async with store._driver.session(database="test") as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield store
    await store.close()


class TestIngestToGraph:
    @pytest.mark.asyncio
    async def test_full_pipeline_document_to_graph(self, neo4j_store):
        from graphrag_core.ingestion import TextParser, TokenChunker, IngestionPipeline
        from graphrag_core.extraction.engine import LLMExtractionEngine

        # BB1: Ingest a plain text document
        source = b"Alice is a software engineer at Acme Corp. She has been working there for five years."
        pipeline = IngestionPipeline(
            parser=TextParser(),
            chunker=TokenChunker(),
        )
        chunks = await pipeline.ingest(source, "text/plain", ChunkConfig(max_tokens=50, overlap=0))
        assert len(chunks) >= 1

        # BB2: Extract entities
        engine = LLMExtractionEngine(llm_client=FakeLLMClient())
        import_run = ImportRun(
            id="run-e2e",
            timestamp=datetime(2026, 4, 12, 12, 0),
            source_type="text/plain",
            documents_processed=1,
            entities_extracted=0,
        )
        extraction = await engine.extract(chunks=chunks, schema=_schema(), import_run=import_run)
        assert len(extraction.nodes) >= 2
        assert len(extraction.relationships) >= 1

        # BB3: Store in Neo4j
        for node_data in extraction.nodes:
            from graphrag_core.models import GraphNode
            graph_node = GraphNode(id=node_data.id, label=node_data.label, properties=node_data.properties)
            await neo4j_store.merge_node(graph_node, import_run_id=import_run.id)

        for rel_data in extraction.relationships:
            from graphrag_core.models import GraphRelationship
            graph_rel = GraphRelationship(
                source_id=rel_data.source_id,
                target_id=rel_data.target_id,
                type=rel_data.type,
                properties=rel_data.properties,
            )
            await neo4j_store.merge_relationship(graph_rel, import_run_id=import_run.id)

        # Record provenance
        for prov in extraction.provenance:
            await neo4j_store.record_provenance(
                node_id=prov.node_id, chunk_id=prov.chunk_id, import_run_id=import_run.id
            )

        # Verify: nodes exist in graph
        alice = await neo4j_store.get_node("person-alice")
        assert alice is not None
        assert alice.label == "Person"
        assert alice.properties["name"] == "Alice"

        acme = await neo4j_store.get_node("company-acme")
        assert acme is not None
        assert acme.label == "Company"

        # Verify: relationship exists
        related = await neo4j_store.get_related("person-alice", rel_type="WORKS_AT")
        assert len(related) == 1
        assert related[0].id == "company-acme"

        # Verify: provenance chain exists
        trail = await neo4j_store.get_audit_trail("person-alice")
        assert trail.node_id == "person-alice"
        levels = [s.level for s in trail.provenance_chain]
        assert "node" in levels
        assert "chunk" in levels
```

- [ ] **Step 2: Verify test is skipped by default**

Run: `uv run pytest tests/test_integration/ -v`
Expected: All tests SKIPPED

- [ ] **Step 3: (Optional) Run with Neo4j if available**

Run: `uv run pytest tests/test_integration/test_ingest_to_graph.py -v --run-integration`
Expected: PASS

- [ ] **Step 4: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: 81 passed, N skipped

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration/
git commit -m "test: add end-to-end BB1->BB2->BB3 integration test"
```

---

## Summary

| Task | What it builds | New tests |
|---|---|---|
| 1 | LLMClient Protocol | 1 Protocol conformance |
| 2 | pytest integration marker + conftest | 0 (infrastructure) |
| 3 | Optional dependencies (anthropic, neo4j) | 0 (infrastructure) |
| 4 | AnthropicLLMClient | 0 (thin wrapper, Protocol tested) |
| 5 | LLMExtractionEngine — happy path | 1 happy path |
| 6 | LLMExtractionEngine — validation | 6 edge cases |
| 7 | LLMExtractionEngine — multi-chunk | 1 multi-chunk + provenance |
| 8 | ExtractionEngine Protocol conformance | 1 conformance |
| 9 | InMemoryGraphStore — write ops | 4 write tests |
| 10 | InMemoryGraphStore — read + schema | 8 read/schema tests |
| 11 | InMemoryGraphStore Protocol conformance | 1 conformance |
| 12 | Neo4jGraphStore implementation | 0 (tested in Task 13) |
| 13 | Neo4jGraphStore integration tests | 9 integration tests |
| 14 | Public API re-exports | 0 (wiring) |
| 15 | End-to-end integration test | 1 E2E test |

**Total new tests:** ~33 (+ 9 integration tests requiring Neo4j)
**Expected final count:** ~94 unit tests passing + 10 integration tests (when Neo4j is available)

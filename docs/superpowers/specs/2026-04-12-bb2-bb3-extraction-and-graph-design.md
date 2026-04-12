# BB2 + BB3: Schema-Guided Extraction & Provenance-Native Graph Store

**Date:** 2026-04-12
**Status:** Approved
**Goal:** Implement BB2 (ExtractionEngine) and BB3 (Neo4jGraphStore) to create a working ingest-to-graph pipeline.

---

## Decisions

| Decision | Rationale |
|---|---|
| LLMClient Protocol + AnthropicLLMClient default | Provider-agnostic; supports Claude, Ollama, Azure Foundry, LM Studio via future implementations |
| Strict schema validation in extraction | Off-schema results are dropped. Discovery is a Layer 2 concern (broader schema, two-pass extraction) |
| InMemoryGraphStore for unit tests | Fast, no Docker. Reusable test fixture for all future modules |
| Real Neo4j for BB3 integration tests | BB3 *is* the Neo4j layer; mocking it tests nothing |
| Parallel development (BB2 + BB3) | Independent behind the GraphStore Protocol. InMemoryGraphStore bridges the gap |
| Sequential chunk processing in v1 | Concurrent LLM calls are a performance optimization, deferred |
| Optional extras for heavy deps | `anthropic` and `neo4j` are opt-in via `pip install graphrag-core[anthropic,neo4j]` |

---

## Section 1: New Interfaces

### LLMClient Protocol (added to `interfaces.py`)

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

Minimal surface: messages in, string out. No tool use, no streaming. Provider-specific features live in concrete implementations, not the interface.

No changes to the existing `ExtractionEngine` or `GraphStore` Protocols.

---

## Section 2: BB2 — LLMExtractionEngine

### File: `src/graphrag_core/extraction/engine.py`

**Constructor:**
```python
class LLMExtractionEngine:
    def __init__(self, llm_client: LLMClient) -> None:
```

Schema is passed per-call via `extract()`, not at construction. Allows the same engine instance to serve different schemas.

**Extraction flow:**

1. Build a system prompt encoding the `OntologySchema` as extraction instructions:
   - All allowed node types with their properties (marking which are required)
   - All allowed relationship types with source/target type constraints
   - Explicit instruction: "Do not extract entities or relationships not listed above"
   - Output format: JSON with `nodes` and `relationships` arrays
2. For each chunk, send chunk text + system prompt to the LLM via `LLMClient.complete()`
3. Parse JSON response into `ExtractedNode` and `ExtractedRelationship` instances
4. **Strict validation:**
   - Drop nodes whose `label` is not in `schema.node_types`
   - Drop relationships whose `type` is not in `schema.relationship_types`
   - Drop relationships whose source/target labels violate `source_types`/`target_types` constraints
   - Drop relationships referencing dropped nodes
5. Build `ProvenanceLink` entries: each extracted node maps back to its source chunk ID
6. Aggregate across all chunks into a single `ExtractionResult`

**Re-exports:** `src/graphrag_core/extraction/__init__.py` exports `LLMExtractionEngine`.

### Testing (unit, no infrastructure)

Tests use a `FakeLLMClient` returning canned JSON:

| Test case | Assertion |
|---|---|
| Happy path (valid JSON, schema-conformant) | Correct `ExtractionResult` with expected nodes, rels, provenance |
| Off-schema node type | Node dropped from result |
| Off-schema relationship type | Relationship dropped from result |
| Dangling relationship (references dropped node) | Relationship also dropped |
| Malformed JSON from LLM | Graceful handling (empty result or exception) |
| Empty chunks list | Empty `ExtractionResult` |
| Provenance links | Each node maps to its source chunk |

---

## Section 3: AnthropicLLMClient

### File: `src/graphrag_core/llm/anthropic.py`

```python
class AnthropicLLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
```

Thin wrapper around the `anthropic` AsyncAnthropic SDK:
- Maps `messages` and `system` to Anthropic API format
- Passes through `temperature` and `max_tokens`
- Returns `response.content[0].text`
- No retry logic in v1

**Re-exports:** `src/graphrag_core/llm/__init__.py` exports `AnthropicLLMClient`.

### Dependency

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
```

No unit tests for the client — it's a thin wrapper. The `FakeLLMClient` in extraction tests validates the Protocol shape.

---

## Section 4: BB3 — Neo4jGraphStore

### File: `src/graphrag_core/graph/neo4j.py`

```python
class Neo4jGraphStore:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "development"),
        database: str = "neo4j",
    ) -> None:
```

Uses `neo4j` async driver (`AsyncGraphDatabase`). Implements all 8 `GraphStore` Protocol methods.

**Write operations:**

- `merge_node()` — `MERGE` on `(id, label)`, sets all properties + `_import_run_id` and `_updated_at` metadata. Returns node ID.
- `merge_relationship()` — `MATCH` source/target by ID, `MERGE` relationship by type, sets properties + metadata. Returns relationship identifier.
- `record_provenance()` — `MERGE`s a `(:Chunk {id})` node if it doesn't exist, then creates `(:Chunk)-[:SOURCED]->(node)` with `import_run_id`. Provenance backbone. Higher-level provenance (chunk -> document -> source) is populated by the ingestion pipeline when storing chunks, not by `record_provenance()` itself.

**Read operations:**

- `get_node()` — `MATCH (n {id: $id}) RETURN n`. Returns `GraphNode | None`.
- `get_audit_trail()` — Traverses provenance chain: node <- chunk <- document <- source. Returns `AuditTrail` with `ProvenanceStep` per level.
- `get_related()` — Variable-length path query up to `depth` hops, optionally filtered by `rel_type`.

**Schema operations:**

- `apply_schema()` — Creates uniqueness constraints on `id` per node type. Creates indexes for required properties. Idempotent.
- `validate_schema()` — Queries for missing required properties and source/target type violations. Returns `list[SchemaViolation]`.

### Dependency

```toml
[project.optional-dependencies]
anthropic = ["anthropic>=0.40"]
neo4j = ["neo4j>=5.0"]
all = ["graphrag-core[anthropic,neo4j]"]
```

### InMemoryGraphStore

### File: `src/graphrag_core/graph/memory.py`

Dict-based `GraphStore` implementation. Stores nodes and relationships in plain Python dicts. Purposes:
- BB2 extraction tests (mock graph target)
- Lightweight graph backend for any future module's tests
- Protocol shape validation

**Re-exports:** `src/graphrag_core/graph/__init__.py` exports `Neo4jGraphStore` and `InMemoryGraphStore`.

### Testing — two tiers

**Tier 1: InMemoryGraphStore unit tests** (`tests/test_graph/test_memory.py`)
- All 8 Protocol methods tested against the in-memory implementation
- Provenance chain construction and traversal
- Schema validation logic

**Tier 2: Neo4jGraphStore integration tests** (`tests/test_graph/test_neo4j.py`)
- Marked `@pytest.mark.integration`, skipped by default
- Uses `test` database, wiped between tests via fixture
- All 8 Protocol methods against real Neo4j
- Provenance chain traversal end-to-end
- Schema constraint creation and violation detection
- `conftest.py` fixture checks for running Neo4j, skips gracefully if unavailable

---

## Section 5: End-to-End Integration

### File: `tests/test_integration/test_ingest_to_graph.py`

Wires BB1 -> BB2 -> BB3:

```
PDF bytes -> PdfParser -> TokenChunker -> LLMExtractionEngine -> Neo4jGraphStore
```

1. Parse a test PDF into chunks (BB1)
2. Extract entities with `LLMExtractionEngine` + `FakeLLMClient` (BB2, deterministic)
3. Merge nodes/relationships into `Neo4jGraphStore` (BB3)
4. Record provenance links
5. Assert: nodes exist, relationships exist, `get_audit_trail()` returns valid chain

Marked `@pytest.mark.integration`. Uses `FakeLLMClient` (no real API calls), but requires Neo4j.

---

## Public API Updates

**`interfaces.py`:** Add `LLMClient` Protocol.

**`__init__.py` (top-level):** Add re-exports:
- `LLMExtractionEngine` from `extraction`
- `Neo4jGraphStore`, `InMemoryGraphStore` from `graph`
- `AnthropicLLMClient` from `llm`

---

## New File Tree (additions only)

```
src/graphrag_core/
├── llm/
│   ├── __init__.py              # re-exports AnthropicLLMClient
│   └── anthropic.py             # AnthropicLLMClient
├── extraction/
│   ├── __init__.py              # re-exports LLMExtractionEngine
│   └── engine.py                # LLMExtractionEngine
├── graph/
│   ├── __init__.py              # re-exports Neo4jGraphStore, InMemoryGraphStore
│   ├── neo4j.py                 # Neo4jGraphStore
│   └── memory.py                # InMemoryGraphStore
tests/
├── test_extraction/
│   ├── __init__.py
│   └── test_engine.py           # FakeLLMClient, unit tests
├── test_graph/
│   ├── __init__.py
│   ├── test_memory.py           # InMemoryGraphStore unit tests
│   └── test_neo4j.py            # @integration, real Neo4j
├── test_integration/
│   ├── __init__.py
│   └── test_ingest_to_graph.py  # @integration, end-to-end
└── conftest.py                  # Neo4j fixture, integration marker
```

---

## Not Included (Deferred)

- Concurrent chunk processing (performance optimization)
- Retry/backoff logic on LLM calls
- Additional LLMClient implementations (Ollama, Azure Foundry, LM Studio)
- BB4 SearchEngine concrete implementation
- BB5-BB8 interfaces and implementations

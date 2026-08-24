# graphrag-core — Interface Specification

**Version:** 0.2.0-spec (revised 2026-05-15)
**License:** MIT
**Status:** Draft — revised to reflect BB7 4-of-8 split and pending push-down (audit decision E1)

> A domain-agnostic framework for building governed, auditable Knowledge Graphs from documents using LLM-powered extraction, provenance-native storage, and an agent-callable tool contract (consumed by external agents over MCP).

---

## Overview

graphrag-core provides 8 building blocks for enterprise-grade Graph RAG applications. Each block defines abstract interfaces that can be implemented with different backends. The framework ships with default implementations for Neo4j, LangGraph, and common LLM providers.

```
pip install graphrag-core                    # core interfaces + Neo4j default
pip install graphrag-core[langgraph]         # + LangGraph orchestrator
pip install graphrag-core[all]               # all default implementations
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  YOUR DOMAIN LAYER (Layer 2)                             │
│  Ontology, domain tools, domain agents, templates        │
│  You build this. graphrag-core doesn't touch it.         │
└────────────────────────┬────────────────────────────────┘
                         │ imports
┌────────────────────────▼────────────────────────────────┐
│  graphrag-core (Layer 1)                                 │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Ingestion│ │Extraction│ │  Graph   │ │  Search    │ │
│  │ Pipeline │ │  Engine  │ │  Store   │ │  Engine    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Curation │ │ Entity   │ │  Tool    │ │ Orchestr.  │ │
│  │ Pipeline │ │ Registry │ │ Library  │ │ + Report   │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Building Block 1: Document Ingestion Pipeline

Accepts raw documents, parses, chunks, and prepares them for entity extraction.

### Interface

```python
from graphrag_core.ingestion import DocumentParser, Chunker, IngestionPipeline

class DocumentParser(Protocol):
    """Converts raw file bytes into structured text sections."""
    async def parse(self, source: bytes, content_type: str) -> ParsedDocument: ...

class Chunker(Protocol):
    """Splits parsed documents into semantic chunks."""
    def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> list[DocumentChunk]: ...

class IngestionPipeline:
    """Orchestrates parse → chunk → embed → store."""
    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_model: EmbeddingModel,
        graph_store: GraphStore,
    ): ...

    async def ingest(self, source: bytes, metadata: DocumentMetadata) -> ImportRun: ...
```

### Data Models

```python
class ParsedDocument(BaseModel):
    sections: list[TextSection]
    metadata: DocumentMetadata

class DocumentChunk(BaseModel):
    id: str
    text: str
    embedding: list[float] | None = None
    page: int | None = None
    position: int | None = None
    chunk_type: str = "text"

class DocumentMetadata(BaseModel):
    title: str
    source: str
    doc_type: str        # pdf, xlsx, docx, email, transcript
    date: date | None
    period: str | None   # canonical doc-time field, lexically-sortable (e.g. "2025-Q4")
    sha256: str

class ImportRun(BaseModel):
    id: str
    timestamp: datetime
    source_type: str
    documents_processed: int
    entities_extracted: int
```

---

## Building Block 2: Schema-Guided Entity Extraction

Extracts entities and relationships from text, guided by a domain ontology schema.

### Interface

```python
from graphrag_core.extraction import ExtractionEngine, OntologySchema

class OntologySchema(BaseModel):
    """Defines expected node types, properties, and relationship types."""
    node_types: list[NodeTypeDefinition]
    relationship_types: list[RelationshipTypeDefinition]

class NodeTypeDefinition(BaseModel):
    label: str
    properties: list[PropertyDefinition]
    required_properties: list[str] = []

class ExtractionEngine(Protocol):
    """Extracts entities and relations from text guided by schema."""
    async def extract(
        self,
        chunks: list[DocumentChunk],
        schema: OntologySchema,
        import_run: ImportRun,
    ) -> ExtractionResult: ...

class ExtractionResult(BaseModel):
    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]
    provenance: list[ProvenanceLink]  # chunk → node mapping
```

---

## Building Block 3: Provenance-Native Knowledge Graph

Graph storage where every node and edge is traceable to its source.

### Interface

```python
from graphrag_core.graph import GraphStore

class GraphStore(Protocol):
    """Abstract graph storage with provenance as core architecture."""

    # Write
    async def merge_node(self, node: GraphNode, import_run_id: str) -> str: ...
    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str: ...
    async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None: ...

    # Read
    async def get_node(self, node_id: str) -> GraphNode | None: ...
    async def get_audit_trail(self, node_id: str) -> AuditTrail: ...
    async def get_related(
        self, node_id: str, rel_type: str | None = None, depth: int = 1
    ) -> list[GraphNode]: ...

    # Schema
    async def apply_schema(self, schema: OntologySchema) -> None: ...
    async def validate_schema(self) -> list[SchemaViolation]: ...

class AuditTrail(BaseModel):
    node_id: str
    provenance_chain: list[ProvenanceStep]

class ProvenanceStep(BaseModel):
    level: str  # "node", "chunk", "document", "source"
    id: str
    metadata: dict[str, Any]
```

---

## Building Block 4: Hybrid Search Engine

Combines vector similarity, graph traversal, and fulltext search.

### Interface

```python
from graphrag_core.search import SearchEngine, SearchResult

class SearchEngine(Protocol):
    """Multi-modal search across the knowledge graph."""

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]: ...

    async def fulltext_search(
        self, query: str, node_types: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]: ...

    async def graph_search(
        self, start_node_id: str, pattern: str, depth: int = 2
    ) -> list[SearchResult]: ...

    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]: ...
```

---

## Building Block 5: Governed Curation — retired (merged into BB3)

> **Retired.** The BB5 governed-curation *seat* was retired by [ADR-0039](https://github.com/cdel1/tessera/blob/main/docs/adr/0039-bb-taxonomy-realignment-pipeline-vs-infrastructure.md) (D3), and its L1 Python was removed by [ADR-0038](https://github.com/cdel1/tessera/blob/main/docs/adr/0038-curation-substrate-l3-contract-only.md) (shipped v0.12.0). The three-layer QA pipeline — the `DetectionLayer` / `LLMCurationLayer` / `ApprovalGateway` Protocols, the `DeterministicDetectionLayer` implementation, the `CurationPipeline` orchestrator, and the `CurationIssue` / `CurationReport` / `ApprovalBatch` / `ApplyResult` models — had zero L2 consumers and was deleted. Graph-quality QA is play-shaped and lives at L2.
>
> What survives is the **Layer-3 attestation contract**, now owned by BB3 (Knowledge Graph): *Layer 3 is a contract, not a Protocol shape.* Every L2 surface that mutates the graph at Tier 3 must emit a promotion event recording attestor kind (`human` | `agent`), attestor id, rationale, supporting excerpts, timestamp, and an immutable audit record of the mutation. No L1 Python enforces it. The contract text lives in `graph/INTERFACE.md` § *Layer 3 — the attestation contract*.

---

## Building Block 6: Known Entity Registry

Pre-seeds known entities and prevents duplicates during extraction.

### Interface

```python
from graphrag_core.registry import EntityRegistry

class EntityRegistry(Protocol):
    """Manages known entities for deduplication during extraction."""

    async def register(self, entity: KnownEntity) -> str: ...
    async def lookup(
        self, name: str, entity_type: str, match_strategy: str = "fuzzy"
    ) -> list[RegistryMatch]: ...
    async def bulk_register(self, entities: list[KnownEntity]) -> int: ...

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

## Building Block 7: Core Tool Library (Semantic Layer)

Pre-defined, tested graph queries exposed as callable tools for agents.

### Interface

```python
from graphrag_core.tools import ToolLibrary, Tool, ToolResult

class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, ToolParameter]
    handler: Callable[..., Awaitable[ToolResult]]

class ToolLibrary:
    """Registry of tested, schema-validated graph query tools."""

    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def list_tools(self) -> list[Tool]: ...
    async def execute(self, name: str, **kwargs) -> ToolResult: ...

# Core tools — current shipping state (as of 2026-05-15):
CORE_TOOLS_SHIPPED = [
    "get_entity",             # (entity_id) → Node
    "search_entities",        # (query, node_types, top_k) → list[SearchResult]
    "get_audit_trail",        # (node_id) → AuditTrail
    "get_related",             # (node_id, rel_type, depth) → list[Node]
]

# Temporal tools — currently implemented in Lacuna (Layer 2);
# scheduled for push-down to graphrag-core BB7 before the next PyPI release
# (audit decision E1). Naming may differ slightly post-push-down
# (e.g. `compare_periods` is more general than `compare_quarters`).
CORE_TOOLS_PENDING_PUSHDOWN = [
    "get_entity_history",     # (entity_id, from_period, to_period) → EntityHistory
    "compare_periods",        # (topic_id, period_from, period_to) → PeriodDiff
    "find_unaddressed_topics",# (period) → list[Topic]
    "find_trend",             # (topic_id) → TrendSignal
]
```

The 8-tool framing was the v0.1.0 spec target. Current shipping state: 4 generic core tools in graphrag-core; 4 temporal tools in Lacuna pending push-down. The temporal tools are domain-agnostic (any Layer 2 consumer with period-tagged claims benefits) and pass the Push-Down Rule; the push-down is scheduled before the next graphrag-core PyPI release.

Domain-specific tools (e.g. a gap-finder over declared vs. observed state, report-section and summary generators, lens/scope filters) are registered by the Layer 2 application and stay in the consumer, not in graphrag-core.

---

## Building Block 8: Multi-Agent Orchestration & Report Generation — retired

> **Retired.** The BB8 seat was retired by [ADR-0039](https://github.com/cdel1/tessera/blob/main/docs/adr/0039-bb-taxonomy-realignment-pipeline-vs-infrastructure.md) and its L1 Python removed in v0.13.0. The `Agent` / `Orchestrator` Protocols, `SequentialOrchestrator`, `AgentContext`, the `ReportRenderer` Protocol, and the `AgentResult` / `WorkflowResult` / `ReportData` / `RenderConfig` models are gone — they had no production consumers and were superseded by two doctrines: the TfT human-orchestrator model, and the agentic-substrate model in which **agents are external** (Claude Code, MCP clients, custom harnesses) and drive the framework by consuming its `ToolLibrary` (BB7) over MCP. graphrag-core ships the tool contract, not an agent runtime; report rendering is an L2 concern. See [`2026-05-15-agentic-substrate-design.md`](https://github.com/cdel1/tessera/blob/main/docs/specs/2026-05-15-agentic-substrate-design.md) §4.1.

---

## Extension Points

graphrag-core is designed to be extended, not forked.

### Adding a Domain (Layer 2)

```python
from graphrag_core.extraction import OntologySchema
from graphrag_core.tools import ToolLibrary

# 1. Define your domain ontology (example: a legal-compliance domain)
compliance_schema = OntologySchema(
    node_types=[
        NodeTypeDefinition(label="Obligation", properties=[...]),
        NodeTypeDefinition(label="Control", properties=[...]),
    ],
    relationship_types=[...]
)

# 2. Register domain-specific tools
tool_library.register(Tool(
    name="find_unmet_obligations",
    description="Find obligations with no satisfying control",
    parameters={...},
    handler=find_unmet_obligations_handler
))

# 3. Expose the tools to external agents over MCP
#    graphrag-core ships no agent runtime — external agents (Claude Code, MCP
#    clients, custom harnesses) drive the pipeline by calling the ToolLibrary.
#    See 2026-05-15-agentic-substrate-design.md §4.1.
```

### Swapping a Backend

```python
from graphrag_core.graph import GraphStore

# Implement the interface for a different backend
class TigerGraphStore(GraphStore):
    async def merge_node(self, node, import_run_id): ...
    async def get_audit_trail(self, node_id): ...
    # ... etc.

# Use it
pipeline = IngestionPipeline(
    parser=PdfParser(),
    chunker=SemanticChunker(),
    embedding_model=NomicEmbedding(),
    graph_store=TigerGraphStore(config),  # swapped
)
```

---

## Default Implementations

graphrag-core ships with production-ready defaults. **Honest current state (2026-05-15):**

| Interface | Default Implementation | Package | Shipping status |
|---|---|---|---|
| `GraphStore` | `Neo4jGraphStore` (+ `MemoryGraphStore` for tests) | `graphrag-core` | Shipped |
| `ExtractionEngine` | `LLMExtractionEngine` | `graphrag-core` | Shipped |
| `ExtractionPromptBuilder` | `DefaultPromptBuilder` | `graphrag-core` | Shipped |
| `ExtractionPostProcessor` (Protocol only) | n/a | `graphrag-core` | Protocol shipped; default impl is domain concern |
| `SearchEngine` | `Neo4jHybridSearch` (+ `MemorySearch` for tests) | `graphrag-core` | Shipped |
| `EmbeddingModel` (Protocol only) | n/a (named `NomicEmbedding` in v0.1.0 spec; not yet implemented) | — | **Protocol only** |
| ~~`DetectionLayer`~~ | `GDSDetectionLayer` was named in v0.1.0 spec, never implemented | — | **Removed** (ADR-0038, v0.12.0) |
| ~~`LLMCurationLayer`~~ | n/a | — | **Removed** (ADR-0038, v0.12.0) |
| ~~`ApprovalGateway`~~ | `CLIApprovalGateway` was named in v0.1.0 spec, never implemented | — | **Removed** (ADR-0038, v0.12.0) |
| `EntityRegistry` | `MemoryEntityRegistry` | `graphrag-core` | Shipped (in-memory; no Neo4j-backed registry yet) |
| `ToolLibrary` | `ToolLibrary` + 4 core tools (see above) | `graphrag-core` | Shipped (partial — 4 of 8 tools) |
| ~~`Orchestrator`~~ | `SequentialOrchestrator` (+ `Agent`, `AgentContext`) | — | **Removed** (ADR-0039, v0.13.0) |
| ~~`ReportRenderer`~~ | `DocxRenderer` was named in v0.1.0 spec, never implemented | — | **Removed** (ADR-0039, v0.13.0) |
| `CommunityDetector` (Protocol only) | implemented in Lacuna (`LeidenCommunityDetector` via graspologic) | Lacuna L2 | Protocol shipped; default impl in Lacuna |

**v0.2.0 spec correction:** v0.1.0 listed default implementations that were aspirational. v0.2.0 spec separates "Protocol shipped" from "default impl shipped." `EmbeddingModel` ships as an interface with no default implementation in graphrag-core yet — consumers must supply their own or implement against the Protocol. (`DetectionLayer`, `ApprovalGateway`, and `ReportRenderer`, also listed here originally, were later removed entirely per ADR-0038 / ADR-0039.)

---

## Project Structure

```
graphrag-core/
├── src/
│   └── graphrag_core/
│       ├── __init__.py
│       ├── interfaces.py          # All Protocol definitions
│       ├── models.py              # All Pydantic models
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pipeline.py        # IngestionPipeline
│       │   ├── parsers.py         # PdfParser, DocxParser, ...
│       │   └── chunkers.py        # SemanticChunker, ...
│       ├── extraction/
│       │   ├── __init__.py
│       │   ├── engine.py          # LLMExtractionEngine
│       │   └── schema.py          # OntologySchema, NodeTypeDefinition
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── store.py           # GraphStore Protocol
│       │   ├── neo4j.py           # Neo4jGraphStore
│       │   └── provenance.py      # AuditTrail, ProvenanceChain
│       ├── search/
│       │   ├── __init__.py
│       │   └── hybrid.py          # Neo4jHybridSearch
│       ├── registry/
│       │   ├── __init__.py
│       │   └── known_entities.py  # EntityRegistry
│       └── tools/
│           ├── __init__.py
│           ├── library.py         # ToolLibrary
│           └── core_tools.py      # 8 core tools
├── tests/
│   ├── test_ingestion.py
│   ├── test_extraction.py
│   ├── test_graph.py
│   ├── test_tools.py
│   └── conftest.py
├── pyproject.toml
├── LICENSE                        # MIT
├── README.md
└── CLAUDE.md
```

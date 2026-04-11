# graphrag-core — Interface Specification

**Version:** 0.1.0
**License:** MIT
**Status:** Draft

> A domain-agnostic framework for building governed, auditable Knowledge Graphs from documents using LLM-powered extraction, provenance-native storage, and multi-agent orchestration.

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
    quarter: str | None  # e.g. "Q4/2025"
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

## Building Block 5: Governed Curation Pipeline

Three-layer quality assurance: deterministic checks → LLM suggestions → human approval.

### Interface

```python
from graphrag_core.curation import (
    CurationPipeline, DetectionLayer, LLMCurationLayer, ApprovalGateway
)

class CurationIssue(BaseModel):
    id: str
    issue_type: str          # "duplicate", "orphan", "schema_violation", "merge_candidate"
    severity: str            # "info", "warning", "error"
    affected_nodes: list[str]
    suggested_action: str
    auto_fixable: bool
    source_layer: str        # "deterministic", "llm"

class DetectionLayer(Protocol):
    """Layer 1: Deterministic, zero LLM cost."""
    async def detect(self, import_run_id: str) -> list[CurationIssue]: ...

class LLMCurationLayer(Protocol):
    """Layer 2: LLM-based suggestions (entity resolution, relevance)."""
    async def curate(self, issues: list[CurationIssue]) -> list[CurationIssue]: ...

class ApprovalGateway(Protocol):
    """Layer 3: Human approval for high-impact operations."""
    async def submit_for_approval(self, issues: list[CurationIssue]) -> str: ...  # returns batch_id
    async def get_approval_status(self, batch_id: str) -> ApprovalBatch: ...
    async def apply_approved(self, batch_id: str) -> ApplyResult: ...

class CurationPipeline:
    """Orchestrates the 3-layer flow."""
    def __init__(
        self,
        detection: DetectionLayer,
        llm_curation: LLMCurationLayer,
        approval: ApprovalGateway,
        graph_store: GraphStore,
    ): ...

    async def run(self, import_run_id: str) -> CurationReport: ...
```

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

# Core tools shipped with graphrag-core:
CORE_TOOLS = [
    "get_entity",             # (entity_type, entity_id) → Entity
    "search_entities",        # (entity_type, query, top_k) → list[Entity]
    "get_audit_trail",        # (node_id) → AuditTrail
    "get_related",            # (node_id, rel_type, depth) → Subgraph
    "get_entity_history",     # (entity_id, from_q, to_q) → Evolution
    "compare_quarters",       # (topic_id, q1, q2) → Diff
    "find_unaddressed_topics",# (quarter) → list[Topic]
    "find_trend",             # (topic_id) → TrendVector
]
```

Domain-specific tools (e.g. `find_divergent_topics`, `generate_report_section`) are registered by the Layer 2 application, not by graphrag-core.

---

## Building Block 8: Multi-Agent Orchestration & Report Generation

Coordinates agents in a workflow and renders output documents.

### Interface

```python
from graphrag_core.agents import Orchestrator, Agent, AgentContext
from graphrag_core.report import ReportRenderer

class Agent(Protocol):
    """A single agent with a defined role."""
    name: str
    async def execute(self, context: AgentContext) -> AgentResult: ...

class Orchestrator(Protocol):
    """Coordinates multi-agent workflows."""
    async def run_workflow(
        self, workflow_id: str, agents: list[Agent], context: AgentContext
    ) -> WorkflowResult: ...

class ReportRenderer(Protocol):
    """Renders structured report data into output format."""
    async def render(
        self, report_data: ReportData, template: str, config: RenderConfig
    ) -> bytes: ...  # returns file bytes (docx, pdf, html)

class AgentContext(BaseModel):
    """Shared context passed between agents."""
    graph_store: GraphStore
    tool_library: ToolLibrary
    search_engine: SearchEngine
    quarter: str
    workflow_state: dict[str, Any] = {}
```

---

## Extension Points

graphrag-core is designed to be extended, not forked.

### Adding a Domain (Layer 2)

```python
from graphrag_core.extraction import OntologySchema
from graphrag_core.tools import ToolLibrary

# 1. Define your domain ontology
construction_schema = OntologySchema(
    node_types=[
        NodeTypeDefinition(label="MonitoringTopic", properties=[...]),
        NodeTypeDefinition(label="Perspective", properties=[...]),
    ],
    relationship_types=[...]
)

# 2. Register domain-specific tools
tool_library.register(Tool(
    name="find_divergent_topics",
    description="Find topics with conflicting stakeholder perspectives",
    parameters={...},
    handler=find_divergent_topics_handler
))

# 3. Define domain-specific agents
class PerspectiveAgent(Agent):
    name = "perspective_agent"
    async def execute(self, context: AgentContext) -> AgentResult: ...
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

graphrag-core ships with production-ready defaults:

| Interface | Default Implementation | Package |
|---|---|---|
| `GraphStore` | `Neo4jGraphStore` | `graphrag-core` |
| `ExtractionEngine` | `LLMExtractionEngine` | `graphrag-core` |
| `SearchEngine` | `Neo4jHybridSearch` | `graphrag-core` |
| `EmbeddingModel` | `NomicEmbedding` | `graphrag-core` |
| `DetectionLayer` | `GDSDetectionLayer` (WCC, Node Similarity) | `graphrag-core` |
| `Orchestrator` | `LangGraphOrchestrator` | `graphrag-core[langgraph]` |
| `ReportRenderer` | `DocxRenderer` | `graphrag-core[docx]` |
| `ApprovalGateway` | `CLIApprovalGateway` | `graphrag-core` |

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
│       ├── curation/
│       │   ├── __init__.py
│       │   ├── pipeline.py        # CurationPipeline
│       │   ├── detection.py       # GDSDetectionLayer
│       │   └── approval.py        # ApprovalGateway Protocol
│       ├── registry/
│       │   ├── __init__.py
│       │   └── known_entities.py  # EntityRegistry
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── library.py         # ToolLibrary
│       │   └── core_tools.py      # 8 core tools
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── orchestrator.py    # Orchestrator Protocol
│       │   └── langgraph.py       # LangGraphOrchestrator
│       └── report/
│           ├── __init__.py
│           └── renderer.py        # ReportRenderer Protocol
├── tests/
│   ├── test_ingestion.py
│   ├── test_extraction.py
│   ├── test_graph.py
│   ├── test_curation.py
│   ├── test_tools.py
│   └── conftest.py
├── pyproject.toml
├── LICENSE                        # MIT
├── README.md
└── CLAUDE.md
```

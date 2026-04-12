"""Protocol definitions for graphrag-core (BB1-BB4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from graphrag_core.models import (
    AgentResult,
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
    RenderConfig,
    ReportData,
    SchemaViolation,
    SearchResult,
    WorkflowResult,
)


# ---------------------------------------------------------------------------
# BB1: Document Ingestion
# ---------------------------------------------------------------------------

@runtime_checkable
class DocumentParser(Protocol):
    """Converts raw file bytes into structured text sections."""

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits parsed documents into semantic chunks."""

    def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> list[DocumentChunk]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Produces vector embeddings from text."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class IngestionPipeline(Protocol):
    """Orchestrates parsing, chunking, and optional embedding."""

    async def ingest(
        self,
        source: bytes,
        content_type: str,
        config: ChunkConfig | None = None,
    ) -> list[DocumentChunk]: ...


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


# ---------------------------------------------------------------------------
# BB2: Schema-Guided Entity Extraction
# ---------------------------------------------------------------------------

@runtime_checkable
class ExtractionEngine(Protocol):
    """Extracts entities and relations from text guided by schema."""

    async def extract(
        self,
        chunks: list[DocumentChunk],
        schema: OntologySchema,
        import_run: ImportRun,
    ) -> ExtractionResult: ...


# ---------------------------------------------------------------------------
# BB3: Provenance-Native Knowledge Graph
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphStore(Protocol):
    """Abstract graph storage with provenance as core architecture."""

    async def merge_node(self, node: GraphNode, import_run_id: str) -> str: ...

    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str: ...

    async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None: ...

    async def get_node(self, node_id: str) -> GraphNode | None: ...

    async def get_audit_trail(self, node_id: str) -> AuditTrail: ...

    async def get_related(
        self, node_id: str, rel_type: str | None = None, depth: int = 1
    ) -> list[GraphNode]: ...

    async def apply_schema(self, schema: OntologySchema) -> None: ...

    async def validate_schema(self) -> list[SchemaViolation]: ...

    async def list_nodes(self) -> list[GraphNode]: ...

    async def count_relationships(self) -> int: ...


# ---------------------------------------------------------------------------
# BB4: Hybrid Search
# ---------------------------------------------------------------------------

@runtime_checkable
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


# ---------------------------------------------------------------------------
# BB8: Multi-Agent Orchestration
# ---------------------------------------------------------------------------

@runtime_checkable
class Agent(Protocol):
    """A single agent with a defined role."""

    name: str

    async def execute(self, context: object) -> AgentResult: ...


@runtime_checkable
class Orchestrator(Protocol):
    """Coordinates multi-agent workflows."""

    async def run_workflow(
        self, workflow_id: str, agents: list[Agent], context: object
    ) -> WorkflowResult: ...


@runtime_checkable
class ReportRenderer(Protocol):
    """Renders structured report data into output format."""

    async def render(
        self, report_data: ReportData, template: str, config: RenderConfig
    ) -> bytes: ...

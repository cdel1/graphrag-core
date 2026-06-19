"""Protocol definitions for graphrag-core (BB1-BB4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from graphrag_core.models import (
    AgentResult,
    AuditTrail,
    ChunkConfig,
    Community,
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

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> BaseModel: ...


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


@runtime_checkable
class ExtractionPromptBuilder(Protocol):
    """Builds the system prompt for LLM-based entity extraction."""

    def build_system_prompt(self, schema: OntologySchema) -> str: ...


@runtime_checkable
class ExtractionPostProcessor(Protocol):
    """Transforms an ExtractionResult before it is written to the graph store."""

    async def process(
        self,
        result: ExtractionResult,
        existing_entities: list[GraphNode] | None = None,
    ) -> ExtractionResult: ...


# ---------------------------------------------------------------------------
# BB3: Provenance-Native Knowledge Graph
# ---------------------------------------------------------------------------

@runtime_checkable
class GraphStore(Protocol):
    """Abstract graph storage with provenance as core architecture."""

    async def merge_node(self, node: GraphNode, import_run_id: str) -> str: ...

    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str:
        """Strict upsert (ADR-0034): raises MissingEndpointError if the source
        or target node does not exist; re-merging the same
        (source_id, target_id, type) updates properties in place — edges
        never duplicate."""
        ...

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

    async def list_relationships(self) -> list[GraphRelationship]: ...

    async def flush(self) -> None:
        """Make all previously applied mutations durable.

        Contract (ADR-0033): mutations are visible to reads on the same
        instance immediately; durability is guaranteed only after flush()
        returns. No-op where every mutation is already durable, or where
        durability is out of scope (ephemeral stores). Raises
        GraphStoreError if durability cannot be guaranteed; in-instance
        state remains visible and retrying flush() is legal.
        """
        ...

    async def clear(self) -> None:
        """Remove all nodes, relationships, provenance, and applied schema.

        Contract (ADR-0034): after clear() returns, every public read
        method reflects the empty state.
        """
        ...


@runtime_checkable
class CommunityDetector(Protocol):
    """Detects communities in a knowledge graph."""

    async def detect(self, graph_store: GraphStore) -> list[Community]: ...


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

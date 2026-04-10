"""Protocol definitions for graphrag-core (BB1-BB4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from graphrag_core.models import (
    AuditTrail,
    ChunkConfig,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    OntologySchema,
    ParsedDocument,
    SchemaViolation,
    SearchResult,
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

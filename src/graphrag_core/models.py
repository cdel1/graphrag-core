"""Pydantic data models for graphrag-core (BB1-BB4)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# BB1: Document Ingestion
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    title: str
    source: str
    doc_type: str
    date: date | None
    quarter: str | None
    sha256: str


class TextSection(BaseModel):
    heading: str | None
    text: str
    page: int | None = None


class ParsedDocument(BaseModel):
    sections: list[TextSection]
    metadata: DocumentMetadata


class ChunkConfig(BaseModel):
    max_tokens: int = 512
    overlap: int = 50


class DocumentChunk(BaseModel):
    id: str
    text: str
    embedding: list[float] | None = None
    page: int | None = None
    position: int | None = None
    chunk_type: str = "text"


class ImportRun(BaseModel):
    id: str
    timestamp: datetime
    source_type: str
    documents_processed: int
    entities_extracted: int


# ---------------------------------------------------------------------------
# BB2: Schema-Guided Entity Extraction
# ---------------------------------------------------------------------------

class PropertyDefinition(BaseModel):
    name: str
    type: str
    required: bool = False


class NodeTypeDefinition(BaseModel):
    label: str
    properties: list[PropertyDefinition]
    required_properties: list[str] = []


class RelationshipTypeDefinition(BaseModel):
    type: str
    source_types: list[str]
    target_types: list[str]


class OntologySchema(BaseModel):
    node_types: list[NodeTypeDefinition]
    relationship_types: list[RelationshipTypeDefinition]


class ExtractedNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any]


class ExtractedRelationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = {}


class ProvenanceLink(BaseModel):
    chunk_id: str
    node_id: str
    confidence: float


class ExtractionResult(BaseModel):
    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]
    provenance: list[ProvenanceLink]


# ---------------------------------------------------------------------------
# BB3: Provenance-Native Knowledge Graph
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any]


class GraphRelationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = {}


class ProvenanceStep(BaseModel):
    level: str
    id: str
    metadata: dict[str, Any]


class AuditTrail(BaseModel):
    node_id: str
    provenance_chain: list[ProvenanceStep]


class SchemaViolation(BaseModel):
    node_id: str
    violation_type: str
    message: str


# ---------------------------------------------------------------------------
# BB4: Hybrid Search
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    node_id: str
    label: str
    score: float
    source: str
    properties: dict[str, Any] = {}

"""Pydantic data models for graphrag-core (BB1-BB4)."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# BB1: Document Ingestion
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    title: str
    source: str
    doc_type: str
    date: date | None
    period: str | None = None   # canonical doc-time field, lexically-sortable
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


class Chunk(BaseModel):
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
    description: str | None = None


class RelationshipTypeDefinition(BaseModel):
    type: str
    source_types: list[str]
    target_types: list[str]
    description: str | None = None


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


class RejectionReason(str, Enum):
    """Why the schema did not admit an emission.

    A consumer deciding severity needs the cause, not a bare list: an
    undeclared label is a different signal from an edge whose endpoints are
    the right types in the wrong order.
    """

    UNDECLARED_NODE_LABEL = "undeclared_node_label"
    UNDECLARED_RELATIONSHIP_TYPE = "undeclared_relationship_type"
    DANGLING_ENDPOINT = "dangling_endpoint"
    ENDPOINT_TYPE_VIOLATION = "endpoint_type_violation"


class RejectedNode(BaseModel):
    """A node the extractor emitted and the schema did not admit.

    The node is preserved verbatim — properties included. ``chunk_id`` is the
    chunk the emission came from, i.e. the provenance link the node would have
    received had it been admitted; ``None`` when the caller validated outside
    a chunk context.
    """

    node: ExtractedNode
    reason: RejectionReason
    chunk_id: str | None = None


class RejectedRelationship(BaseModel):
    """A relationship the extractor emitted and the schema did not admit.

    Preserved verbatim, endpoints and properties included — a relationship's
    properties are where an extractor puts the passage that justified it, so
    discarding the relationship would destroy that passage.
    """

    relationship: ExtractedRelationship
    reason: RejectionReason
    chunk_id: str | None = None


class SchemaAdmission(BaseModel):
    """What a schema admitted, and what it rejected, from one set of emissions.

    Admission gates what enters the typed graph; it never destroys what the
    extractor returned. ``nodes`` ∪ ``rejected_nodes`` is exactly the emitted
    node set, and likewise for relationships.
    """

    nodes: list[ExtractedNode] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    rejected_nodes: list[RejectedNode] = Field(default_factory=list)
    rejected_relationships: list[RejectedRelationship] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]
    provenance: list[ProvenanceLink]
    rejected_nodes: list[RejectedNode] = Field(default_factory=list)
    """Emitted nodes the schema did not admit, preserved verbatim and typed by reason."""
    rejected_relationships: list[RejectedRelationship] = Field(default_factory=list)
    """Emitted relationships the schema did not admit, preserved verbatim and typed by reason."""
    quality_signals: dict[str, int | float] | None = None
    """Optional per-strategy diagnostic counters / values.

    Populated by extraction strategies that emit internal diagnostics
    (e.g., two-pass strategies tracking dropped invalid edges). Consumers
    (QualityReport aggregators, benchmark CLI) read this field as
    strategy-opaque key/value pairs. None when no strategy populates it.
    """


class ChunkExtractionResult(BaseModel):
    """LLM extraction output for a single chunk (no provenance — engine adds that).

    Both fields default to an empty list so that LLM tool-use responses that
    omit the ``nodes`` or ``relationships`` key (a common LLM behaviour when a
    chunk yields no entities of one type) still validate successfully.
    """

    nodes: list[ExtractedNode] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


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


class ProvenanceTrail(BaseModel):
    node_id: str
    provenance_chain: list[ProvenanceStep]


class SchemaViolation(BaseModel):
    node_id: str
    violation_type: str
    message: str


class Community(BaseModel):
    """A group of related nodes discovered by community detection."""

    id: str
    node_ids: list[str]
    size: int
    modularity_score: float | None = None
    metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# BB4: Hybrid Search
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    node_id: str
    label: str
    score: float
    source: str
    properties: dict[str, Any] = {}


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


# ---------------------------------------------------------------------------
# BB7: Core Tool Library
# ---------------------------------------------------------------------------

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None


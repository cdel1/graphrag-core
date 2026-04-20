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


class ExtractionResult(BaseModel):
    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]
    provenance: list[ProvenanceLink]


class ChunkExtractionResult(BaseModel):
    """LLM extraction output for a single chunk (no provenance — engine adds that)."""
    nodes: list[ExtractedNode]
    relationships: list[ExtractedRelationship]


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


# ---------------------------------------------------------------------------
# BB8: Multi-Agent Orchestration
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    agent_name: str
    success: bool
    data: dict[str, Any] = {}
    error: str | None = None


class WorkflowResult(BaseModel):
    workflow_id: str
    success: bool
    agent_results: list[AgentResult]


class ReportData(BaseModel):
    title: str
    sections: list[dict[str, Any]]


class RenderConfig(BaseModel):
    format: str = "markdown"
    options: dict[str, Any] = {}

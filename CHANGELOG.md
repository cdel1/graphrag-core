# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.6.0] — 2026-05-18

### Added

- **BB3 — `get_audit_trail` reaches document level.** `GraphStore.get_audit_trail(node_id)` now returns `ProvenanceStep`s with `level ∈ {"node", "chunk", "document"}`. The document-level step carries `DocumentMetadata` fields in its `metadata` (`title`, `source`, `doc_type`, `date`, `period`, `sha256`). Closes the long-standing gap where the Protocol promised "node → chunks → documents" but implementations stopped at chunks.
- **BB1 — `IngestionPipeline.ingest()` writes `:Document` nodes and `CHUNKED_FROM` edges.** New optional keyword-only kwargs `graph_store: GraphStore | None = None` and `import_run_id: str | None = None`. When provided, the pipeline writes one `:Document` node per ingested source (idempotent on SHA-256) and a `CHUNKED_FROM` edge per chunk, enabling document-level audit trails out of the box.
- **BB7 — three temporal tools.** `get_node_history`, `compare_periods`, `find_trend` in `graphrag_core/tools/core_tools_temporal.py`. Register all three via `register_temporal_tools(library, graph_store)`. The tools consume `get_audit_trail` for period resolution — zero hardcoded Lacuna labels or edge names. Optional `rel_type` kwarg lets callers filter neighbors at the call site.
- **`DocumentMetadata.period: str | None`** — canonical free-form lexically-sortable document-time field. Accepts ISO date (`"2026-05-18"`), period-string (`"2026-Q2"`, `"2026-05"`), or any monotone-sortable value the caller chooses.
- **`:Document(id)` uniqueness constraint** added to `Neo4jGraphStore.apply_schema`.

### Changed

- **`neo4j` driver is now an optional pip extra.** Install via `pip install graphrag-core[neo4j]` to use `Neo4jGraphStore`. Default install only ships `InMemoryGraphStore`.
- **Temporal tools renamed.** Lacuna's `get_entity_history` becomes `get_node_history` in graphrag-core (the framework operates on `GraphNode`, not the Lacuna Tier-1 `Entity` label).
- **`TrendSignal.direction` values are neutral.** Was `improving`/`deteriorating`/`stable`/`insufficient_data` (Lacuna's monitoring interpretation). Now `increasing`/`decreasing`/`stable`/`insufficient_data` — domain interpretation belongs in consumers.
- **`DocumentMetadata.quarter` is deprecated** via Pydantic `Field(deprecated=...)`. Slated for removal at v0.7.0. Use `period` instead. Transition: `IngestionPipeline` copies `quarter` into `period` when `period is None and quarter is set`, then strips `quarter` from the persisted Document properties.
- **`AuditTrail` chain ordering is implementation-defined.** Document and chunk steps may be emitted in any order relative to each other; consumers must filter by `step.level`, not by position. The `level="node"` step is always first.

### Soft-breaking semantic changes

- `pip install graphrag-core==0.6.0` no longer pulls the `neo4j` driver. Update install command to `pip install graphrag-core[neo4j]` if you use `Neo4jGraphStore`.
- `GraphStore` implementations conforming to v0.5.0 (no document-level provenance) remain Protocol-valid, but BB7 temporal tools will return empty results when used against them. Update your backend to walk `chunk → document` to take advantage of v0.6.0 tools.

### Did NOT push down

- `find_unaddressed_topics` — references Tier-3 `Topic` (human-curated) and Lacuna's `HAS_RECOMMENDATION` edge; not domain-agnostic. Stays in Lacuna's `intelligence/curation.py`.

### Internal

- `InMemoryGraphStore` indexes chunk→document edges (`self._chunk_to_doc`) on `merge_relationship` to support the extended audit trail.
- `Neo4jGraphStore.get_audit_trail` Cypher uses `OPTIONAL MATCH (c)-[:CHUNKED_FROM]->(d:Document)` with a `CASE WHEN d IS NOT NULL` guard in the `collect` to exclude null entries.
- Lazy import of the `neo4j` driver inside `Neo4jGraphStore.__init__` raises a clean `ImportError` with install hint when the extra is missing.
- All BB7 temporal-tool handlers wrap exceptions into `ToolResult(success=False, error=...)` per the BB7 contract.

### Design references

- Spec: `tessera/docs/superpowers/specs/2026-05-17-graphrag-core-v0.6.0-temporal-tools-design.md`
- ADR: `tessera/docs/adr/0001-audit-trail-reaches-document-level.md`
- Plan: `tessera/docs/superpowers/plans/2026-05-17-graphrag-core-v0.6.0-temporal-tools.md`

## [0.2.0] - 2026-04-12

### Added

- **BB1: Document Ingestion** — PDF, DOCX, Text, Markdown parsers; TokenChunker; IngestionPipeline
- **BB2: Schema-Guided Extraction** — LLMClient Protocol, AnthropicLLMClient, LLMExtractionEngine with strict schema validation
- **BB3: Provenance-Native Graph** — InMemoryGraphStore, Neo4jGraphStore with full provenance tracking
- **BB4: Hybrid Search** — InMemorySearchEngine, Neo4jHybridSearch with Reciprocal Rank Fusion
- **BB5: Governed Curation** — DeterministicDetectionLayer (duplicates, orphans, schema violations), CurationPipeline
- **BB6: Entity Registry** — InMemoryEntityRegistry with exact/fuzzy matching (token normalization + SequenceMatcher)
- **BB7: Tool Library** — ToolLibrary with 4 core tools (get_entity, search_entities, get_audit_trail, get_related)
- **BB8: Multi-Agent Orchestration** — Agent/Orchestrator/ReportRenderer Protocols, SequentialOrchestrator, AgentContext
- Cypher injection protection via identifier validation
- Integration test framework with `--run-integration` flag
- Optional dependencies: `graphrag-core[neo4j]`, `graphrag-core[anthropic]`, `graphrag-core[all]`

### Protocols (defined, no default implementation yet)

- `LLMCurationLayer`, `ApprovalGateway` (BB5 layers 2-3)
- `ReportRenderer` (BB8)
- `EmbeddingModel` (cross-cutting)

## [0.1.0] - 2026-04-10

### Added

- Initial commit establishing prior art
- BB1-BB4 Protocol interfaces (`DocumentParser`, `Chunker`, `ExtractionEngine`, `GraphStore`, `SearchEngine`)
- Pydantic data models for BB1-BB4
- Project scaffolding with hatchling build system

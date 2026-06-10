# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] — 2026-06-10

### Added

- **New `graphrag_core.eval` package — pluggable eval harness.** Offline / CI evaluation engine for any consumer building on graphrag-core. Tier 1 deterministic invariants + Tier 2 reference scoring against pluggable manifest/scorer pairs; slice-gated regression with in-repo versioned baselines. Mechanism is L1; manifest/scorer pairs plug in via the `graphrag_core.eval_pairs` entry-point group.
  - Models (`models.py`): `SliceScore`, `SliceGateRule`, `Violation`, `BaselineFile`, `GateFailure`, `RunReport`.
  - Protocols (`protocols.py`): `Manifest`, `ManifestLoader`, `PipelineRunner`, `Scorer`, `TierOneCheck`, `BaselineStore`, `SliceGate` (all `@runtime_checkable`).
  - Default impls: `JSONFileBaselineStore` (sharded per manifest), `DefaultSliceGate`.
  - Tier-1 invariant checks (`tier_one.py`): `ProvenanceCompletenessCheck`, `NoOrphanIntelligenceCheck`, `SchemaConformanceCheck`.
  - Orchestrator (`harness.py`): `EvalHarness` with fail-fast on Tier 1 violations.
  - Typer CLI (`cli.py`): `graphrag-core eval {run|rebaseline|list}` with cross-package pair discovery via entry points.
- **New console script `graphrag-core`** (Typer-based) — exposes the eval harness CLI as a top-level entry point.

### Changed

- Adds `typer>=0.12` as a top-level dependency (with transitive: `rich`, `markdown-it-py`, `mdurl`, `annotated-types`, `shellingham`).

### Rationale + spec

See ADR-0030 and the eval-harness design spec in the tessera workspace (`docs/adr/0030-eval-harness-l1-placement.md`, `docs/superpowers/specs/2026-06-10-eval-harness-design.md`). DocRED manifest/scorer pair lands in a follow-up release.

## [0.6.1] — 2026-05-18

> **Release-note retrospective:** This release is a clean republish of the same code first uploaded to PyPI as `0.6.0` on 2026-05-18. That earlier upload happened with an unbumped `pyproject.toml` (`version = "0.6.0"`) even though the artifact contained the v0.6.1 fix code and `__version__ = "0.6.1"` in-package. PyPI's `0.6.0` has been yanked; install `>=0.6.1`. No pre-yank `0.6.0` was ever published with the broken pre-fix code — the broken state existed only in the v0.6.0 PR before this hotfix landed.

### Fixed

- **BB1 — `IngestionPipeline.ingest` now merges `:Chunk` nodes before merging `CHUNKED_FROM` edges.** v0.6.0 wrote the `CHUNKED_FROM` edge from each chunk to the new `:Document` node but never created the chunk node first. `InMemoryGraphStore.merge_relationship` is permissive (just appends to a list) so v0.6.0 tests passed on Memory, but `Neo4jGraphStore.merge_relationship` does `MATCH (a {id: $source_id}), (b {id: $target_id})` — when the chunk node didn't exist, the MATCH returned no record and the caller hit `TypeError: 'NoneType' object is not subscriptable`. End-to-end ingest against a live Neo4j was broken in v0.6.0.
- Chunk nodes now carry `text`, `page`, `position`, and `chunk_type` properties when present on the source `DocumentChunk`.

### Tests added

- `test_ingest_creates_chunk_nodes_before_chunked_from_edges` — Memory-side regression catching the chunk-node count vs. chunk-list mismatch.
- `test_neo4j_ingest_creates_chunk_nodes_and_chunked_from` — Neo4j-side integration regression. Reproduces the v0.6.0 failure mode (would raise `TypeError` without the fix) and verifies the `MATCH (c:Chunk)-[:CHUNKED_FROM]->(d:Document)` Cypher returns the expected counts.

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

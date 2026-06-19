# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.13.0] — 2026-06-19

### BREAKING

BB taxonomy realignment per [ADR-0039](https://github.com/cdel1/tessera/blob/main/docs/adr/0039-bb-taxonomy-realignment-pipeline-vs-infrastructure.md). BB5 merges into BB3, BB8 deleted, BB9 LLM and BB10 Retrieval seats created.

**Removed public symbols (all zero L2 consumers):**

- `graphrag_core.Agent` (Protocol)
- `graphrag_core.Orchestrator` (Protocol)
- `graphrag_core.ReportRenderer` (Protocol)
- `graphrag_core.AgentContext`
- `graphrag_core.SequentialOrchestrator`
- `graphrag_core.AgentResult` (model)
- `graphrag_core.WorkflowResult` (model)
- `graphrag_core.ReportData` (model)
- `graphrag_core.RenderConfig` (model)
- `graphrag_core.curation` namespace (already empty post-0.12.0)
- `graphrag_core.agents` namespace

**Moved public surfaces (import path changed):**

- `EmbeddingModel` — primary import path remains `from graphrag_core import EmbeddingModel`. New BB10 path: `from graphrag_core.retrieval import EmbeddingModel`. The old "BB1 supporting" framing in `ingestion/INTERFACE.md` is retired; BB10 is the canonical home.

**Doctrine relocation (no symbol change):**

- Layer-3 attestation contract doctrine moved from `graphrag_core.curation.INTERFACE.md` to `graphrag_core.graph.INTERFACE.md`. Behaviour unchanged; consumers implementing Layer-3 surfaces continue to satisfy the same contract.

**Per ADR-0006b Rule 1, public-symbol removal is a BREAKING change → minor bump at pre-1.0 (0.12.0 → 0.13.0).**

### Added

- `graphrag_core.retrieval` namespace (BB10 seat) — re-exports `EmbeddingModel`; documented future `Reranker` Protocol home.
- `tests/test_retrieval/` — Protocol-conformance smoke tests.

### Changed

- `CLAUDE.md` § Architecture rewritten with two-axis (pipeline / cross-cutting) framing.
- `README.md` refreshed (tagline, architecture diagram, BB table, Extension Pattern).
- `llm/INTERFACE.md` header BB-attribution: "BB1 supporting" → "BB9".
- `ingestion/INTERFACE.md` drops EmbeddingModel section; adds BB10 pointer.

### Migration

Consumers of `graphrag-core 0.12.0` should:

1. Remove any `from graphrag_core import Agent, Orchestrator, SequentialOrchestrator, AgentContext, ReportRenderer, AgentResult, WorkflowResult, ReportData, RenderConfig` imports. None of these had production consumers; if you implemented `Agent` subclasses or `ReportRenderer`, replace with the MCP-tool extension pattern per the agentic-substrate doctrine (`2026-05-15-agentic-substrate-design.md` §4.1).
2. Optionally update `from graphrag_core.interfaces import EmbeddingModel` to `from graphrag_core.retrieval import EmbeddingModel` to match the new BB10 location (top-level `from graphrag_core import EmbeddingModel` continues to work).
3. Bump pinned dependency: `graphrag-core>=0.13.0`.

## [0.12.0] — 2026-06-18

### BREAKING

Per ADR-0038, the BB5 L1 surface shrinks to the Layer-3 attestation *contract*
only. The Python that the 2026-06-15 BB5 doctrine ADR retained is removed.

Removed public symbols (importable from `graphrag_core` or `graphrag_core.curation`):

- Protocols (`graphrag_core.interfaces`):
  - `DetectionLayer`
  - `LLMCurationLayer`
  - `ApprovalGateway`
- Models (`graphrag_core.models`):
  - `CurationIssue`
  - `CurationReport`
  - `ApprovalBatch`
  - `ApplyResult`
- Implementations (`graphrag_core.curation`):
  - `DeterministicDetectionLayer`
  - `CurationPipeline`

`curation/INTERFACE.md` is rewritten to contract-only. The package now exports
nothing; consumers should not import from `graphrag_core.curation`. The
Layer-3 attestation contract (text doctrine) is the authoritative artifact and
binds every L2 Layer-3 surface.

Migration: there were no documented external consumers of any of the removed
symbols. If your application imports them, either drop the dependency or
re-derive the deterministic detectors at your application layer — they are
straightforward to reimplement (orphan = nodes with no relationships; duplicate =
node-name fuzzy match within label; schema violation = property / edge-type
check against `OntologySchema`).

### Changed

- `curation/INTERFACE.md` rewritten to document only the Layer-3 attestation contract.
- `CLAUDE.md` BB5 row updated to reflect the contract-only state.

## [0.11.0] — 2026-06-15

### Added

- `graphrag_core.llm.from_env()` — env-var-selected factory returning a
  configured `LLMClient`. Reads `GRAPHRAG_LLM_PROVIDER` (`"anthropic"`
  (default) or `"openai"`, case-insensitive). Unknown values raise
  `ValueError`. API-key validation is delegated to the provider SDK.
- Two new eval pairs: `docred_anthropic` and `feverous_anthropic`,
  mirroring the existing OpenAI-pinned `docred` and `feverous` pairs
  but pinned to `claude-sonnet-4-6`. Use side-by-side for
  provider-portability sanity checks.

### Changed

- `AnthropicLLMClient` default model: `claude-sonnet-4-20250514` →
  `claude-sonnet-4-6`. Consumers relying on the previous default
  should pass `model="claude-sonnet-4-20250514"` explicitly.
- `llm/INTERFACE.md` rewritten to document the new factory + the
  framework's "tuned-default provider" concept (per-version, one
  provider receives active prompt + model tuning; other reference
  impls remain Protocol-compliant). The previous claim that an
  external env var (`LLM_PROVIDER`) drove provider selection was
  removed — the actual mechanism is now `GRAPHRAG_LLM_PROVIDER`
  consumed by `from_env()`.
- **DocRED and FEVEROUS pipeline runners now invoke the configured
  `LLMClient`** instead of being Band-3 stubs. Each runner takes an
  `llm` constructor argument; the pair registrations build the
  appropriate provider client from the manifest's `model_pin` and
  inject it. The L1 example benchmarks now produce a real provider-
  portability signal: the OpenAI-pinned and Anthropic-pinned pairs
  hit different model providers and the per-slice deltas reflect
  actual extraction differences.
  - FEVEROUS runner asks the model to classify each claim as
    `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` from world knowledge
    and emits a `:Claim` node with the verdict.
  - DocRED runner asks the model to extract typed relations (Wikidata
    PIDs) between the entities named in the gold vertex set and emits
    `:Entity` endpoints + relationships. Absolute precision/recall is
    expected to be low without retrieval; the benchmark's purpose at
    L1 is portability comparison, not state-of-the-art extraction.
- The `0.1.0.json` baseline files under `eval/baselines/docred@2026-06-10/`
  and `eval/baselines/feverous@2026-06-10/` were regenerated against
  the LLM-driven runners. Stub-era 0.0 baselines have been replaced
  with real OpenAI numbers.

### Notes

- **FEVEROUS `NOT ENOUGH INFO` (NEI) slice is excluded from comparative
  gating in the closed-book pipeline shipped at L1.** The FEVEROUS NEI
  label is a property of *evidence availability in Wikipedia* — it
  means annotators couldn't verify or refute the claim from the bound
  evidence pool. The L1 runner here is *closed-book* (no retrieval),
  so the model can only express confidence in the claim's truth, not
  in Wikipedia coverage. The two are different signals; treating
  per-slice ΔR on NEI as a portability gate measures model hedging
  tendency, not extraction capability. Comparative gating in this
  release uses the `SUPPORTS` and `REFUTES` labels (plus the
  per-challenge slices) where closed-book inference produces a
  meaningful signal. Future runners that consume FEVEROUS evidence
  before classifying can restore the NEI slice to the gate.

### Internal

- CI boundary check regex tightened: removed proper-noun guards that
  were word-boundary-fragile (matched inside unrelated identifiers); the
  forbidden list is now structural domain terms only.

## [0.10.0] — 2026-06-12

### Added

- `graphrag_core.testing.contracts.graph_store.GraphStoreContractTests` —
  subclass-and-run conformance suite for any GraphStore implementation
  (ADR-0034, realising ADR-0006b Rule 9 for BB3). Six mandatory tests plus
  three capability-gated ones (`persists_across_instances`,
  `persists_schema_across_instances`, `requires_concurrency_safety`).
  pytest is an import-time requirement of `graphrag_core.testing` only —
  not a runtime dependency.
- `GraphStore.clear()` — required Protocol method: remove all nodes,
  relationships, provenance, and applied schema.
- `graphrag_core.exceptions.MissingEndpointError` (subclass of
  `GraphStoreError`), with `source_id` / `target_id` attributes.

### BREAKING

- `merge_relationship` is now canonically a **strict upsert** (ADR-0034):
  a missing source or target node raises `MissingEndpointError` (was:
  silent append on InMemory — the v0.6.0 incident shape — and silent
  `None`/`TypeError` on Neo4j); re-merging the same
  `(source_id, target_id, type)` updates properties in place.
- `GraphStore` implementations must add `clear()`.
- Callers that relied on permissive merges must merge endpoint nodes first
  or catch `MissingEndpointError` at their extraction boundary.

## [0.9.0] — 2026-06-11

### Added

- `GraphStore.flush()` — required Protocol method marking the persistence
  boundary (ADR-0033). `graphrag_core.exceptions.GraphStoreError` seeded
  (ADR-0006b Rule 4); `flush()` raises it when durability cannot be
  guaranteed. `IngestionPipeline.ingest()` now flushes once at end-of-ingest.

### BREAKING

- `GraphStore` implementations must add `flush()` (a no-op if every mutation
  is already durable). The write contract is now visible-now, durable-at-flush:
  mutations are immediately visible on the same instance; durability is
  guaranteed only after `flush()` returns.

### Changed

- **BB4 (search): retro-grill correctness + design cleanup.** Closes the S2 deferred retro-grill candidate.
  - `InMemorySearchEngine.fulltext_search` now uses **BM25** (TF-normalized + IDF-weighted, k1=1.5, b=0.75) instead of substring matching. The substring implementation incorrectly ranked exact-match short documents over high-term-frequency longer ones; BM25 fixes IDF weighting and term-frequency scoring. **Soft-breaking** for callers that relied on substring semantics (e.g., single-character queries no longer match arbitrary substrings — they must tokenize to real words).
  - `InMemorySearchEngine.vector_search` now raises `ValueError` on query-embedding dimensionality mismatch (was: silent garbage cosines). Aligns with the contract written in `INTERFACE.md`.
  - `InMemorySearchEngine.graph_search` now delegates to `GraphStore.get_related` when constructed with a new `graph_store=...` kwarg. Previously the method always returned `[]`. **Backward-compatible**: existing `InMemorySearchEngine(nodes=...)` callers still work (graph_search returns `[]` if no store is wired).
  - `InMemorySearchEngine.hybrid_search` and `Neo4jHybridSearch.hybrid_search` now run vector + fulltext **concurrently** (`asyncio.gather`) and fetch `top_k * 2` candidates from each engine for fusion (was: sequential `await`, `top_k` from each).
  - Both `hybrid_search` impls expose `rrf_k: int = 60` as a keyword-only parameter — exposes the RRF k constant from Cormack et al. 2009 for callers wanting to tune the rank-weighting curve.
  - `Neo4jHybridSearch.fulltext_search` applies the `node_types` filter inside the Cypher `WHERE` clause **before** `LIMIT $top_k`. Previously the filter ran in Python after the database had already limited the result set, which could return fewer than `top_k` matches even when matching results existed.
  - **`hybrid_search` is now formally bi-modal (vector + fulltext only).** `graph_search` is intentionally **not** fused into hybrid: it needs a `start_node_id` hybrid doesn't have, and graph-distance scores don't compose meaningfully with similarity/BM25 scores in RRF. Callers wanting tri-modal results should call `graph_search` separately and fuse via `fusion.reciprocal_rank_fusion` explicitly. `INTERFACE.md`'s earlier "vector + fulltext + (optional) graph" wording was aspirational, not implemented; it has been corrected.
- **`SearchResult.score` contract clarified in `INTERFACE.md`.** Vector and fulltext scores have well-defined bounds (cosine `[-1,1]`, BM25 unbounded non-negative). **Hybrid (RRF) scores are unbounded and not normalized to `[0,1]`** — only ordering is meaningful. The earlier "normalized to `[0,1]`" wording was wrong for the RRF path.
- **`filters` dict supported keys documented per-impl in `INTERFACE.md`.** Both `Neo4jHybridSearch.vector_search` and `InMemorySearchEngine.vector_search` accept `{"label": str}`; other keys are silently ignored.

### Doctrine

- `tessera/docs/adr/0032-bb4-hybrid-search-rrf-and-protocol-shape.md` — RRF as canonical fusion, bi-modal hybrid, dual L1 impls.

## [0.8.0] — 2026-06-10

### Added

- **Two concrete manifest/scorer pairs on top of the v0.7.0 eval harness — `docred` and `feverous`.** Both ship with their own JSONL fixtures under `eval/fixtures/`, register via the `graphrag_core.eval_pairs` entry-point group, and are runnable via `graphrag-core eval run <pair>`:
  - **`docred`** — Document-level relation extraction benchmark (Wikipedia-domain). 50-doc subsample of the DocRED dev split (MIT license), seed 42. Slice axis: `relation_type`. Per-relation precision/recall via identity mapping `GraphRelationship.type → DocRED relation`.
  - **`feverous`** — Fact-verification benchmark with table+text evidence. 50-claim subsample of the FEVEROUS dev split (CC-BY-SA 3.0), seed 42, stratified across `label × challenge`. Slice axes: `label` (SUPPORTS/REFUTES/NEI) and `challenge` (6 buckets). Predictions read from produced `:Claim` nodes' `verdict` property.
- **CI workflows** — `eval-fast.yml` (Tier 1 invariants on every PR + push, no LLM cost) and `eval-full.yml` (full T1+T2 against both pairs, path-filtered, fork-safe via mutually-exclusive jobs).
- **`NOTICES.md`** at repo root — canonical index of bundled third-party fixtures and their licenses (DocRED MIT, FEVEROUS CC-BY-SA 3.0). Top-level `README.md` and `eval/fixtures/feverous/LICENSE` cross-reference it.

### Changed

- `pyproject.toml` declares two entry points under the `graphrag_core.eval_pairs` group: `docred` and `feverous`.

### Notes

- The L1 harness mechanism itself was shipped in `0.7.0`. This release is the first that exposes registered pairs to external consumers via PyPI; `graphrag-core eval list` will now show `docred` and `feverous` out of the box.
- The pipeline runners for both pairs are Band-3 stubs (emit Documents/Claims only). Scores will be near-zero until real extraction is wired in via downstream consumers.

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

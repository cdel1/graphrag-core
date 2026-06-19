# graphrag-core

> Domain-agnostic Graph RAG framework. MIT License. Open Source.

## What This Is
Layer 1 of a 3-layer architecture. This repo contains ONLY domain-agnostic platform code.
Domain-specific logic (construction monitoring, due diligence, compliance) lives in separate repos that import graphrag-core as a dependency.

## The One Rule That Cannot Be Broken
**No domain logic in this repo.** If you're importing a construction-specific concept, a customer-specific schema, or any business-domain term — stop and refactor. This code must work equally for construction monitoring, transaction due diligence, forensic investigations, or any other document-heavy knowledge work.

Test: Could a team building a legal compliance graph use this code without modification? If no → it doesn't belong here.

## Architecture

8 active Building Blocks, organised on two axes — **pipeline-stage seats** (the data-flow stages every consumer walks) and **cross-cutting infrastructure seats** (Protocol families consumed across multiple stages or by external systems). Each BB exposes a Protocol or contract family Lacuna imports against; **a BB seat earns its existence by real consumer pressure** — speculative or doctrine-only seats get retired (per [ADR-0039](https://github.com/cdel1/tessera/blob/main/docs/adr/0039-bb-taxonomy-realignment-pipeline-vs-infrastructure.md)).

| # | Block | Axis | Interface |
|---|---|---|---|
| 1 | Document Ingestion | pipeline | `DocumentParser`, `Chunker`, `IngestionPipeline` |
| 2 | Entity Extraction | pipeline | `ExtractionEngine`, `ExtractionPromptBuilder`, `ExtractionPostProcessor`, `OntologySchema` |
| 3 | Knowledge Graph + Layer-3 attestation contract | pipeline + doctrine | `GraphStore`, `CommunityDetector`; Layer-3 attestation contract (text doctrine in `graph/INTERFACE.md`) |
| 4 | Hybrid Search | pipeline | `SearchEngine` |
| ~~5~~ | ~~Governed Curation~~ | retired | Layer-3 doctrine merged into BB3; no Python surface |
| 6 | Entity Registry | pipeline | `EntityRegistry` |
| 7 | Core Tool Library (agent contract) | cross-cutting | `Tool`, `ToolLibrary` |
| ~~8~~ | ~~Orchestration~~ | retired | Deleted in v0.13.0; agent-callability flows through BB7 + external MCP |
| 9 | LLM Client | cross-cutting | `LLMClient` |
| 10 | Retrieval Models | cross-cutting | `EmbeddingModel` (+ `Reranker` planned) |

For per-block shipping state, default implementations, and Protocol-only entries, see each package's `INTERFACE.md`. Run `uv run pytest --collect-only -q` for the current test count.

## Tech Stack
- Python 3.12+
- Pydantic v2 for all data models
- Neo4j (default graph backend, swappable via GraphStore interface)
- pytest + pytest-asyncio for tests
- Type hints everywhere. No exceptions.

## Code Rules
- All interfaces are `Protocol` classes in `interfaces.py`
- All data models are `BaseModel` classes in `models.py`
- Async by default for all I/O
- Functions < 30 lines. Extract early.
- Docstrings: Google style, English only.
- No hardcoded technology references in interface definitions
- Default implementations live alongside interfaces but are clearly separated

## Project Structure
```
src/graphrag_core/
├── interfaces.py       # ALL Protocol definitions
├── models.py           # ALL Pydantic models
├── ingestion/          # BB1: Parse, chunk                    | + INTERFACE.md
├── extraction/         # BB2: Schema-guided entity extraction | + INTERFACE.md
├── graph/              # BB3: GraphStore + Neo4j (+ L3        | + INTERFACE.md
│                       #      attestation contract doctrine)
├── search/             # BB4: Hybrid search                   | + INTERFACE.md
├── registry/           # BB6: Known entity dedup              | + INTERFACE.md
├── tools/              # BB7: Core tool library (agent surf.) | + INTERFACE.md
├── llm/                # BB9: LLMClient (multi-provider)      | + INTERFACE.md
└── retrieval/          # BB10: EmbeddingModel + future        | + INTERFACE.md
                        #       Reranker
```

## Per-Package Interface Documentation

Each package directory contains an `INTERFACE.md` documenting:
- Which Protocols live in that package and their signatures
- Contracts (idempotency, atomicity, determinism)
- Error modes (what raises, what returns falsy, what propagates)
- Performance invariants (O(n) characteristics, hot-path warnings)
- Reference implementations and their trade-offs
- Implementation skeleton for adding a new backend
- Test checklist

These docs were added 2026-05-17 per audit Action 3 (interface clarity for agents implementing or debugging a Protocol — a Protocol on its own only documents method signatures, not contracts).

When implementing a new Protocol backend (e.g., a new `GraphStore`, a new `LLMClient`), start with the package's `INTERFACE.md`. When adding a new Protocol, update the package `INTERFACE.md` in the same commit.

## Extension Pattern
Domain layers extend graphrag-core by:
1. Defining an `OntologySchema` (node types, relationships)
2. Registering domain tools via `ToolLibrary.register()` — the agent-callable surface (BB7)
3. (Optional) Exposing the `ToolLibrary` via an external MCP server — graphrag-core itself does NOT ship an MCP server; the framework is tool-agnostic per the agentic-substrate doctrine (`2026-05-15-agentic-substrate-design.md` §4.1).
4. (Optional) Providing a domain-specific renderer in the consuming application — `ReportRenderer` Protocol removed in v0.13.0 per ADR-0039; renderers are L2 concerns.

```python
# Example: domain-tool registration
from graphrag_core import OntologySchema, ToolLibrary, Tool

schema = OntologySchema(node_types=[...], relationship_types=[...])
tool_library.register(my_domain_tool)
```

Per the agentic-substrate doctrine: agents are external (Claude Code, MCP clients, custom harnesses); they consume your `ToolLibrary` over MCP. graphrag-core's job is the tool contract.

## Commands
```bash
pytest tests/ -x -q                    # tests (fail fast)
pytest tests/ -x -q --cov             # with coverage
docker compose up neo4j                # start Neo4j for integration tests
python -m graphrag_core.graph.schema   # apply schema
```

## What Does NOT Belong Here
- Employer-specific anything (deployment configs, client references, internal tooling)
- Domain-specific terms (MonitoringTopic, SubjectArea, Perspective, CapturePoint, SollIstAbgleich, InvestorAlert)
- Hardcoded LLM model names (use config/env vars)
- Any reference to specific organizations or engagements

## Release Strategy
- Semantic versioning (MAJOR.MINOR.PATCH)
- Public GitHub repo
- Published to PyPI as `graphrag-core`
- CHANGELOG.md tracks all changes
- First public commit establishes prior art before any organizational use

For per-release planning and historical release blockers, see the CHANGELOG and the GitHub milestone/issue tracker.

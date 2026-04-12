# Release Readiness: graphrag-core v0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare graphrag-core for PyPI publication as v0.2.0 with updated metadata, documentation, CI/CD, and type marker.

**Architecture:** No code changes — this sprint is metadata, docs, and infrastructure only. pyproject.toml gets PyPI-ready metadata, README is overhauled to reflect all 8 building blocks, CHANGELOG is created, GitHub Actions workflows handle CI/CD, and a py.typed marker enables type checker support.

**Tech Stack:** GitHub Actions, uv, hatchling, pypa/gh-action-pypi-publish

**Design spec:** `docs/superpowers/specs/2026-04-12-release-readiness-design.md`

---

## File Map

### New files

| File | Responsibility |
|---|---|
| `CHANGELOG.md` | Version history |
| `.github/workflows/test.yml` | CI: unit tests + boundary check on push/PR |
| `.github/workflows/release.yml` | CD: build + publish to PyPI on tag |
| `src/graphrag_core/py.typed` | PEP 561 type marker |

### Modified files

| File | Change |
|---|---|
| `pyproject.toml` | Version bump, author, readme, classifiers, urls |
| `README.md` | Full overhaul reflecting BB1-BB8 |

---

## Task 1: Update pyproject.toml with PyPI metadata

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Replace the `[project]` section and add `[project.urls]`. Keep all other sections unchanged (`[build-system]`, `[tool.hatch.build.targets.wheel]`, `[tool.pytest.ini_options]`, `[project.optional-dependencies]`, `[dependency-groups]`):

```toml
[project]
name = "graphrag-core"
version = "0.2.0"
description = "Domain-agnostic Graph RAG framework for building governed, auditable Knowledge Graphs"
license = "MIT"
requires-python = ">=3.12"
readme = "README.md"
authors = [
    {name = "Dino Celi"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = ["pydantic>=2.0", "pypdf>=4.0", "python-docx>=1.0"]

[project.urls]
Homepage = "https://github.com/cdel1/graphrag-core"
Repository = "https://github.com/cdel1/graphrag-core"
Issues = "https://github.com/cdel1/graphrag-core/issues"
```

- [ ] **Step 2: Update version in __init__.py**

In `src/graphrag_core/__init__.py`, change:

```python
__version__ = "0.2.0"
```

- [ ] **Step 3: Verify build works**

Run: `uv build`
Expected: Builds `dist/graphrag_core-0.2.0-py3-none-any.whl` and `dist/graphrag_core-0.2.0.tar.gz` without errors.

Run: `rm -rf dist/` (clean up)

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: 163 passed, 19 skipped

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/graphrag_core/__init__.py
git commit -m "chore: bump version to 0.2.0, add PyPI metadata"
```

---

## Task 2: Create py.typed marker

**Files:**
- Create: `src/graphrag_core/py.typed`

- [ ] **Step 1: Create empty marker file**

Create `src/graphrag_core/py.typed` as an empty file (0 bytes). This tells type checkers (mypy, pyright) that the package ships inline type information per PEP 561.

- [ ] **Step 2: Verify it's included in the build**

Run: `uv build && unzip -l dist/graphrag_core-0.2.0-py3-none-any.whl | grep py.typed`
Expected: `graphrag_core/py.typed` appears in the wheel contents.

Run: `rm -rf dist/`

- [ ] **Step 3: Commit**

```bash
git add src/graphrag_core/py.typed
git commit -m "chore: add py.typed marker for PEP 561 type checker support"
```

---

## Task 3: Overhaul README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md contents**

Replace the entire file with:

```markdown
# graphrag-core

A domain-agnostic framework for building governed, auditable Knowledge Graphs from documents using LLM-powered extraction, provenance-native storage, and multi-agent orchestration.

## Architecture

```
YOUR DOMAIN LAYER (Layer 2)
  Ontology, domain tools, domain agents, templates
                    |
                    | imports
                    v
graphrag-core (Layer 1)

  Ingestion   Extraction   Graph Store   Search
  Curation    Registry     Tool Library  Orchestration
```

## Install

```bash
pip install graphrag-core                    # core (in-memory backends)
pip install graphrag-core[neo4j]             # + Neo4j graph store and search
pip install graphrag-core[anthropic]         # + Claude LLM client
pip install graphrag-core[all]               # everything
```

## Quick Start

```python
import asyncio
from graphrag_core import (
    TextParser, TokenChunker, IngestionPipeline,
    InMemoryGraphStore, InMemorySearchEngine,
    LLMExtractionEngine, OntologySchema, NodeTypeDefinition,
    PropertyDefinition, RelationshipTypeDefinition,
    ToolLibrary, register_core_tools,
)
from graphrag_core.models import ChunkConfig, DocumentChunk, GraphNode, ImportRun
from datetime import datetime

async def main():
    # 1. Ingest a document
    pipeline = IngestionPipeline(parser=TextParser(), chunker=TokenChunker())
    chunks = await pipeline.ingest(b"Alice works at Acme Corp.", "text/plain")

    # 2. Define your domain schema
    schema = OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Person",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
            NodeTypeDefinition(
                label="Company",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(type="WORKS_AT", source_types=["Person"], target_types=["Company"]),
        ],
    )

    # 3. Extract entities (requires an LLMClient implementation)
    # engine = LLMExtractionEngine(llm_client=your_client)
    # result = await engine.extract(chunks, schema, import_run)

    # 4. Store in graph
    store = InMemoryGraphStore()
    await store.merge_node(GraphNode(id="p1", label="Person", properties={"name": "Alice"}), "run-1")
    await store.merge_node(GraphNode(id="c1", label="Company", properties={"name": "Acme Corp"}), "run-1")

    # 5. Search
    search = InMemorySearchEngine(
        nodes=[await store.get_node("p1"), await store.get_node("c1")],
    )
    results = await search.fulltext_search("Acme", top_k=5)
    print(results)

    # 6. Wire up tools for agents
    library = ToolLibrary()
    register_core_tools(library, store, search)
    result = await library.execute("get_entity", entity_id="p1")
    print(result)

asyncio.run(main())
```

## Building Blocks

| # | Block | Interface | Implementation | Status |
|---|---|---|---|---|
| 1 | Document Ingestion | `DocumentParser`, `Chunker` | PDF, DOCX, Text, Markdown parsers; TokenChunker | Done |
| 2 | Entity Extraction | `ExtractionEngine`, `LLMClient` | LLMExtractionEngine, AnthropicLLMClient | Done |
| 3 | Knowledge Graph | `GraphStore` | InMemoryGraphStore, Neo4jGraphStore | Done |
| 4 | Hybrid Search | `SearchEngine` | InMemorySearchEngine, Neo4jHybridSearch (RRF) | Done |
| 5 | Governed Curation | `DetectionLayer` | DeterministicDetectionLayer, CurationPipeline | Done (detection layer) |
| 6 | Entity Registry | `EntityRegistry` | InMemoryEntityRegistry (fuzzy matching) | Done |
| 7 | Tool Library | `ToolLibrary` | 4 core tools (get_entity, search, audit_trail, related) | Done |
| 8 | Orchestration | `Agent`, `Orchestrator` | SequentialOrchestrator, AgentContext | Done |

Protocols marked with `(Protocol only)` have no default implementation yet:
- `LLMCurationLayer`, `ApprovalGateway` (BB5 layers 2-3)
- `ReportRenderer` (BB8)
- `EmbeddingModel` (cross-cutting)

## Extension Pattern

```python
from graphrag_core import OntologySchema, ToolLibrary, Tool

# 1. Define your domain ontology
schema = OntologySchema(node_types=[...], relationship_types=[...])

# 2. Register domain-specific tools
library = ToolLibrary()
library.register(Tool(name="my_tool", description="...", parameters={}, handler=my_handler))

# 3. Implement domain agents
class MyAgent:
    name = "analyst"
    async def execute(self, context):
        result = await context.tool_library.execute("my_tool")
        context.workflow_state["analysis"] = result.data
        return AgentResult(agent_name=self.name, success=True)
```

## Development

```bash
# Clone and install
git clone https://github.com/cdel1/graphrag-core.git
cd graphrag-core
uv sync --all-extras

# Run unit tests
uv run pytest tests/ -x -q

# Run integration tests (requires Neo4j)
docker run -d --name neo4j-test -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/development neo4j:5-community
uv run pytest tests/ -x --run-integration

# Build
uv build
```

## License

MIT
```

- [ ] **Step 2: Verify README renders**

Run: `uv run python -c "import pathlib; print(len(pathlib.Path('README.md').read_text()), 'chars')"`
Expected: Prints char count (sanity check file was written).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: overhaul README for v0.2.0 with all 8 building blocks"
```

---

## Task 4: Create CHANGELOG

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG for v0.1.0 and v0.2.0"
```

---

## Task 5: CI workflow — test.yml

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Create .github/workflows directory and test.yml**

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run unit tests
        run: uv run pytest tests/ -x -q

      - name: Boundary check — no domain leakage
        run: |
          FORBIDDEN="MonitoringTopic|Perspective|SubjectArea|Interview|Dalux|CapturePoint|InvestorAlert|SollIst|EY|Parthenon|Prague"
          if grep -rn -E "$FORBIDDEN" src/; then
            echo "DOMAIN LEAKAGE DETECTED in graphrag-core"
            exit 1
          else
            echo "graphrag-core is clean"
          fi
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add test workflow with unit tests and boundary check"
```

---

## Task 6: CD workflow — release.yml

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create release.yml**

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  id-token: write

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install 3.12

      - name: Build package
        run: uv build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

Note: This uses PyPI's trusted publisher (OIDC). Before the first release, you need to configure the trusted publisher on PyPI:
1. Go to https://pypi.org/manage/account/publishing/
2. Add a "pending publisher" for `graphrag-core` with GitHub repo `cdel1/graphrag-core`, workflow `release.yml`, environment `pypi`
3. Create a `pypi` environment in GitHub repo Settings > Environments

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add release workflow for PyPI publication on tag"
```

---

## Task 7: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: 163 passed, 19 skipped

- [ ] **Step 2: Verify build**

Run: `uv build`
Expected: Builds cleanly.

Run: `rm -rf dist/`

- [ ] **Step 3: Verify README imports work**

Run the quick start snippet to confirm imports:

```bash
uv run python -c "
from graphrag_core import (
    TextParser, TokenChunker, IngestionPipeline,
    InMemoryGraphStore, InMemorySearchEngine,
    LLMExtractionEngine, OntologySchema, NodeTypeDefinition,
    PropertyDefinition, RelationshipTypeDefinition,
    ToolLibrary, register_core_tools,
    Agent, Orchestrator, ReportRenderer,
    DetectionLayer, EntityRegistry,
    CurationPipeline, DeterministicDetectionLayer,
    InMemoryEntityRegistry,
    AgentContext, SequentialOrchestrator,
)
print(f'graphrag-core v{__import__(\"graphrag_core\").__version__} — all imports OK')
"
```

Expected: `graphrag-core v0.2.0 — all imports OK`

- [ ] **Step 4: Verify boundary check**

Run: `grep -rn -E "MonitoringTopic|Perspective|SubjectArea|Interview|Dalux|CapturePoint|InvestorAlert|SollIst|EY|Parthenon|Prague" src/ && echo "LEAK" || echo "CLEAN"`
Expected: `CLEAN`

---

## Summary

| Task | What it does |
|---|---|
| 1 | pyproject.toml: version 0.2.0, author, classifiers, urls, readme |
| 2 | py.typed marker for type checker support |
| 3 | README overhaul: all 8 BBs, install, quick start, dev instructions |
| 4 | CHANGELOG: v0.1.0 and v0.2.0 entries |
| 5 | CI workflow: unit tests + boundary check on push/PR |
| 6 | CD workflow: build + publish to PyPI on tag |
| 7 | Final verification: tests, build, imports, boundary check |

**After this sprint:** Review the README on GitHub, then tag `v0.2.0` and push the tag to trigger the release workflow. Configure PyPI trusted publisher first.

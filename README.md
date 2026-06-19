# graphrag-core

A domain-agnostic framework for building governed, auditable Knowledge Graphs from documents using LLM-powered extraction, provenance-native storage, and agent-callable tools.

## Architecture

```
YOUR DOMAIN LAYER (Layer 2)
  Ontology, domain tools, agents-via-MCP
                    |
                    | imports
                    v
graphrag-core (Layer 1)

  Pipeline:        Ingest -> Extract -> Graph Store (+ L3 attestation)
                                            -> Search     (+ Entity Registry)

  Cross-cutting:   LLM Client  ·  Retrieval Models  ·  Tool Library (agent surface)
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

| #  | Block                                       | Type                  | Interface                                              | Default impls                                    | Status   |
|----|---------------------------------------------|-----------------------|--------------------------------------------------------|--------------------------------------------------|----------|
| 1  | Document Ingestion                          | pipeline              | `DocumentParser`, `Chunker`, `IngestionPipeline`       | PDF/DOCX/Text/Markdown parsers; TokenChunker     | Done     |
| 2  | Entity Extraction                           | pipeline              | `ExtractionEngine`, `ExtractionPromptBuilder`, `ExtractionPostProcessor` | `LLMExtractionEngine`, `DefaultPromptBuilder` | Done     |
| 3  | Knowledge Graph + Layer-3 attestation       | pipeline + doctrine   | `GraphStore`, `CommunityDetector`                      | `InMemoryGraphStore`, `Neo4jGraphStore`          | Done     |
| 4  | Hybrid Search                               | pipeline              | `SearchEngine`                                         | `InMemorySearchEngine`, `Neo4jHybridSearch` (RRF) | Done     |
| ~~5~~ | ~~Governed Curation~~                    | retired               | merged into BB3 per ADR-0039 (doctrine in `graph/INTERFACE.md`) |                                          | Retired  |
| 6  | Entity Registry                             | pipeline              | `EntityRegistry`                                       | `InMemoryEntityRegistry` (fuzzy + embedding match) | Done   |
| 7  | Tool Library (agent contract)               | cross-cutting         | `Tool`, `ToolLibrary`                                  | 4 core + 3 temporal tools                        | Done     |
| ~~8~~ | ~~Orchestration~~                        | retired               | deleted per ADR-0039; agent-callability = BB7 + MCP    |                                                  | Retired  |
| 9  | LLM Client                                  | cross-cutting         | `LLMClient`                                            | `AnthropicLLMClient`, `OpenAILLMClient`          | Done     |
| 10 | Retrieval Models                            | cross-cutting         | `EmbeddingModel` (+ `Reranker` planned)                | (`BB10-01` in flight; see capability map)        | Designed |

## Extension Pattern

```python
from graphrag_core import OntologySchema, ToolLibrary, Tool

# 1. Define your domain ontology
schema = OntologySchema(node_types=[...], relationship_types=[...])

# 2. Register domain-specific tools (the agent-callable surface — BB7)
library = ToolLibrary()
library.register(Tool(name="my_tool", description="...", parameters={}, handler=my_handler))
```

External agents (Claude Code, MCP clients, custom harnesses) consume your `ToolLibrary` over MCP — graphrag-core ships the tool contract; the agent harness is out-of-scope per the agentic-substrate doctrine.

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

The code in this repository is MIT-licensed — see [`LICENSE`](LICENSE).

Data fixtures bundled under `eval/fixtures/` retain their own licenses (DocRED is MIT; **FEVEROUS is CC-BY-SA 3.0**). See [`NOTICES.md`](NOTICES.md) for attribution and redistribution terms for each fixture.

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

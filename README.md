# graphrag-core

A domain-agnostic framework for building governed, auditable Knowledge Graphs from documents using LLM-powered extraction, provenance-native storage, and multi-agent orchestration.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  YOUR DOMAIN LAYER (Layer 2)                             │
│  Ontology, domain tools, domain agents, templates        │
└────────────────────────┬────────────────────────────────┘
                         │ imports
┌────────────────────────▼────────────────────────────────┐
│  graphrag-core (Layer 1)                                 │
│                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Ingestion│ │Extraction│ │  Graph   │ │  Search    │ │
│  │ Pipeline │ │  Engine  │ │  Store   │ │  Engine    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Install

```bash
pip install graphrag-core
```

## Status

**v0.1.0** — Interface definitions for Building Blocks 1-4 (Ingestion, Extraction, Graph Store, Search). No default implementations yet.

## Extension Pattern

```python
from graphrag_core import GraphStore, OntologySchema, NodeTypeDefinition

# Define your domain ontology
schema = OntologySchema(
    node_types=[
        NodeTypeDefinition(label="Company", properties=[...]),
    ],
    relationship_types=[...],
)

# Implement the interface for your backend
class MyGraphStore:
    async def merge_node(self, node, import_run_id): ...
    async def get_audit_trail(self, node_id): ...
    # ...
```

## License

MIT

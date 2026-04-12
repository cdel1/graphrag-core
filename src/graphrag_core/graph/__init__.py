"""BB3: Graph store implementations."""

from graphrag_core.graph.memory import InMemoryGraphStore

__all__ = ["InMemoryGraphStore"]

try:
    from graphrag_core.graph.neo4j import Neo4jGraphStore
    __all__.append("Neo4jGraphStore")
except ImportError:
    pass

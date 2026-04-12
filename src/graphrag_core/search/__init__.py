"""BB4: Search engine implementations."""

from graphrag_core.search.memory import InMemorySearchEngine

__all__ = ["InMemorySearchEngine"]

try:
    from graphrag_core.search.neo4j import Neo4jHybridSearch
    __all__.append("Neo4jHybridSearch")
except ImportError:
    pass

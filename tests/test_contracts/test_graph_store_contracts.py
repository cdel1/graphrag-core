"""Bundled backends prove conformance against the contract suite (ADR-0034)."""

from __future__ import annotations

import pytest

from graphrag_core.graph.memory import InMemoryGraphStore
from graphrag_core.interfaces import GraphStore
from graphrag_core.testing.contracts.graph_store import GraphStoreContractTests
from tests.conftest import nominated_neo4j


class TestInMemoryGraphStoreContract(GraphStoreContractTests):
    async def store_factory(self) -> GraphStore:
        return InMemoryGraphStore()


@pytest.mark.integration
class TestNeo4jGraphStoreContract(GraphStoreContractTests):
    persists_across_instances = True
    # persists_schema_across_instances stays False: Neo4jGraphStore.validate_schema()
    # is a stub — tracked as ADR-0034 follow-up debt.

    async def store_factory(self) -> GraphStore:
        from graphrag_core.graph.neo4j import Neo4jGraphStore

        target = nominated_neo4j()
        return Neo4jGraphStore(uri=target.uri, auth=target.auth, database=target.database)

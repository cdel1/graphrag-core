"""flush() conformance: ADR-0033 — visible-now, durable-at-flush.

InMemory is the ephemeral no-op (durability out of scope);
Neo4j is the always-durable no-op (per-call commit). Both must
expose flush() as an async method returning None.
"""

from __future__ import annotations

import inspect

import pytest

from graphrag_core.graph.memory import InMemoryGraphStore
from graphrag_core.models import GraphNode


@pytest.mark.asyncio
async def test_in_memory_flush_is_noop_and_state_stays_visible() -> None:
    store = InMemoryGraphStore()
    await store.merge_node(
        GraphNode(id="n1", label="Entity", properties={}), "run-1"
    )
    result = await store.flush()
    assert result is None
    assert await store.get_node("n1") is not None


def test_neo4j_flush_is_async_method() -> None:
    # No live DB needed: assert the method exists and is a coroutine function.
    Neo4jGraphStore = pytest.importorskip(
        "graphrag_core.graph.neo4j", reason="neo4j extra not installed"
    ).Neo4jGraphStore
    assert inspect.iscoroutinefunction(Neo4jGraphStore.flush)


def test_store_without_flush_fails_protocol() -> None:
    from graphrag_core.interfaces import GraphStore

    assert "flush" in GraphStore.__protocol_attrs__

    class _FlushlessStore:
        async def merge_node(self, node, import_run_id): ...
        async def merge_relationship(self, rel, import_run_id): ...
        async def record_provenance(self, node_id, chunk_id, import_run_id): ...
        async def get_node(self, node_id): ...
        async def get_provenance(self, node_id): ...
        async def get_related(self, node_id, rel_type=None, depth=1): ...
        async def apply_schema(self, schema): ...
        async def validate_schema(self): ...
        async def list_nodes(self): ...
        async def count_relationships(self): ...
        async def list_relationships(self): ...

    assert not isinstance(_FlushlessStore(), GraphStore)
    assert isinstance(InMemoryGraphStore(), GraphStore)

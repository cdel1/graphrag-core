"""GraphStore Protocol conformance suite (ADR-0034).

Subclass GraphStoreContractTests, implement ``store_factory``, and pytest
discovers the contract tests::

    class TestMyStoreContract(GraphStoreContractTests):
        persists_across_instances = True

        async def store_factory(self) -> GraphStore:
            return MyStore(...)

``store_factory`` must return a store bound to the same underlying storage
on every call within one test (this is what makes the lifecycle round-trip
assertable for persistent backends). The suite calls ``clear()`` itself —
the factory does not need to return an empty store. Tests may call ``store_factory`` twice within one test — once to write,
once to obtain a fresh instance reading back the same physical storage; a
factory that returns a brand-new empty store on each call will fail the
round-trip tests.

Requires pytest-asyncio with ``asyncio_mode = "auto"`` in the consumer's
pyproject.toml or pytest.ini. Strict mode is not supported: the async test
methods live on this base class and cannot be marked per-test from a
subclass.
"""

from __future__ import annotations

import asyncio

import pytest

from graphrag_core.exceptions import MissingEndpointError
from graphrag_core.interfaces import GraphStore
from graphrag_core.models import (
    GraphNode,
    GraphRelationship,
    NodeTypeDefinition,
    OntologySchema,
)


def _node(node_id: str, label: str = "Entity", **props: object) -> GraphNode:
    return GraphNode(id=node_id, label=label, properties=dict(props))


def _rel(source: str, target: str, rel_type: str = "RELATES_TO") -> GraphRelationship:
    return GraphRelationship(source_id=source, target_id=target, type=rel_type, properties={})


def _schema_requiring_name() -> OntologySchema:
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(label="Entity", properties=[], required_properties=["name"])
        ],
        relationship_types=[],
    )


class GraphStoreContractTests:
    """Subclass and implement `store_factory` to verify GraphStore conformance."""

    persists_across_instances: bool = False  # enables lifecycle round-trip (state survives re-instantiation)
    requires_concurrency_safety: bool = False  # enables interleaved-writers test
    persists_schema_across_instances: bool = False  # enables schema round-trip via validate_schema

    async def store_factory(self) -> GraphStore:
        raise NotImplementedError("subclass must implement store_factory")

    async def _store(self) -> GraphStore:
        store = await self.store_factory()
        await store.clear()
        return store

    # -- mandatory ---------------------------------------------------------

    async def test_clear_resets_all_state(self) -> None:
        store = await self._store()
        await store.apply_schema(_schema_requiring_name())
        await store.merge_node(_node("a", name="A"), "run-1")
        await store.merge_node(_node("b", name="B"), "run-1")
        await store.merge_relationship(_rel("a", "b"), "run-1")
        await store.record_provenance("a", "chunk-1", "run-1")

        await store.clear()

        assert await store.list_nodes() == []
        assert await store.list_relationships() == []
        assert await store.count_relationships() == 0
        assert await store.get_node("a") is None
        assert (await store.get_provenance("a")).provenance_chain == []
        await store.merge_node(_node("c"), "run-2")  # violates old schema only
        assert await store.validate_schema() == []

    async def test_merge_relationship_missing_endpoint_raises(self) -> None:
        store = await self._store()
        with pytest.raises(MissingEndpointError):
            await store.merge_relationship(_rel("ghost-src", "ghost-tgt"), "run-1")
        await store.merge_node(_node("a"), "run-1")
        with pytest.raises(MissingEndpointError):
            await store.merge_relationship(_rel("a", "ghost-tgt"), "run-1")
        with pytest.raises(MissingEndpointError):
            await store.merge_relationship(_rel("ghost-src", "a"), "run-1")
        await store.merge_node(_node("b"), "run-1")
        rel_id = await store.merge_relationship(_rel("a", "b"), "run-1")
        assert rel_id is not None

    async def test_merge_node_idempotent(self) -> None:
        store = await self._store()
        await store.merge_node(_node("a", name="first"), "run-1")
        await store.merge_node(_node("a", name="second", extra="x"), "run-2")
        assert len(await store.list_nodes()) == 1
        node = await store.get_node("a")
        assert node is not None
        assert node.properties["name"] == "second"
        assert node.properties["extra"] == "x"

    async def test_merge_relationship_upserts_on_key(self) -> None:
        store = await self._store()
        await store.merge_node(_node("a"), "run-1")
        await store.merge_node(_node("b"), "run-1")
        first = GraphRelationship(
            source_id="a", target_id="b", type="REL", properties={"weight": 1}
        )
        second = GraphRelationship(
            source_id="a", target_id="b", type="REL", properties={"weight": 2}
        )
        await store.merge_relationship(first, "run-1")
        await store.merge_relationship(second, "run-2")
        assert await store.count_relationships() == 1
        rels = await store.list_relationships()
        assert rels[0].properties["weight"] == 2

    async def test_audit_trail_reaches_document(self) -> None:
        store = await self._store()
        await store.merge_node(_node("doc-1", label="Document", title="T"), "run-1")
        await store.merge_node(_node("chunk-1", label="Chunk", text="body"), "run-1")
        # Document traversal is expected to follow CHUNKED_FROM edges (ADR-0001).
        await store.merge_relationship(_rel("chunk-1", "doc-1", "CHUNKED_FROM"), "run-1")
        await store.merge_node(_node("entity-1", name="E"), "run-1")
        await store.record_provenance("entity-1", "chunk-1", "run-1")

        trail = await store.get_provenance("entity-1")
        levels = [step.level for step in trail.provenance_chain]
        assert "node" in levels
        assert "chunk" in levels
        assert "document" in levels
        doc_steps = [s for s in trail.provenance_chain if s.level == "document"]
        assert doc_steps[0].id == "doc-1"

    async def test_flush_is_legal_and_state_remains_visible(self) -> None:
        store = await self._store()
        await store.merge_node(_node("a"), "run-1")
        await store.flush()
        assert await store.get_node("a") is not None
        await store.flush()  # retry is legal (ADR-0033)

    # -- gated: persists_across_instances ----------------------------------

    async def test_lifecycle_round_trip(self) -> None:
        if not self.persists_across_instances:
            pytest.skip("backend does not claim cross-instance persistence")
        store = await self._store()
        await store.merge_node(_node("doc-1", label="Document", title="T"), "run-1")
        await store.merge_node(_node("chunk-1", label="Chunk", text="body"), "run-1")
        await store.merge_relationship(_rel("chunk-1", "doc-1", "CHUNKED_FROM"), "run-1")
        await store.merge_node(_node("entity-1", name="E"), "run-1")
        await store.record_provenance("entity-1", "chunk-1", "run-1")
        await store.flush()

        reborn = await self.store_factory()  # same storage, fresh instance

        assert await reborn.get_node("entity-1") is not None
        assert await reborn.count_relationships() == 1
        trail = await reborn.get_provenance("entity-1")
        levels = [step.level for step in trail.provenance_chain]
        assert "document" in levels  # _chunk_to_doc survived (S5 bug class)

    # -- gated: persists_schema_across_instances ----------------------------

    async def test_schema_survives_round_trip(self) -> None:
        if not self.persists_schema_across_instances:
            pytest.skip("backend does not claim cross-instance schema persistence")
        store = await self._store()
        await store.apply_schema(_schema_requiring_name())
        await store.flush()

        reborn = await self.store_factory()  # same storage, fresh instance

        await reborn.merge_node(_node("no-name"), "run-2")
        violations = await reborn.validate_schema()
        assert any(v.node_id == "no-name" for v in violations)  # _schema survived (S5 bug class)

    # -- gated: requires_concurrency_safety ---------------------------------

    async def test_concurrent_writers_no_lost_updates(self) -> None:
        if not self.requires_concurrency_safety:
            pytest.skip("backend does not claim concurrency safety")
        store = await self._store()

        async def write(prefix: str) -> None:
            for i in range(50):
                await store.merge_node(_node(f"{prefix}-{i}"), prefix)

        await asyncio.gather(write("w1"), write("w2"))
        assert len(await store.list_nodes()) == 100

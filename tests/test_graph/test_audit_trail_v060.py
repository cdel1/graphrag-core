# tests/test_graph/test_audit_trail_v060.py
import pytest
from graphrag_core.graph.memory import InMemoryGraphStore
from graphrag_core.models import GraphNode, GraphRelationship


@pytest.mark.asyncio
async def test_memory_store_indexes_chunked_from_edges():
    store = InMemoryGraphStore()
    # Create a Document node + Chunk node + CHUNKED_FROM edge
    await store.merge_node(
        GraphNode(id="doc:1", label="Document",
                  properties={"title": "Q2 report", "period": "2026-Q2"}),
        import_run_id="run-1",
    )
    chunk_id = "chunk:1"
    await store.merge_relationship(
        GraphRelationship(source_id=chunk_id, target_id="doc:1",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-1",
    )

    # The internal index used by get_audit_trail must resolve chunk -> doc
    assert store._chunk_to_doc.get(chunk_id) == "doc:1"


@pytest.mark.asyncio
async def test_chunked_from_index_updates_on_remerge():
    """A re-merged CHUNKED_FROM edge (same triple) must keep the index in sync."""
    store = InMemoryGraphStore()

    # First merge: chunk:1 -> doc:1 (insert path)
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:1",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-1",
    )
    assert store._chunk_to_doc["chunk:1"] == "doc:1"

    # Re-merge the identical triple with updated properties (update path)
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:1",
                          type="CHUNKED_FROM", properties={"updated": True}),
        import_run_id="run-2",
    )
    # Index must still be correct after the update path
    assert store._chunk_to_doc["chunk:1"] == "doc:1"
    # Only one relationship row should exist (idempotent merge)
    rels = await store.list_relationships()
    assert len(rels) == 1


@pytest.mark.asyncio
async def test_chunked_from_index_different_target_is_new_row():
    """Re-merging with a different target_id creates a second row; both are indexed."""
    store = InMemoryGraphStore()

    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:1",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-1",
    )
    assert store._chunk_to_doc["chunk:1"] == "doc:1"

    # Different target_id -> different triple key -> new row (not an update)
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:2",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-2",
    )
    # Second insert updates the index to the latest value
    assert store._chunk_to_doc["chunk:1"] == "doc:2"
    # Two separate relationship rows exist
    rels = await store.list_relationships()
    assert len(rels) == 2

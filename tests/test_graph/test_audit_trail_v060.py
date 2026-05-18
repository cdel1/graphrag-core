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

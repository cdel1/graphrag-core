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
    await store.merge_node(
        GraphNode(id=chunk_id, label="Chunk", properties={}),
        import_run_id="run-1",
    )
    await store.merge_relationship(
        GraphRelationship(source_id=chunk_id, target_id="doc:1",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-1",
    )

    # The internal index used by get_provenance must resolve chunk -> doc
    assert store._chunk_to_doc.get(chunk_id) == "doc:1"


@pytest.mark.asyncio
async def test_chunked_from_index_updates_on_remerge():
    """A re-merged CHUNKED_FROM edge (same triple) must keep the index in sync."""
    store = InMemoryGraphStore()

    await store.merge_node(GraphNode(id="chunk:1", label="Chunk", properties={}), "run-1")
    await store.merge_node(GraphNode(id="doc:1", label="Document", properties={}), "run-1")

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

    await store.merge_node(GraphNode(id="chunk:1", label="Chunk", properties={}), "run-1")
    await store.merge_node(GraphNode(id="doc:1", label="Document", properties={}), "run-1")
    await store.merge_node(GraphNode(id="doc:2", label="Document", properties={}), "run-1")

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


@pytest.mark.asyncio
async def test_memory_audit_trail_reaches_document():
    store = InMemoryGraphStore()
    # Document
    await store.merge_node(
        GraphNode(id="doc:1", label="Document",
                  properties={"title": "Q2 report", "period": "2026-Q2",
                              "date": "2026-05-01"}),
        import_run_id="run-1",
    )
    # Node + provenance to chunk
    await store.merge_node(GraphNode(id="claim:1", label="Claim", properties={}),
                           "run-1")
    await store.record_provenance("claim:1", "chunk:1", "run-1")
    await store.merge_node(GraphNode(id="chunk:1", label="Chunk", properties={}), "run-1")
    # Chunk -> Document edge
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:1",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-1",
    )

    trail = await store.get_provenance("claim:1")
    levels = [step.level for step in trail.provenance_chain]
    assert "document" in levels

    doc_step = next(s for s in trail.provenance_chain if s.level == "document")
    assert doc_step.id == "doc:1"
    assert doc_step.metadata.get("period") == "2026-Q2"
    assert doc_step.metadata.get("title") == "Q2 report"


@pytest.mark.asyncio
async def test_memory_audit_trail_missing_node_returns_empty_chain():
    store = InMemoryGraphStore()
    trail = await store.get_provenance("nonexistent")
    assert trail.node_id == "nonexistent"
    assert trail.provenance_chain == []


@pytest.mark.asyncio
async def test_memory_audit_trail_node_without_provenance_returns_node_step_only():
    """A node that exists but has no chunks yields just the node-level step."""
    store = InMemoryGraphStore()
    await store.merge_node(
        GraphNode(id="claim:lonely", label="Claim", properties={}),
        import_run_id="run-1",
    )
    trail = await store.get_provenance("claim:lonely")
    levels = [s.level for s in trail.provenance_chain]
    assert levels == ["node"]


@pytest.mark.asyncio
async def test_memory_audit_trail_deduplicates_document_step():
    """A node with multiple chunks from the same document should emit
    the document-level ProvenanceStep exactly once (seen_docs dedup)."""
    store = InMemoryGraphStore()
    await store.merge_node(
        GraphNode(id="doc:1", label="Document",
                  properties={"period": "2026-Q2", "title": "Q2 report"}),
        import_run_id="run-1",
    )
    await store.merge_node(GraphNode(id="claim:1", label="Claim", properties={}),
                           import_run_id="run-1")
    await store.record_provenance("claim:1", "chunk:1", "run-1")
    await store.record_provenance("claim:1", "chunk:2", "run-1")
    for chunk_id in ("chunk:1", "chunk:2"):
        await store.merge_node(GraphNode(id=chunk_id, label="Chunk", properties={}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id=chunk_id, target_id="doc:1",
                              type="CHUNKED_FROM", properties={}),
            import_run_id="run-1",
        )

    trail = await store.get_provenance("claim:1")
    doc_steps = [s for s in trail.provenance_chain if s.level == "document"]
    assert len(doc_steps) == 1, (
        f"expected exactly 1 document step, got {len(doc_steps)}"
    )
    # Node must be first; remaining steps may be in any backend-defined order.
    levels = [s.level for s in trail.provenance_chain]
    assert levels[0] == "node"  # node always first
    assert sorted(levels[1:]) == ["chunk", "chunk", "document"]  # rest in any order


# ---------------------------------------------------------------------------
# Neo4j integration tests — skipped unless --run-integration is passed
# ---------------------------------------------------------------------------

import os as _os

_NEO4J_TEST_DB = _os.environ.get("NEO4J_TEST_DATABASE", "neo4j")


@pytest.fixture
async def neo4j_test_store():
    from graphrag_core.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(database=_NEO4J_TEST_DB)
    async with store._driver.session(database=_NEO4J_TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield store
    await store.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_neo4j_audit_trail_reaches_document(neo4j_test_store):
    """Requires running Neo4j; gated by integration marker."""
    store = neo4j_test_store
    await store.merge_node(
        GraphNode(id="doc:1", label="Document",
                  properties={"title": "Q2 report", "period": "2026-Q2"}),
        import_run_id="run-1",
    )
    await store.merge_node(
        GraphNode(id="claim:1", label="Claim", properties={}),
        import_run_id="run-1",
    )
    await store.record_provenance("claim:1", "chunk:1", "run-1")
    await store.merge_relationship(
        GraphRelationship(source_id="chunk:1", target_id="doc:1",
                          type="CHUNKED_FROM", properties={}),
        import_run_id="run-1",
    )

    trail = await store.get_provenance("claim:1")
    doc_step = next((s for s in trail.provenance_chain if s.level == "document"), None)
    assert doc_step is not None
    assert doc_step.id == "doc:1"
    assert doc_step.metadata.get("period") == "2026-Q2"
    assert doc_step.metadata.get("title") == "Q2 report"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_document_uniqueness_constraint(neo4j_test_store):
    """After apply_schema runs, Neo4j has a uniqueness constraint on :Document(id)."""
    from graphrag_core.models import OntologySchema

    schema = OntologySchema(node_types=[], relationship_types=[])
    await neo4j_test_store.apply_schema(schema)

    async with neo4j_test_store._driver.session(
        database=neo4j_test_store._database
    ) as session:
        result = await session.run("SHOW CONSTRAINTS")
        constraints = [dict(record) async for record in result]
        doc_constraints = [
            c for c in constraints
            if c.get("labelsOrTypes") == ["Document"]
               and "id" in (c.get("properties") or [])
        ]
        assert len(doc_constraints) >= 1, (
            f"Expected a :Document(id) uniqueness constraint, "
            f"got constraints: {constraints}"
        )

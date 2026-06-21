"""BB1 v0.6.0: IngestionPipeline writes :Document nodes + FROM_DOCUMENT edges."""

import pytest
from graphrag_core.ingestion.pipeline import IngestionPipeline
from graphrag_core.ingestion.parsers import TextParser
from graphrag_core.ingestion.chunker import TokenChunker
from graphrag_core.graph.memory import InMemoryGraphStore
from graphrag_core.models import (
    ChunkConfig,
    DocumentMetadata,
    ParsedDocument,
    TextSection,
)


def _make_fake_parse(metadata: DocumentMetadata, text: str = "body"):
    async def fake_parse(source, content_type):
        return ParsedDocument(
            sections=[TextSection(heading="", text=text, page=1)],
            metadata=metadata,
        )
    return fake_parse


@pytest.mark.asyncio
async def test_ingest_writes_document_node(monkeypatch):
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="My Doc", source="src1", doc_type="report",
        date=None, quarter=None, period="2026-Q2", sha256="sha-1",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    await pipeline.ingest(
        b"body", "text/plain", config=ChunkConfig(),
        graph_store=store, import_run_id="run-1",
    )

    docs = [n for n in await store.list_nodes() if n.label == "Document"]
    assert len(docs) == 1
    assert docs[0].properties["title"] == "My Doc"
    assert docs[0].properties["period"] == "2026-Q2"
    assert docs[0].properties["sha256"] == "sha-1"


@pytest.mark.asyncio
async def test_ingest_writes_chunked_from_edges(monkeypatch):
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter=None, period="2026-Q2", sha256="sha-2",
    )
    # Long enough to produce multiple chunks
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="paragraph " * 200))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    chunks = await pipeline.ingest(
        b"body", "text/plain", config=ChunkConfig(),
        graph_store=store, import_run_id="run-1",
    )

    rels = [r for r in await store.list_relationships() if r.type == "FROM_DOCUMENT"]
    assert len(rels) == len(chunks)
    assert all(r.target_id == "doc:sha-2" for r in rels)


@pytest.mark.asyncio
async def test_ingest_idempotent_on_same_sha(monkeypatch):
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter=None, period="P", sha256="same-sha",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="x"))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    await pipeline.ingest(b"x", "text/plain", graph_store=store, import_run_id="r1")
    await pipeline.ingest(b"x", "text/plain", graph_store=store, import_run_id="r2")

    docs = [n for n in await store.list_nodes() if n.label == "Document"]
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_ingest_without_graph_store_returns_chunks_only(monkeypatch):
    """Backward-compat path: no graph_store kwarg, no exception, no mutations."""
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter=None, period=None, sha256="x",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="x"))

    pipeline = IngestionPipeline(parser, chunker)
    chunks = await pipeline.ingest(b"x", "text/plain")
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_quarter_falls_back_to_period(monkeypatch):
    """When period is None and quarter is set, BB1 copies quarter into period
    on the Document node properties (Task 6 — backwards-compat transition)."""
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter="2026-Q2", period=None, sha256="qfall-1",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="x"))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    await pipeline.ingest(
        b"x", "text/plain", graph_store=store, import_run_id="r1",
    )

    docs = [n for n in await store.list_nodes() if n.label == "Document"]
    assert len(docs) == 1
    assert docs[0].properties.get("period") == "2026-Q2"


@pytest.mark.asyncio
async def test_quarter_is_stripped_from_persisted_document(monkeypatch):
    """quarter is a deprecated field — it must not land on the :Document node."""
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter="2026-Q2", period=None, sha256="strip-test",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="x"))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    await pipeline.ingest(b"x", "text/plain", graph_store=store, import_run_id="r1")

    docs = [n for n in await store.list_nodes() if n.label == "Document"]
    assert len(docs) == 1
    # quarter must not appear on the Document node properties
    assert "quarter" not in docs[0].properties
    # period must still be populated from the fallback
    assert docs[0].properties["period"] == "2026-Q2"


@pytest.mark.asyncio
async def test_ingest_raises_if_graph_store_without_import_run_id(monkeypatch):
    """Defensive: graph_store provided but import_run_id missing -> ValueError."""
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter=None, period=None, sha256="x",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="x"))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    with pytest.raises(ValueError, match="import_run_id"):
        await pipeline.ingest(b"x", "text/plain", graph_store=store)


@pytest.mark.asyncio
async def test_ingest_creates_chunk_nodes_before_chunked_from_edges(monkeypatch):
    """Regression: BB1 must merge :Chunk nodes before FROM_DOCUMENT edges so
    Neo4j MERGE-with-MATCH-endpoints semantics work end-to-end (see v0.6.1 fix).

    The InMemoryGraphStore is permissive (relationships don't require nodes
    to exist), so prior to the fix this test would have passed silently on
    Memory while Neo4j crashed with a None record at MERGE time.
    """
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="t", source="s", doc_type="d",
        date=None, quarter=None, period="2026-Q1", sha256="chunknode-1",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="paragraph " * 50))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    chunks = await pipeline.ingest(
        b"x", "text/plain", config=ChunkConfig(),
        graph_store=store, import_run_id="run-1",
    )

    all_nodes = await store.list_nodes()
    chunk_nodes = [n for n in all_nodes if n.label == "Chunk"]
    assert len(chunk_nodes) == len(chunks), (
        f"expected {len(chunks)} :Chunk nodes, got {len(chunk_nodes)}"
    )
    # Chunk nodes must carry the chunk text + page/position from the parsed doc
    chunk_ids = {n.id for n in chunk_nodes}
    assert chunk_ids == {c.id for c in chunks}
    for n in chunk_nodes:
        assert "text" in n.properties


# ---------------------------------------------------------------------------
# Neo4j integration regression — the bug the v0.6.1 fix addresses

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
async def test_neo4j_ingest_creates_chunk_nodes_and_chunked_from(
    neo4j_test_store, monkeypatch,
):
    """Regression: with Neo4j, BB1 must merge :Chunk nodes BEFORE the
    FROM_DOCUMENT edges or merge_relationship's MATCH (a),(b) returns None
    and the call raises TypeError. The InMemoryGraphStore path is
    permissive and won't catch this — only a live-Neo4j test will.
    """
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="Neo4j Smoke", source="smoke", doc_type="report",
        date=None, quarter=None, period="2026-Q1", sha256="neo4j-smoke",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="paragraph " * 50))

    pipeline = IngestionPipeline(parser, chunker)
    chunks = await pipeline.ingest(
        b"x", "text/plain", config=ChunkConfig(),
        graph_store=neo4j_test_store, import_run_id="run-1",
    )

    # Document node exists with period
    async with neo4j_test_store._driver.session(database=_NEO4J_TEST_DB) as session:
        result = await session.run(
            "MATCH (d:Document {id: $id}) RETURN d.period AS period, d.title AS title",
            id="doc:neo4j-smoke",
        )
        record = await result.single()
    assert record is not None
    assert record["period"] == "2026-Q1"

    # Chunk nodes exist, count == chunks returned
    async with neo4j_test_store._driver.session(database=_NEO4J_TEST_DB) as session:
        result = await session.run("MATCH (c:Chunk) RETURN count(c) AS n")
        n_chunks_in_graph = (await result.single())["n"]
    assert n_chunks_in_graph == len(chunks)

    # FROM_DOCUMENT edges link every chunk to the document
    async with neo4j_test_store._driver.session(database=_NEO4J_TEST_DB) as session:
        result = await session.run(
            "MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document {id: $id}) RETURN count(c) AS n",
            id="doc:neo4j-smoke",
        )
        n_edges = (await result.single())["n"]
    assert n_edges == len(chunks)


# ---------------------------------------------------------------------------
# BB1-07: NEXT_CHUNK adjacency edges


@pytest.mark.asyncio
async def test_ingest_writes_next_chunk_adjacency(monkeypatch):
    """BB1-07: consecutive chunks linked (prev)-[:NEXT_CHUNK]->(next), single-directed."""
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="adj", source="s", doc_type="d",
        date=None, quarter=None, period="2026-Q2", sha256="adj-1",
    )
    # Long enough to produce multiple chunks
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="paragraph " * 200))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    chunks = await pipeline.ingest(
        b"body", "text/plain", config=ChunkConfig(max_tokens=10),
        graph_store=store, import_run_id="run-1",
    )

    assert len(chunks) >= 2, "need multiple chunks to test adjacency"

    next_chunk_rels = [r for r in await store.list_relationships() if r.type == "NEXT_CHUNK"]

    # exactly one edge per consecutive pair
    assert len(next_chunk_rels) == len(chunks) - 1

    # edges are ordered: (chunks[i]) -> (chunks[i+1])
    for i, (prev, nxt) in enumerate(zip(chunks, chunks[1:])):
        assert next_chunk_rels[i].source_id == prev.id
        assert next_chunk_rels[i].target_id == nxt.id

    # no PREV_CHUNK reverse edges
    prev_chunk_rels = [r for r in await store.list_relationships() if r.type == "PREV_CHUNK"]
    assert len(prev_chunk_rels) == 0


@pytest.mark.asyncio
async def test_ingest_single_chunk_no_next_chunk_edges(monkeypatch):
    """BB1-07: single-chunk doc writes zero NEXT_CHUNK edges."""
    parser = TextParser()
    chunker = TokenChunker()
    metadata = DocumentMetadata(
        title="single", source="s", doc_type="d",
        date=None, quarter=None, period="2026-Q2", sha256="single-1",
    )
    monkeypatch.setattr(parser, "parse", _make_fake_parse(metadata, text="short"))

    pipeline = IngestionPipeline(parser, chunker)
    store = InMemoryGraphStore()
    chunks = await pipeline.ingest(
        b"short", "text/plain", config=ChunkConfig(),
        graph_store=store, import_run_id="run-1",
    )

    assert len(chunks) == 1
    next_chunk_rels = [r for r in await store.list_relationships() if r.type == "NEXT_CHUNK"]
    assert len(next_chunk_rels) == 0

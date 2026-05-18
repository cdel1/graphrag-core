"""BB1 v0.6.0: IngestionPipeline writes :Document nodes + CHUNKED_FROM edges."""

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

    rels = [r for r in await store.list_relationships() if r.type == "CHUNKED_FROM"]
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

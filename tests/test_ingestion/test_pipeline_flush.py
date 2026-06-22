"""ADR-0033 D3: ingest() flushes exactly once, after all mutations;
a failing final flush fails the ingest."""

import pytest

from graphrag_core.exceptions import GraphStoreError
from graphrag_core.ingestion.pipeline import IngestionPipeline
from graphrag_core.models import ChunkConfig, Chunk, ParsedDocument


class _FakeParser:
    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        return ParsedDocument(sections=[_section()], metadata=_metadata())


class _FakeChunker:
    def chunk(self, parsed: ParsedDocument, config: ChunkConfig) -> list[Chunk]:
        return [Chunk(id="chunk:1", text=parsed.sections[0].text)]


def _metadata():
    from graphrag_core.models import DocumentMetadata

    return DocumentMetadata(
        title="", source="", doc_type="text/plain",
        date=None, quarter=None, sha256="abc123",
    )


def _section():
    from graphrag_core.models import TextSection

    return TextSection(heading=None, text="hello world")


class _RecordingStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def merge_node(self, node, import_run_id):
        self.calls.append("merge_node")
        return node.id

    async def merge_relationship(self, rel, import_run_id):
        self.calls.append("merge_relationship")
        return f"{rel.source_id}->{rel.target_id}"

    async def flush(self) -> None:
        self.calls.append("flush")


class _FailingFlushStore(_RecordingStore):
    async def flush(self) -> None:
        raise GraphStoreError("disk full")


async def test_ingest_flushes_exactly_once_after_all_mutations() -> None:
    store = _RecordingStore()
    pipeline = IngestionPipeline(parser=_FakeParser(), chunker=_FakeChunker())
    await pipeline.ingest(
        b"raw", "text/plain", graph_store=store, import_run_id="run-1"
    )
    assert store.calls.count("flush") == 1
    assert store.calls[-1] == "flush"


async def test_ingest_without_store_needs_no_flush() -> None:
    pipeline = IngestionPipeline(parser=_FakeParser(), chunker=_FakeChunker())
    chunks = await pipeline.ingest(b"raw", "text/plain")
    assert len(chunks) == 1


async def test_failed_final_flush_fails_the_ingest() -> None:
    store = _FailingFlushStore()
    pipeline = IngestionPipeline(parser=_FakeParser(), chunker=_FakeChunker())
    with pytest.raises(GraphStoreError):
        await pipeline.ingest(
            b"raw", "text/plain", graph_store=store, import_run_id="run-1"
        )

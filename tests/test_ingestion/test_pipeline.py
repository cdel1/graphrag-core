"""Tests for IngestionPipeline."""

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentMetadata,
    ParsedDocument,
    TextSection,
)


class FakeParser:
    """Fake DocumentParser that returns a fixed document."""

    def __init__(self, doc: ParsedDocument):
        self._doc = doc

    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        return self._doc


class FakeChunker:
    """Fake Chunker that returns one chunk per section."""

    def chunk(self, doc, config):
        from graphrag_core.models import DocumentChunk
        return [
            DocumentChunk(id=f"chunk-{i}", text=s.text, position=i)
            for i, s in enumerate(doc.sections)
        ]


class FakeEmbeddingModel:
    """Fake EmbeddingModel that returns fixed-length vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


def _make_doc() -> ParsedDocument:
    return ParsedDocument(
        sections=[
            TextSection(heading=None, text="Hello world."),
            TextSection(heading=None, text="Goodbye world."),
        ],
        metadata=DocumentMetadata(
            title="", source="", doc_type="text/plain",
            date=None, quarter=None, sha256="abc123",
        ),
    )


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_ingest_without_embedding(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()
        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=FakeChunker(),
        )
        chunks = await pipeline.ingest(b"dummy", "text/plain")

        assert len(chunks) == 2
        assert chunks[0].text == "Hello world."
        assert chunks[1].text == "Goodbye world."
        assert chunks[0].embedding is None
        assert chunks[1].embedding is None

    @pytest.mark.asyncio
    async def test_ingest_with_embedding(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()
        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=FakeChunker(),
            embedding_model=FakeEmbeddingModel(),
        )
        chunks = await pipeline.ingest(b"dummy", "text/plain")

        assert len(chunks) == 2
        assert chunks[0].embedding == [1.0, 2.0, 3.0]
        assert chunks[1].embedding == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_ingest_uses_default_config(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()

        class ConfigCapturingChunker:
            captured_config = None

            def chunk(self, doc, config):
                ConfigCapturingChunker.captured_config = config
                return []

        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=ConfigCapturingChunker(),
        )
        await pipeline.ingest(b"dummy", "text/plain")

        assert ConfigCapturingChunker.captured_config is not None
        assert ConfigCapturingChunker.captured_config.max_tokens == 512
        assert ConfigCapturingChunker.captured_config.overlap == 50

    @pytest.mark.asyncio
    async def test_ingest_uses_custom_config(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline

        doc = _make_doc()

        class ConfigCapturingChunker:
            captured_config = None

            def chunk(self, doc, config):
                ConfigCapturingChunker.captured_config = config
                return []

        pipeline = IngestionPipeline(
            parser=FakeParser(doc),
            chunker=ConfigCapturingChunker(),
        )
        custom = ChunkConfig(max_tokens=100, overlap=10)
        await pipeline.ingest(b"dummy", "text/plain", config=custom)

        assert ConfigCapturingChunker.captured_config.max_tokens == 100
        assert ConfigCapturingChunker.captured_config.overlap == 10

    @pytest.mark.asyncio
    async def test_pipeline_satisfies_protocol(self):
        from graphrag_core.ingestion.pipeline import IngestionPipeline as PipelineImpl
        from graphrag_core.interfaces import IngestionPipeline as PipelineProtocol

        assert isinstance(
            PipelineImpl(parser=FakeParser(_make_doc()), chunker=FakeChunker()),
            PipelineProtocol,
        )

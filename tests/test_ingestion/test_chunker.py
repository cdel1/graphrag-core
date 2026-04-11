"""Tests for TokenChunker."""

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentChunk,
    DocumentMetadata,
    ParsedDocument,
    TextSection,
)


def _doc(texts: list[str], sha: str = "abc123def456") -> ParsedDocument:
    """Build a minimal ParsedDocument for testing."""
    return ParsedDocument(
        sections=[TextSection(heading=None, text=t) for t in texts],
        metadata=DocumentMetadata(
            title="", source="", doc_type="text/plain",
            date=None, quarter=None, sha256=sha,
        ),
    )


class TestTokenChunker:
    @pytest.fixture
    def chunker(self):
        from graphrag_core.ingestion.chunker import TokenChunker
        return TokenChunker()

    def test_single_chunk_below_max(self, chunker):
        doc = _doc(["one two three"])
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 1
        assert chunks[0].text == "one two three"
        assert chunks[0].position == 0
        assert chunks[0].id == "abc123def456-0"
        assert chunks[0].embedding is None
        assert chunks[0].chunk_type == "text"

    def test_splits_at_max_tokens(self, chunker):
        doc = _doc(["a b c d e f g h i j"])
        config = ChunkConfig(max_tokens=5, overlap=0)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 2
        assert chunks[0].text == "a b c d e"
        assert chunks[1].text == "f g h i j"
        assert chunks[0].position == 0
        assert chunks[1].position == 1

    def test_overlap(self, chunker):
        doc = _doc(["a b c d e f g h i j"])
        config = ChunkConfig(max_tokens=5, overlap=2)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 3
        assert chunks[0].text == "a b c d e"
        assert chunks[1].text == "d e f g h"
        assert chunks[2].text == "g h i j"

    def test_multiple_sections_concatenated(self, chunker):
        doc = _doc(["a b c", "d e f"])
        config = ChunkConfig(max_tokens=4, overlap=0)
        chunks = chunker.chunk(doc, config)

        assert len(chunks) == 2
        assert chunks[0].text == "a b c d"
        assert chunks[1].text == "e f"

    def test_empty_document(self, chunker):
        doc = _doc([])
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)
        assert chunks == []

    def test_page_number_from_section(self, chunker):
        doc = ParsedDocument(
            sections=[TextSection(heading=None, text="word " * 3, page=5)],
            metadata=DocumentMetadata(
                title="", source="", doc_type="text/plain",
                date=None, quarter=None, sha256="abc123def456",
            ),
        )
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)
        assert chunks[0].page == 5

    def test_id_uses_sha_prefix(self, chunker):
        doc = _doc(["hello world"], sha="abcdef1234567890abcdef")
        config = ChunkConfig(max_tokens=10, overlap=0)
        chunks = chunker.chunk(doc, config)
        assert chunks[0].id == "abcdef123456-0"

    def test_default_config(self, chunker):
        config = ChunkConfig()
        assert config.max_tokens == 512
        assert config.overlap == 50

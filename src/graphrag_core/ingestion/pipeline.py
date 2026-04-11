"""BB1: Ingestion pipeline orchestrator."""

from __future__ import annotations

from graphrag_core.interfaces import Chunker, DocumentParser, EmbeddingModel
from graphrag_core.models import ChunkConfig, DocumentChunk


class IngestionPipeline:
    """Wires together parser, chunker, and optional embedding model."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_model = embedding_model

    async def ingest(
        self,
        source: bytes,
        content_type: str,
        config: ChunkConfig | None = None,
    ) -> list[DocumentChunk]:
        parsed = await self._parser.parse(source, content_type)
        chunks = self._chunker.chunk(parsed, config or ChunkConfig())

        if self._embedding_model is not None:
            embeddings = await self._embedding_model.embed([c.text for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

        return chunks

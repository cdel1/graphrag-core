"""BB1: Token-based chunker."""

from __future__ import annotations

from graphrag_core.models import ChunkConfig, DocumentChunk, ParsedDocument


class TokenChunker:
    """Splits documents into chunks by whitespace token count with overlap."""

    def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> list[DocumentChunk]:
        words: list[tuple[str, int | None]] = []
        for section in doc.sections:
            for word in section.text.split():
                words.append((word, section.page))

        if not words:
            return []

        sha_prefix = doc.metadata.sha256[:12]
        chunks: list[DocumentChunk] = []
        start = 0
        position = 0

        step = max(config.max_tokens - config.overlap, 1)

        while start < len(words):
            end = min(start + config.max_tokens, len(words))
            # Skip chunks whose content was already covered by the previous chunk
            if chunks and start < chunks[-1].position * step + config.max_tokens and end <= start:
                break
            chunk_words = words[start:end]
            text = " ".join(w for w, _ in chunk_words)
            page = chunk_words[0][1] if chunk_words else None

            chunks.append(DocumentChunk(
                id=f"{sha_prefix}-{position}",
                text=text,
                page=page,
                position=position,
            ))

            position += 1
            if end == len(words):
                break
            start += step

        return chunks

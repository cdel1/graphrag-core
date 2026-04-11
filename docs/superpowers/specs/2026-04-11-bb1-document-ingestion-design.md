# BB1: Document Ingestion — Design Spec

## Overview

Default implementation of BB1 (Document Ingestion) for graphrag-core. Provides parsers for PDF, DOCX, plain text, and Markdown; a token-based chunker; and a pipeline that wires them together.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| File formats | PDF, DOCX, plain text, Markdown | Covers majority of document-heavy knowledge work |
| Chunking strategy | Token-based with overlap (whitespace split) | Matches existing ChunkConfig model, zero dependencies |
| Embedding model | No default implementation | Domain layers bring their own; avoids vendor coupling |
| Pipeline | Yes, IngestionPipeline class | Thin coordinator, single entry point for BB1 |
| PDF library | pypdf | Lightweight, pure Python, sufficient for text PDFs |
| DOCX library | python-docx | Standard, handles headings/paragraphs/structure |
| Code organization | Package with separate files | Matches CLAUDE.md project structure |

## Package Structure

```
src/graphrag_core/ingestion/
├── __init__.py          # Re-exports: PdfParser, DocxParser, TextParser, MarkdownParser, TokenChunker, IngestionPipeline
├── parsers.py           # PdfParser, DocxParser, TextParser, MarkdownParser
├── chunker.py           # TokenChunker
└── pipeline.py          # IngestionPipeline
```

## Parsers (`parsers.py`)

All parsers implement `DocumentParser` Protocol: `async def parse(self, source: bytes, content_type: str) -> ParsedDocument`.

All parsers:
- Accept `source: bytes`, decode as UTF-8 (except PDF which is binary)
- Construct `DocumentMetadata` with `sha256` hash of input bytes, `doc_type` from content type, `title`/`source` as empty string, `date`/`quarter` as `None`
- Raise `ValueError` if `content_type` doesn't match

### PdfParser
- Uses `pypdf.PdfReader` to extract text per page
- Each page becomes a `TextSection` with `page` number set, `heading=None`
- Expected content type: `application/pdf`

### DocxParser
- Uses `python-docx` to iterate paragraphs
- Groups consecutive paragraphs under headings (detected by `paragraph.style.name.startswith("Heading")`)
- Each heading group becomes a `TextSection`
- Paragraphs before the first heading: `heading=None`
- Expected content type: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`

### TextParser
- Splits on double newlines into sections
- Each section: `TextSection` with `heading=None`, no page info
- Expected content type: `text/plain`

### MarkdownParser
- Splits on `#`/`##`/`###` heading patterns (regex: `^#{1,6}\s+`)
- Each heading becomes a `TextSection` with heading text extracted
- Content before first heading: `heading=None` section
- Expected content type: `text/markdown`

## Chunker (`chunker.py`)

### TokenChunker
- Implements `Chunker` Protocol: `def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> list[DocumentChunk]`
- Tokenization: whitespace split (word-level)
- Walks all `TextSection`s, building chunks up to `config.max_tokens` words
- Overlap: last `config.overlap` words of a chunk become the start of the next
- `DocumentChunk` fields:
  - `id`: `{doc.metadata.sha256[:12]}-{position}`
  - `text`: chunk content
  - `page`: from `TextSection` if available
  - `position`: sequential index (0, 1, 2, ...)
  - `embedding`: `None`
  - `chunk_type`: `"text"`

## Pipeline (`pipeline.py`)

### IngestionPipeline
- Concrete class
- Constructor: `parser: DocumentParser`, `chunker: Chunker`, `embedding_model: EmbeddingModel | None = None`
- Single public method:
  ```python
  async def ingest(self, source: bytes, content_type: str, config: ChunkConfig | None = None) -> list[DocumentChunk]
  ```
- Flow:
  1. `parsed = await self.parser.parse(source, content_type)`
  2. `chunks = self.chunker.chunk(parsed, config or ChunkConfig())`
  3. If `embedding_model` provided: `embeddings = await self.embedding_model.embed([c.text for c in chunks])`, assign back to chunks
  4. Return chunks

## Interface Update

Add `IngestionPipeline` Protocol to `interfaces.py`:
```python
@runtime_checkable
class IngestionPipeline(Protocol):
    async def ingest(self, source: bytes, content_type: str, config: ChunkConfig | None = None) -> list[DocumentChunk]: ...
```

Add to `__init__.py` re-exports.

## Dependencies

Add to `pyproject.toml`:
- `pypdf>=4.0`
- `python-docx>=1.0`

## Testing

Tests in `tests/test_ingestion/`:

### `test_parsers.py`
- Each parser tested with small inline byte inputs
- Verifies correct `TextSection` structure, `DocumentMetadata` fields, `sha256` hash
- Verifies `ValueError` on wrong content type

### `test_chunker.py`
- Token splitting respects `max_tokens`
- Overlap works correctly
- Edge cases: empty document, single section shorter than max_tokens, section with exact max_tokens length

### `test_pipeline.py`
- Wiring test: parser -> chunker -> chunks returned
- With embedding model: embeddings assigned to chunks
- Without embedding model: embeddings remain None
- Uses simple fakes implementing the Protocols

All tests are unit tests with no external service dependencies.

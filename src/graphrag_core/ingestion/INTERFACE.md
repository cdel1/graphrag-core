# `ingestion/` — INTERFACE (BB1)

**Protocols:** `DocumentParser`, `Chunker`, `EmbeddingModel`, `IngestionPipeline`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 38–68
**Default implementations:** [`PdfParser`, `DocxParser`, `MarkdownParser`, `TextParser`](parsers.py); [`TokenChunker`](chunker.py); `IngestionPipeline` orchestrator at [`pipeline.py`](pipeline.py)
**Vocabulary:** `Document`, `DocumentChunk`, `DataSource`, `ImportRun` — see `tessera/CONTEXT.md`

---

## `DocumentParser`

Converts raw file bytes into a structured `ParsedDocument` (sections + metadata).

### Interface

```python
async def parse(self, source: bytes, content_type: str) -> ParsedDocument: ...
```

### Contracts

- **Format dispatch happens *before* the parser.** A `DocumentParser` is registered per MIME type; the caller picks the right parser. Parsers don't sniff.
- **Returns a `ParsedDocument`** with `sections: list[TextSection]` and `metadata: DocumentMetadata`. Each section has heading, text, and optional page number.
- **Structure preservation matters.** Headings, table rows, list items should land as separate sections (or carry markup) so downstream chunking and front-matter extraction work correctly. A PDF parser that returns one long string loses page-attribution and breaks `EXTRACTED_FROM` provenance.
- **`metadata.sha256` is required and must be reproducible** for the same bytes (used for delta detection and duplicate suppression).

### Error modes

- Unsupported / malformed content → raise `ValueError` with a message including the content-type. Do not return empty `ParsedDocument` for malformed input — that breaks downstream debugging.
- Encoding issues → decode permissively (replace errors), log warning.

### Performance invariants

- O(file size). PDF parsing dominates; cache by SHA-256 if calling repeatedly.

---

## `Chunker`

Splits a `ParsedDocument` into `DocumentChunk`s suitable for extraction and embedding.

### Interface

```python
def chunk(self, doc: ParsedDocument, config: ChunkConfig) -> list[DocumentChunk]: ...
```

### Contracts

- **Synchronous.** No I/O.
- **`ChunkConfig` governs target size.** Default: `max_tokens=512, overlap=50`. Implementations should respect `max_tokens` as a hard ceiling and use `overlap` for sliding-window context.
- **Returns chunks with stable IDs.** Re-chunking the same `ParsedDocument` with the same `ChunkConfig` produces identical chunk IDs. (Used for delta-aware re-ingestion.)
- **`chunk.page` and `chunk.position` are required** when the source has page structure. They drive provenance display.
- **`chunk_type` defaults to `"text"`** — implementations may use `"table"`, `"heading"`, `"code"` for structurally distinct content.

### Error modes

- Empty `doc.sections` → returns `[]`, does not raise.
- Section longer than `max_tokens` → splits into multiple chunks with `overlap` honored.

### Performance invariants

- O(text length).
- No external dependencies (tokenizer can be local — `tiktoken` for the default `TokenChunker`).

### Default impl note

`TokenChunker` uses `tiktoken` tokens. **Known issue:** strips newlines that downstream `extract_front_matter` (Lacuna) relies on for regex matching. Fix is to switch front-matter extraction to operate on parsed chunks rather than raw bytes — see `feature_requirements.md` "Front matter extraction for binary formats."

---

## `EmbeddingModel`

Produces vector embeddings for chunks. **Protocol-only — no default implementation in graphrag-core yet.**

### Interface

```python
async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

### Contracts

- **Length preservation.** `len(embed(texts)) == len(texts)`. Order preserved.
- **Dimension consistency.** All vectors from one instance have identical dimensionality. Implementations should expose `.dimension` for downstream index configuration.
- **Batchable.** Single batch up to provider limit; caller handles pagination.

### Error modes

- Empty `texts` → returns `[]`, does not raise.
- Provider failure → propagates exception.

### Performance invariants

- Latency dominated by provider call; orchestrate concurrency at caller level.
- No graph I/O.

### Roadmap

`NomicEmbedding` named in the v0.1.0 spec but not yet implemented. A `MemoryEmbeddingModel` for tests is a candidate companion landing in the same release.

---

## `IngestionPipeline`

Orchestrates `parse → chunk → embed (optional) → store`. Single entry point for ingesting bytes into the graph.

### Interface

```python
async def ingest(
    self,
    source: bytes,
    content_type: str,
    config: ChunkConfig | None = None,
) -> list[DocumentChunk]: ...
```

### Contracts

- **Returns `list[DocumentChunk]`** (the chunks produced and stored). Callers can use these for downstream extraction.
- **Does *not* run extraction.** That's BB2. Pipeline composition is left to the caller (Lacuna's `LacunaIngestionPipeline` wires parse → chunk → extract → canonicalize → store).
- **Stable chunk IDs across re-ingestion of the same source.** Delta detection relies on this.
- **(v0.6.0) Writes the `Document` node and the `(:DocumentChunk)-[:CHUNKED_FROM]->(:Document)` edges** when given a `GraphStore`. The `Document` node carries `DocumentMetadata` properties verbatim (`title`, `source`, `doc_type`, `date`, `period`, `sha256`). This makes `GraphStore.get_audit_trail` reach the document level — the contract that BB7 temporal tools (`get_node_history`, `compare_periods`, `find_trend`) depend on.

> **Why BB1 owns this:** every consumer of graphrag-core that wants period-aware tooling needs document-level provenance. Pre-v0.6.0, document-node creation was the caller's responsibility (and Lacuna didn't do it, which silently broke `claim_period`). Pulling it into BB1 makes the provenance chain complete out of the box and removes a class of "tools return empty results" bugs.

### Error modes

- Unsupported `content_type` → `ValueError`.
- Parser / chunker exceptions propagate.

### Performance invariants

- Single-document ingestion is sequential; caller orchestrates fan-out for batches.

---

## Implementation skeleton (new parser)

```python
class MyFormatParser:
    async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
        if content_type != "application/x-myformat":
            raise ValueError(f"MyFormatParser does not handle {content_type}")
        # 1. Decode bytes -> structured representation.
        # 2. Extract sections (with heading/page).
        # 3. Build DocumentMetadata with sha256(source).
        return ParsedDocument(sections=[...], metadata=metadata)
```

Then add to the parser registry in the calling pipeline.

### Test checklist

- Parser: golden file → exact `ParsedDocument` (same SHA, same sections).
- Parser: malformed input → `ValueError`, not empty result.
- Chunker: idempotent — chunk twice, same chunk IDs.
- Chunker: chunks honor `max_tokens` ceiling.
- Chunker: page/position preserved for paginated source.
- IngestionPipeline: unsupported `content_type` → `ValueError`.
- `IngestionPipeline.ingest(...)` without `graph_store` returns chunks and makes no graph mutations (backward-compat).
- `IngestionPipeline.ingest(graph_store=store, import_run_id=R)` writes one `:Document` node with `DocumentMetadata` properties + one `CHUNKED_FROM` edge per chunk.
- Re-ingestion of the same source bytes (same `sha256`) is idempotent — one `:Document` node total.
- `quarter → period` fallback: if `metadata.period is None and metadata.quarter` is set, the Document node carries `period=quarter`.
- The persisted Document node properties do NOT contain `quarter` (it is stripped before write to avoid carrying the deprecated field forward).
- `IngestionPipeline.ingest(graph_store=store)` without `import_run_id` raises `ValueError`.

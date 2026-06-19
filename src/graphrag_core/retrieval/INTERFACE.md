# `retrieval/` — INTERFACE (BB10)

**Protocols:** `EmbeddingModel` (+ future `Reranker`)
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) — `EmbeddingModel` at line ~50
**Default implementations:** none shipped yet (first concrete impl committed per capability `BB10-01`; see [`tessera/docs/lacuna_capability_map.md`](../../../../../tessera/docs/lacuna_capability_map.md))
**Vocabulary:** `EmbeddingModel`, `Reranker` (planned) — see [`tessera/CONTEXT.md`](../../../../../tessera/CONTEXT.md) Band A
**Doctrine:** ADR-0039 (BB10 seat creation; moved from BB1 ingestion). Pairs with future Reranker per the multilingual-substrate workstream (`docs/research/2026-06-06-de-en-retrieval-asymmetry.md` §S4+S8).

---

## `EmbeddingModel`

Provider-agnostic interface for dense-vector encoding of text. Used by `IngestionPipeline` to attach embeddings to `DocumentChunk`s and by `SearchEngine` to embed queries for vector search.

### Interface

```python
async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

### Contracts

- **Length preservation.** `len(embed(texts)) == len(texts)`. Order preserved.
- **Dimension consistency.** All vectors from one instance have identical dimensionality. Implementations should expose `.dimension` for downstream index configuration.
- **Batchable.** Single batch up to provider limit; caller handles pagination.
- **Idempotent for the same `(texts, instance)` pair under deterministic provider settings.**

### Error modes

- Empty `texts` → returns `[]`, does not raise.
- Provider failure → propagates exception.

### Performance invariants

- Latency dominated by provider call; orchestrate concurrency at caller level.
- No graph I/O.

### Roadmap

`BB10-01` (first concrete impl, status `○ Designed`) wires a multilingual embedder (BGE-M3 vs multilingual-e5 — choice deferred to the implementing spec) into the ingestion pipeline + adds vector indexing in BB3. See capability map row `BB10-01` for current state.

---

## `Reranker` (planned, not yet shipped)

Cross-encoder reranker over a candidate list, paired with `EmbeddingModel` per the multilingual-substrate doctrine: a multilingual embedder without a multilingual reranker re-introduces DE/EN bias the first stage eliminated.

Protocol shape TBD; admission via the multilingual-substrate workstream (`BB10-02`, currently `· Aspirational`).

---

## Why BB10 exists as a separate seat (rationale)

`EmbeddingModel` was filed under BB1 ingestion through v0.12.0 — accurate for the *write* side, misleading for the *read* side, where `SearchEngine` (BB4) also depends on it. Microsoft GraphRAG extracts vectors as a peer package (`graphrag-vectors`); LlamaIndex bundles into the index abstraction; nano-graphrag plugs in via ABC. graphrag-core picks the Microsoft-style extraction (cross-cutting seat) so the embedder + reranker pairing — substrate-quality-critical per the multilingual research — sits in one mental load.

Per ADR-0039: a BB seat earns its existence by real consumer pressure. `EmbeddingModel` has cross-stage consumption (BB1 + BB4) and a designed concrete impl in flight — the seat is earned.

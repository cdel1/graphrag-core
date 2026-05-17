# `extraction/` — INTERFACE (BB2)

**Protocols:** `ExtractionEngine`, `ExtractionPromptBuilder`, `ExtractionPostProcessor`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 97–124
**Default implementations:** [`LLMExtractionEngine`](engine.py), `DefaultPromptBuilder` (in `engine.py`)
**Vocabulary:** Tier 1 ontology, three-tier discipline — see `tessera/CONTEXT.md`

---

## `ExtractionEngine`

Extracts Tier-1 entities and relationships from a list of `DocumentChunk`s, guided by an `OntologySchema`.

### Interface

```python
async def extract(
    self,
    chunks: list[DocumentChunk],
    schema: OntologySchema,
    import_run: ImportRun,
) -> ExtractionResult: ...
```

### Contracts

- **Tier discipline:** must only extract types the schema defines. Any node label not in `schema.node_types` is dropped silently or recorded as a validation issue (implementation choice; default `LLMExtractionEngine` drops + logs).
- **No Tier 2 / Tier 3 invention:** the engine produces Claims, Entities, Stakeholders — not `Topic`s, not curated `Risk`s, not `AcceptanceCriterion`s. Those are post-curation. A schema that *allows* the LLM to extract Topic-like types is misconfigured per current Lacuna policy (per `2026-04-21-multi-strategy-extraction-design.md` §"What gets deprecated").
- **Provenance is non-optional.** Every returned node must appear in `ExtractionResult.provenance` linking it back to at least one source chunk.
- **`import_run` is read-only.** The engine doesn't mutate the `ImportRun` passed in; callers may update `entities_extracted` after.
- The engine is **stateless** between calls. Per-document state lives in the caller's pipeline.

### Error modes

- LLM provider failure → propagates the underlying exception (caller chooses retry policy).
- Empty `chunks` → returns `ExtractionResult(nodes=[], relationships=[], provenance=[])`, does not raise.
- Schema with zero `node_types` → returns empty result and logs a warning.

### Performance invariants

- O(chunks) LLM calls in the naive implementation. Batching is implementation-specific.
- No graph I/O during extraction (the engine doesn't touch `GraphStore`).
- LLM call latency dominates; orchestrate concurrency at the caller level.

### Non-determinism

LLM outputs vary across calls. Implementations should use `temperature=0` for extraction. The Pydantic validation layer prunes outputs that don't match the schema — so non-determinism manifests as *recall* variance (which entities are extracted), not as *schema violations*.

---

## `ExtractionPromptBuilder`

Builds the system prompt for `ExtractionEngine`. Decoupled so domain-aware prompts (Lacuna's `ClaimsPromptBuilder`, `LacunaPromptBuilder`) can plug in without forking the engine.

### Interface

```python
def build_system_prompt(self, schema: OntologySchema) -> str: ...
```

### Contracts

- **Pure function of the schema.** Same schema in → same prompt out.
- **Synchronous.** No I/O; runs in the request path.
- Must produce a prompt that constrains the LLM to extract only types in the schema.

### Reference impls

- `DefaultPromptBuilder` (graphrag-core) — purely structural: lists types, properties, descriptions.
- `LacunaPromptBuilder` (Lacuna) — enriches with category context, few-shot examples, negative guidance. Used by the deprecated `schema_guided` strategy.
- `ClaimsPromptBuilder` (Lacuna) — narrowed to 3 types (Claim, Entity, Stakeholder). Used by the production `claims_based` strategy.

---

## `ExtractionPostProcessor`

Optional canonicalization layer between raw LLM output and graph storage. Pushes entity resolution, claim dedup, type normalization out of the engine.

### Interface

```python
async def process(
    self,
    result: ExtractionResult,
    existing_entities: list[GraphNode] | None = None,
) -> ExtractionResult: ...
```

### Contracts

- **Returns a *new* `ExtractionResult`** — never mutates the input.
- **Remaps `provenance` and `relationships` accordingly.** If two extracted nodes are canonicalized to the same node, all relationship endpoints and provenance links must be updated.
- **`existing_entities` is optional cross-document context.** If provided, the post-processor may resolve new extractions against the existing graph (entity registry pattern); if `None`, it operates within-document only.
- **Idempotent.** Calling `process(process(result))` produces the same result as `process(result)`.

### Error modes

- Empty `result` → returns empty result.
- Conflicting canonicalization (rare) → log + drop the lower-confidence node, never raise.

### Reference impl

`ClaimNormalizer` (Lacuna, `lacuna/extraction/normalizer.py`) — entity resolution via fuzzy + embedding matching + claim deduplication. Lives in Lacuna because the canonicalization rules are domain-influenced (initial-to-full-name matching tuned on construction stakeholder data).

---

## Implementation skeleton (custom strategy)

A new extraction strategy is `(PromptBuilder, optional PostProcessor, schema)`. Pattern:

```python
class MyPromptBuilder:
    def build_system_prompt(self, schema):
        return f"Extract only {[t.label for t in schema.node_types]}. ..."

class MyPostProcessor:
    async def process(self, result, existing_entities=None):
        # 1. Resolve duplicates within result.nodes.
        # 2. Resolve against existing_entities if provided.
        # 3. Remap provenance + relationship endpoints.
        # 4. Return new ExtractionResult.
        ...
```

Then register the strategy in Lacuna's `extraction/strategy.py::get_strategy()`.

### Test checklist

- Pure prompt builder: same schema → same prompt (golden-master test).
- Post-processor: idempotency (process(process(r)) == process(r)).
- Post-processor: provenance preserved across canonicalization.
- Engine: empty input → empty output, no exceptions.
- Engine: schema violations dropped (or surfaced as `validation_issues`) — never silently kept.

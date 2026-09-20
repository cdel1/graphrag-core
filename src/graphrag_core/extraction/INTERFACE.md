# `extraction/` — INTERFACE (BB2)

**Protocols:** `ExtractionEngine`, `ExtractionPromptBuilder`, `ExtractionPostProcessor`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 97–124
**Default implementations:** [`LLMExtractionEngine`](engine.py), `DefaultPromptBuilder` (in `engine.py`)
**Vocabulary:** Tier 1 ontology, three-tier discipline — see `tessera/CONTEXT.md`

---

## `ExtractionEngine`

Extracts Tier-1 entities and relationships from a list of `Chunk`s, guided by an `OntologySchema`.

### Interface

```python
async def extract(
    self,
    chunks: list[Chunk],
    schema: OntologySchema,
    import_run: ImportRun,
) -> ExtractionResult: ...
```

### Contracts

- **Tier discipline:** only types the schema defines reach `ExtractionResult.nodes` / `.relationships`. A node label not in `schema.node_types`, or a relationship the schema does not admit, is **not admitted — and not destroyed**: it comes back on `rejected_nodes` / `rejected_relationships` (see *Admission is not destruction* below).
- **No Tier 2 / Tier 3 invention:** the engine produces Claims, Entities, Stakeholders — not `Topic`s, not curated `Risk`s, not `AcceptanceCriterion`s. Those are post-curation. A schema that *allows* the LLM to extract Topic-like types is misconfigured per current Lacuna policy (per `2026-04-21-multi-strategy-extraction-design.md` §"What gets deprecated").
- **Provenance is non-optional.** Every *admitted* node must appear in `ExtractionResult.provenance` linking it back to at least one source chunk. A rejected node carries its source chunk on the rejection's `chunk_id` instead — `provenance` never references a node that is not in `nodes`.
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

LLM outputs vary across calls. Implementations should use `temperature=0` for extraction. Outputs that don't match the schema are held out of the admitted set — so non-determinism manifests as *recall* variance (which entities are extracted), not as *schema violations*.

---

## Admission is not destruction

The schema decides what enters the typed graph. It does **not** decide what the extraction run is allowed to remember. Everything the extractor emitted is recoverable from the `ExtractionResult`:

```python
emitted_nodes == result.nodes + [r.node for r in result.rejected_nodes]
emitted_rels  == result.relationships + [r.relationship for r in result.rejected_relationships]
```

A rejection preserves the emission verbatim — properties included, which is where an extractor puts the passage that justified the emission — plus the chunk it came from and **why** it was not admitted:

| `RejectionReason` | Fires when |
|---|---|
| `undeclared_node_label` | the node's label is not in `schema.node_types` |
| `undeclared_relationship_type` | the relationship's type is not in `schema.relationship_types` |
| `dangling_endpoint` | an endpoint id is not an admitted node (including one rejected for its label) |
| `endpoint_type_violation` | endpoint labels violate the type's `source_types` / `target_types` |

Checks run in that order per relationship and the first failure wins. Consumers decide severity — the engine's job is to report the cause, never to make the emission unreconstructable. A run that rejects everything is a misconfiguration signal, not an empty run.

`validate_extraction(nodes, rels, schema, chunk_id=None) -> SchemaAdmission` is the same split available standalone, for strategies that own their own dispatch.

Distinct from `SchemaViolation` (BB3): that reports on nodes **already persisted** in a `GraphStore` and references them by id; a rejection carries the un-persisted emission itself.

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
- Engine: schema violations never reach `nodes` / `relationships` — and never vanish: each comes back on `rejected_nodes` / `rejected_relationships`, typed by reason, with its properties (and so its warrant) intact.

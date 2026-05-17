# `registry/` — INTERFACE (BB6)

**Protocol:** `EntityRegistry`
**Source:** [`graphrag_core/interfaces.py`](../interfaces.py) lines 227–237
**Default implementations:** [`memory.py`](memory.py) (`InMemoryEntityRegistry`), [`matching.py`](matching.py) (fuzzy + embedding matcher utilities)
**Vocabulary:** `KnownEntity`, `RegistryMatch` — see `tessera/CONTEXT.md`

---

## `EntityRegistry`

Pre-loads known entities (from org charts, contract lists, prior imports) and matches new extractions against them to prevent duplicates. The first line of defense against entity-resolution drift.

### Interface

```python
async def register(self, entity: KnownEntity) -> str: ...
async def lookup(
    self,
    name: str,
    entity_type: str,
    match_strategy: str = "fuzzy",
) -> list[RegistryMatch]: ...
async def bulk_register(self, entities: list[KnownEntity]) -> int: ...
```

### Contracts

- **`register`** — adds one entity. Returns the canonical ID. Idempotent on `(name, entity_type)` — re-registering merges aliases rather than duplicating.
- **`lookup`** — returns ranked `RegistryMatch` candidates. **Empty list on no match — does not raise.** Sorted by `score` descending.
- **`bulk_register`** — convenience for batch loads. Returns count of successfully registered entities.
- **`match_strategy`** values:
  - `"exact"` — case-insensitive exact match on name or aliases. Score is 1.0 or absent.
  - `"fuzzy"` — Levenshtein / Jaccard / token-set ratio. Score in [0, 1].
  - `"embedding"` — semantic similarity. Requires registry to have indexed embeddings. Score in [0, 1].
- **`RegistryMatch.match_method`** records which strategy yielded the match.

### Idempotency

- `register` is idempotent on `(name, entity_type)`. Aliases passed on re-registration are *merged* with existing aliases, not replaced.
- `bulk_register` applies the same rule per-entity.

### Error modes

- Unknown `match_strategy` → `ValueError`.
- Empty `name` → returns `[]` (no match), does not raise.
- Embedding strategy without an embedding index → raises `RuntimeError` with a clear message.

### Performance invariants

- `register`: O(1) amortized.
- `lookup` with `"exact"`: O(1) with hash index.
- `lookup` with `"fuzzy"`: O(registry size) naive; O(log n) with blocking / pre-filter on entity_type.
- `lookup` with `"embedding"`: O(log n) with HNSW; O(n) linear.

### Reference impl

`InMemoryEntityRegistry` — dict-backed, fuzzy match via Levenshtein. No embedding support (returns `[]` for that strategy). Tests + small-scale demos only.

**Roadmap:** Neo4j-backed entity registry promised in v0.1.0 spec — not yet implemented. Production use cases needing >10k entities should implement against the Protocol.

### Lacuna usage

`ClaimNormalizer` (`lacuna/extraction/normalizer.py`) uses an `EntityRegistry` for cross-document entity resolution during canonicalization. The registry is rebuilt per-pipeline-run from `graph_store.list_nodes()` filtered to `Entity` and `Stakeholder` labels.

---

## Implementation skeleton

```python
class MyEntityRegistry:
    def __init__(self):
        self._entities: dict[str, KnownEntity] = {}  # id -> entity
        self._name_index: dict[tuple[str, str], str] = {}  # (name_lower, type) -> id

    async def register(self, entity: KnownEntity) -> str:
        key = (entity.name.lower(), entity.entity_type)
        if key in self._name_index:
            existing_id = self._name_index[key]
            # Merge aliases instead of duplicating.
            self._entities[existing_id].aliases = list(
                set(self._entities[existing_id].aliases) | set(entity.aliases)
            )
            return existing_id
        new_id = generate_id(entity)
        self._entities[new_id] = entity
        self._name_index[key] = new_id
        return new_id

    async def lookup(self, name, entity_type, match_strategy="fuzzy"):
        if match_strategy == "exact":
            key = (name.lower(), entity_type)
            if key in self._name_index:
                return [RegistryMatch(entity_id=self._name_index[key], name=name, score=1.0, match_method="exact")]
            return []
        elif match_strategy == "fuzzy":
            # Iterate over self._entities filtered by entity_type, score with Levenshtein.
            ...
        elif match_strategy == "embedding":
            raise RuntimeError("Embedding strategy requires indexed embeddings; not supported in this impl")
        else:
            raise ValueError(f"Unknown match_strategy: {match_strategy}")

    async def bulk_register(self, entities):
        return sum([1 for e in entities if await self.register(e)])
```

### Test checklist

- `register` idempotent on same name+type (no duplicate, aliases merged).
- `lookup("exact")` returns score 1.0 on exact match, `[]` otherwise.
- `lookup("fuzzy")` returns descending scores.
- `lookup` with empty name → `[]`.
- Unknown `match_strategy` → `ValueError`.
- `bulk_register` count matches input length on first call; matches 0 on re-register.

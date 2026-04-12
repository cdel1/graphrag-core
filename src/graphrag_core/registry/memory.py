"""In-memory EntityRegistry implementation."""

from __future__ import annotations

from graphrag_core.models import KnownEntity, RegistryMatch
from graphrag_core.registry.matching import fuzzy_score, normalize_name

_FUZZY_THRESHOLD = 0.7


class InMemoryEntityRegistry:
    """Dict-based EntityRegistry for unit tests and lightweight usage."""

    def __init__(self) -> None:
        self._entities: dict[str, KnownEntity] = {}

    def _make_id(self, entity: KnownEntity) -> str:
        return f"{entity.entity_type}-{normalize_name(entity.name)}"

    async def register(self, entity: KnownEntity) -> str:
        entity_id = self._make_id(entity)
        existing = self._entities.get(entity_id)
        if existing:
            merged_aliases = list(set(existing.aliases + entity.aliases))
            self._entities[entity_id] = KnownEntity(
                name=existing.name,
                entity_type=existing.entity_type,
                aliases=merged_aliases,
                properties={**existing.properties, **entity.properties},
            )
        else:
            self._entities[entity_id] = entity
        return entity_id

    async def bulk_register(self, entities: list[KnownEntity]) -> int:
        count = 0
        for entity in entities:
            entity_id = self._make_id(entity)
            was_new = entity_id not in self._entities
            await self.register(entity)
            if was_new:
                count += 1
        return count

    async def lookup(
        self, name: str, entity_type: str, match_strategy: str = "fuzzy"
    ) -> list[RegistryMatch]:
        if match_strategy == "embedding":
            return []

        matches: list[RegistryMatch] = []

        for entity_id, entity in self._entities.items():
            if entity.entity_type != entity_type:
                continue

            if match_strategy == "exact":
                all_names = [entity.name] + entity.aliases
                if name in all_names:
                    matches.append(RegistryMatch(
                        entity_id=entity_id,
                        name=entity.name,
                        score=1.0,
                        match_method="exact",
                    ))
            elif match_strategy == "fuzzy":
                all_names = [entity.name] + entity.aliases
                best_score = 0.0
                for candidate in all_names:
                    score = fuzzy_score(name, candidate)
                    best_score = max(best_score, score)
                if best_score >= _FUZZY_THRESHOLD:
                    matches.append(RegistryMatch(
                        entity_id=entity_id,
                        name=entity.name,
                        score=best_score,
                        match_method="fuzzy",
                    ))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

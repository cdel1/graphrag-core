"""Tests for InMemoryEntityRegistry."""

from __future__ import annotations

import pytest

from graphrag_core.models import KnownEntity


class TestRegister:
    @pytest.mark.asyncio
    async def test_stores_and_returns_id(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        entity = KnownEntity(name="Acme Corp", entity_type="Company")
        entity_id = await registry.register(entity)
        assert entity_id == "Company-acme corp"

    @pytest.mark.asyncio
    async def test_merges_aliases_on_duplicate(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        e1 = KnownEntity(name="Acme Corp", entity_type="Company", aliases=["Acme"])
        e2 = KnownEntity(name="Acme Corp", entity_type="Company", aliases=["ACME Inc"])

        await registry.register(e1)
        await registry.register(e2)

        matches = await registry.lookup("Acme Corp", "Company", match_strategy="exact")
        assert len(matches) == 1


class TestBulkRegister:
    @pytest.mark.asyncio
    async def test_returns_count_of_newly_registered(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        entities = [
            KnownEntity(name="Acme Corp", entity_type="Company"),
            KnownEntity(name="Globex", entity_type="Company"),
            KnownEntity(name="Acme Corp", entity_type="Company"),
        ]
        count = await registry.bulk_register(entities)
        assert count == 2


class TestLookupExact:
    @pytest.mark.asyncio
    async def test_matches_on_name(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Acme Corp", "Company", match_strategy="exact")
        assert len(matches) == 1
        assert matches[0].name == "Acme Corp"
        assert matches[0].score == 1.0
        assert matches[0].match_method == "exact"

    @pytest.mark.asyncio
    async def test_matches_on_alias(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(
            KnownEntity(name="Acme Corp", entity_type="Company", aliases=["Acme", "ACME Inc"])
        )

        matches = await registry.lookup("Acme", "Company", match_strategy="exact")
        assert len(matches) == 1
        assert matches[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Globex", "Company", match_strategy="exact")
        assert matches == []

    @pytest.mark.asyncio
    async def test_filters_by_entity_type(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Alice", entity_type="Person"))
        await registry.register(KnownEntity(name="Alice", entity_type="Company"))

        matches = await registry.lookup("Alice", "Person", match_strategy="exact")
        assert len(matches) == 1
        assert matches[0].entity_id.startswith("Person-")


class TestLookupFuzzy:
    @pytest.mark.asyncio
    async def test_catches_reorderings(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Alice Smith", entity_type="Person"))

        matches = await registry.lookup("Smith, Alice", "Person", match_strategy="fuzzy")
        assert len(matches) >= 1
        assert matches[0].score >= 0.7
        assert matches[0].match_method == "fuzzy"

    @pytest.mark.asyncio
    async def test_catches_case_differences(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="ACME Corporation", entity_type="Company"))

        matches = await registry.lookup("acme corporation", "Company", match_strategy="fuzzy")
        assert len(matches) >= 1
        assert matches[0].score >= 0.9

    @pytest.mark.asyncio
    async def test_below_threshold_returns_empty(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Zebra Industries", "Company", match_strategy="fuzzy")
        assert matches == []


class TestLookupEmbedding:
    @pytest.mark.asyncio
    async def test_returns_empty(self):
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        matches = await registry.lookup("Acme", "Company", match_strategy="embedding")
        assert matches == []


class TestEntityRegistryProtocol:
    def test_satisfies_entity_registry_protocol(self):
        from graphrag_core.interfaces import EntityRegistry
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        assert isinstance(registry, EntityRegistry)

import pytest

from graphrag_core.eval.tier_one import (
    NoOrphanIntelligenceCheck,
    ProvenanceCompletenessCheck,
    SchemaConformanceCheck,
)
from graphrag_core.graph import InMemoryGraphStore
from graphrag_core.models import GraphNode, GraphRelationship


async def _seed(store: InMemoryGraphStore, nodes: list[GraphNode], rels: list[GraphRelationship]) -> None:
    for n in nodes:
        await store.merge_node(n, import_run_id="test-run")
    for r in rels:
        await store.merge_relationship(r, import_run_id="test-run")


@pytest.mark.asyncio
async def test_provenance_completeness_passes_when_every_claim_has_sourced_from() -> None:
    store = InMemoryGraphStore()
    await _seed(
        store,
        nodes=[
            GraphNode(id="c1", label="Claim", properties={}),
            GraphNode(id="chunk1", label="Chunk", properties={}),
        ],
        rels=[GraphRelationship(source_id="c1", target_id="chunk1", type="SOURCED_FROM")],
    )
    check = ProvenanceCompletenessCheck()
    assert await check.check(store, manifest=None) == []


@pytest.mark.asyncio
async def test_provenance_completeness_flags_claim_without_sourced_from() -> None:
    store = InMemoryGraphStore()
    await _seed(
        store,
        nodes=[GraphNode(id="c1", label="Claim", properties={})],
        rels=[],
    )
    check = ProvenanceCompletenessCheck()
    violations = await check.check(store, manifest=None)
    assert len(violations) == 1
    assert violations[0].check == "provenance_completeness"


@pytest.mark.asyncio
async def test_no_orphan_intelligence_flags_topic_with_no_claim() -> None:
    store = InMemoryGraphStore()
    await _seed(
        store,
        nodes=[GraphNode(id="t1", label="Topic", properties={})],
        rels=[],
    )
    check = NoOrphanIntelligenceCheck()
    violations = await check.check(store, manifest=None)
    assert any(v.check == "no_orphan_intelligence" for v in violations)


@pytest.mark.asyncio
async def test_no_orphan_intelligence_passes_when_topic_has_claim_edge() -> None:
    store = InMemoryGraphStore()
    await _seed(
        store,
        nodes=[
            GraphNode(id="t1", label="Topic", properties={}),
            GraphNode(id="c1", label="Claim", properties={}),
        ],
        rels=[GraphRelationship(source_id="t1", target_id="c1", type="GROUNDED_IN")],
    )
    check = NoOrphanIntelligenceCheck()
    assert await check.check(store, manifest=None) == []


@pytest.mark.asyncio
async def test_schema_conformance_flags_unknown_node_label() -> None:
    store = InMemoryGraphStore()
    await _seed(
        store,
        nodes=[GraphNode(id="x1", label="Mystery", properties={})],
        rels=[],
    )
    check = SchemaConformanceCheck(allowed_labels={"Claim", "Entity", "Document", "Chunk", "Stakeholder"})
    violations = await check.check(store, manifest=None)
    assert any(v.check == "schema_conformance" for v in violations)


@pytest.mark.asyncio
async def test_schema_conformance_passes_when_all_labels_allowed() -> None:
    store = InMemoryGraphStore()
    await _seed(
        store,
        nodes=[GraphNode(id="c1", label="Claim", properties={})],
        rels=[],
    )
    check = SchemaConformanceCheck(allowed_labels={"Claim"})
    assert await check.check(store, manifest=None) == []

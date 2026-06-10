import pytest

from graphrag_core.eval.benchmarks.docred.manifest import (
    DocREDDocument,
    DocREDGoldRelation,
    DocREDManifest,
)
from graphrag_core.eval.benchmarks.docred.scorer import DocREDScorer
from graphrag_core.graph import InMemoryGraphStore
from graphrag_core.models import GraphNode, GraphRelationship


def _manifest_with_relations(relations: list[DocREDGoldRelation]) -> DocREDManifest:
    return DocREDManifest(
        version="docred@test",
        slice_axes=["relation_type"],
        token_budget=100_000,
        model_pin={},
        documents=[
            DocREDDocument(id="d1", title="d1", sentences=[["x"]])
        ],
        gold_relations=relations,
    )


async def _seed_entities_and_relations(store, ents: list[tuple[str, str]], rels: list[tuple[str, str, str]]) -> None:
    for eid, name in ents:
        await store.merge_node(GraphNode(id=eid, label="Entity", properties={"name": name}), import_run_id="t")
    for src, tgt, rtype in rels:
        await store.merge_relationship(GraphRelationship(source_id=src, target_id=tgt, type=rtype), import_run_id="t")


@pytest.mark.asyncio
async def test_scorer_perfect_when_graph_contains_all_gold_relations() -> None:
    store = InMemoryGraphStore()
    await _seed_entities_and_relations(
        store,
        ents=[("d1:Alice", "Alice"), ("d1:Bob", "Bob")],
        rels=[("d1:Alice", "d1:Bob", "P27")],
    )
    manifest = _manifest_with_relations([
        DocREDGoldRelation(head_id="d1:Alice", tail_id="d1:Bob", relation_type="P27", evidence_sentence_ids=[0]),
    ])
    scores = await DocREDScorer().score(store, manifest)
    assert "relation_type=P27" in scores
    assert scores["relation_type=P27"].precision == 1.0
    assert scores["relation_type=P27"].recall == 1.0


@pytest.mark.asyncio
async def test_scorer_zero_recall_when_graph_empty() -> None:
    store = InMemoryGraphStore()
    manifest = _manifest_with_relations([
        DocREDGoldRelation(head_id="d1:Alice", tail_id="d1:Bob", relation_type="P27", evidence_sentence_ids=[0]),
    ])
    scores = await DocREDScorer().score(store, manifest)
    assert scores["relation_type=P27"].recall == 0.0


@pytest.mark.asyncio
async def test_scorer_one_slice_per_relation_type() -> None:
    store = InMemoryGraphStore()
    manifest = _manifest_with_relations([
        DocREDGoldRelation(head_id="d1:Alice", tail_id="d1:Bob", relation_type="P27", evidence_sentence_ids=[0]),
        DocREDGoldRelation(head_id="d1:Carol", tail_id="d1:Bob", relation_type="P17", evidence_sentence_ids=[1]),
    ])
    scores = await DocREDScorer().score(store, manifest)
    assert "relation_type=P27" in scores
    assert "relation_type=P17" in scores

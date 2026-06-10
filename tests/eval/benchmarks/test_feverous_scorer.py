import pytest

from graphrag_core.eval.benchmarks.feverous.manifest import (
    FEVEROUSGoldClaim,
    FEVEROUSManifest,
)
from graphrag_core.eval.benchmarks.feverous.scorer import FEVEROUSScorer
from graphrag_core.graph import InMemoryGraphStore
from graphrag_core.models import GraphNode


def _manifest(claims: list[FEVEROUSGoldClaim]) -> FEVEROUSManifest:
    return FEVEROUSManifest(
        version="feverous@test",
        slice_axes=["label", "challenge"],
        token_budget=100_000,
        model_pin={},
        gold_claims=claims,
    )


async def _seed_claim_node(store, claim_id: str, predicted_label: str) -> None:
    await store.merge_node(
        GraphNode(label="Claim", id=claim_id, properties={"verdict": predicted_label}),
        import_run_id="t",
    )


@pytest.mark.asyncio
async def test_scorer_perfect_when_predictions_match_gold() -> None:
    store = InMemoryGraphStore()
    await _seed_claim_node(store, "c1", "SUPPORTS")
    manifest = _manifest([
        FEVEROUSGoldClaim(id="c1", label="SUPPORTS", claim_text="x", challenge="Other", evidence_ids=[])
    ])
    scores = await FEVEROUSScorer().score(store, manifest)
    label_slice = next(k for k in scores if "label=SUPPORTS" in k)
    assert scores[label_slice].precision == 1.0
    assert scores[label_slice].recall == 1.0


@pytest.mark.asyncio
async def test_scorer_zero_recall_when_no_claim_nodes() -> None:
    store = InMemoryGraphStore()
    manifest = _manifest([
        FEVEROUSGoldClaim(id="c1", label="REFUTES", claim_text="x", challenge="Other", evidence_ids=[])
    ])
    scores = await FEVEROUSScorer().score(store, manifest)
    label_slice = next(k for k in scores if "label=REFUTES" in k)
    assert scores[label_slice].recall == 0.0


@pytest.mark.asyncio
async def test_scorer_slices_by_label_and_challenge() -> None:
    store = InMemoryGraphStore()
    manifest = _manifest([
        FEVEROUSGoldClaim(id="c1", label="SUPPORTS", claim_text="x", challenge="Multi-hop Reasoning", evidence_ids=[]),
        FEVEROUSGoldClaim(id="c2", label="REFUTES", claim_text="y", challenge="Numerical Reasoning", evidence_ids=[]),
    ])
    scores = await FEVEROUSScorer().score(store, manifest)
    assert any("label=SUPPORTS" in k for k in scores)
    assert any("label=REFUTES" in k for k in scores)
    assert any("challenge=Multi-hop Reasoning" in k for k in scores)
    assert any("challenge=Numerical Reasoning" in k for k in scores)

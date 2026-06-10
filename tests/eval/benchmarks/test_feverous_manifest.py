from pathlib import Path

import pytest

from graphrag_core.eval.benchmarks.feverous.manifest import (
    FEVEROUSManifest,
    FEVEROUSManifestLoader,
)


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).resolve().parents[3] / "eval/fixtures/feverous/sample.jsonl"


def test_loader_returns_manifest_with_version_and_slice_axes(fixture_path: Path) -> None:
    m = FEVEROUSManifestLoader().load(fixture_path)
    assert isinstance(m, FEVEROUSManifest)
    assert m.version.startswith("feverous@")
    assert m.slice_axes == ["label", "challenge"]
    assert m.token_budget > 0


def test_manifest_carries_gold_claims(fixture_path: Path) -> None:
    m = FEVEROUSManifestLoader().load(fixture_path)
    assert len(m.gold_claims) > 0
    c = m.gold_claims[0]
    assert c.id and c.label and c.claim_text and c.challenge


def test_manifest_covers_all_3_labels_and_3plus_challenges(fixture_path: Path) -> None:
    m = FEVEROUSManifestLoader().load(fixture_path)
    labels = {c.label for c in m.gold_claims}
    challenges = {c.challenge for c in m.gold_claims}
    assert labels == {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
    assert len(challenges) >= 3

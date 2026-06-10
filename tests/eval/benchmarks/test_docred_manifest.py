from pathlib import Path

import pytest

from graphrag_core.eval.benchmarks.docred.manifest import (
    DocREDManifest,
    DocREDManifestLoader,
)


@pytest.fixture
def fixture_path() -> Path:
    # tests/eval/benchmarks/test_docred_manifest.py -> repo root via 4 parents
    return Path(__file__).resolve().parents[3] / "eval/fixtures/docred/sample.jsonl"


def test_loader_returns_manifest_with_version_and_slice_axes(fixture_path: Path) -> None:
    loader = DocREDManifestLoader()
    manifest = loader.load(fixture_path)
    assert isinstance(manifest, DocREDManifest)
    assert manifest.version.startswith("docred@")
    assert manifest.slice_axes == ["relation_type"]
    assert manifest.token_budget > 0


def test_manifest_carries_documents_and_gold_relations(fixture_path: Path) -> None:
    loader = DocREDManifestLoader()
    manifest = loader.load(fixture_path)
    assert len(manifest.documents) > 0
    assert len(manifest.gold_relations) > 0
    rel = manifest.gold_relations[0]
    assert rel.head_id and rel.tail_id and rel.relation_type and rel.evidence_sentence_ids


def test_manifest_covers_at_least_6_relation_types(fixture_path: Path) -> None:
    loader = DocREDManifestLoader()
    manifest = loader.load(fixture_path)
    rel_types = {r.relation_type for r in manifest.gold_relations}
    assert len(rel_types) >= 6, f"Expected ≥6 relation types, got {len(rel_types)}: {rel_types}"

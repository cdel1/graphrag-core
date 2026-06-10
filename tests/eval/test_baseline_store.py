from pathlib import Path

import pytest

from graphrag_core.eval.baseline_store import JSONFileBaselineStore
from graphrag_core.eval.models import BaselineFile, SliceGateRule, SliceScore


@pytest.fixture
def baseline_fixture() -> BaselineFile:
    return BaselineFile(
        harness_version="0.1.0",
        manifest_version="docred@2026-06-10",
        created_at="2026-06-10T00:00:00Z",
        created_by="rebaseline-abc1234",
        model_pin={"extraction": "gpt-4o-mini", "seed": 42},
        slice_scores={"relation=P_director": SliceScore(precision=0.9, recall=0.8, n=12)},
        slice_gates={"relation=P_director": SliceGateRule(precision_min=0.85, recall_min=0.75)},
    )


def test_write_creates_sharded_file(tmp_path: Path, baseline_fixture: BaselineFile) -> None:
    store = JSONFileBaselineStore(root=tmp_path)
    path = store.write(baseline_fixture)
    assert path == tmp_path / "docred@2026-06-10" / "0.1.0.json"
    assert path.exists()


def test_read_returns_none_when_absent(tmp_path: Path) -> None:
    store = JSONFileBaselineStore(root=tmp_path)
    assert store.read("docred@2026-06-10", "0.1.0") is None


def test_write_then_read_roundtrip(
    tmp_path: Path, baseline_fixture: BaselineFile
) -> None:
    store = JSONFileBaselineStore(root=tmp_path)
    store.write(baseline_fixture)
    got = store.read(baseline_fixture.manifest_version, baseline_fixture.harness_version)
    assert got == baseline_fixture


def test_list_versions_returns_sorted(tmp_path: Path, baseline_fixture: BaselineFile) -> None:
    store = JSONFileBaselineStore(root=tmp_path)
    store.write(baseline_fixture)
    store.write(baseline_fixture.model_copy(update={"harness_version": "0.2.0"}))
    versions = store.list_versions(baseline_fixture.manifest_version)
    assert versions == ["0.1.0", "0.2.0"]

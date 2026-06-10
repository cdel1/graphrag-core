from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from graphrag_core.eval.baseline_store import JSONFileBaselineStore
from graphrag_core.eval.harness import EvalHarness, ManifestScorerPair
from graphrag_core.eval.models import RunReport, SliceKey, SliceScore, Violation
from graphrag_core.eval.slice_gate import DefaultSliceGate
from graphrag_core.graph import InMemoryGraphStore


@dataclass
class _FakeManifest:
    version: str = "fake@2026-06-10"
    slice_axes: list[str] = None
    token_budget: int = 100_000
    model_pin: dict[str, Any] = None

    def __post_init__(self):
        if self.slice_axes is None:
            self.slice_axes = ["k"]
        if self.model_pin is None:
            self.model_pin = {}


class _FakeRunner:
    async def run(self, corpus_path: Path, graph_store) -> None:
        pass  # no-op; tests don't need a populated graph


class _FakeScorerOneSlice:
    async def score(self, graph_store, manifest) -> dict[SliceKey, SliceScore]:
        return {"k": SliceScore(precision=0.9, recall=0.85, n=10)}


class _FakeScorerEmpty:
    async def score(self, graph_store, manifest) -> dict[SliceKey, SliceScore]:
        return {}


class _PassingTierOneCheck:
    async def check(self, graph_store, manifest) -> list[Violation]:
        return []


class _FailingTierOneCheck:
    async def check(self, graph_store, manifest) -> list[Violation]:
        return [Violation(check="fake_invariant", message="boom")]


def _pair(checks, scorer) -> ManifestScorerPair:
    return ManifestScorerPair(
        manifest=_FakeManifest(),
        corpus_path=Path("/nonexistent"),
        pipeline_runner=_FakeRunner(),
        tier_one_checks=checks,
        scorer=scorer,
    )


@pytest.mark.asyncio
async def test_harness_runs_pipeline_then_tier_one_then_scorer_then_gate(tmp_path: Path) -> None:
    harness = EvalHarness(
        pair=_pair(checks=[_PassingTierOneCheck()], scorer=_FakeScorerOneSlice()),
        baseline_store=JSONFileBaselineStore(root=tmp_path),
        slice_gate=DefaultSliceGate(),
        graph_store_factory=lambda: InMemoryGraphStore(),
        harness_version="0.1.0",
    )
    report: RunReport = await harness.run()
    assert isinstance(report, RunReport)
    assert report.passed is True
    assert report.harness_version == "0.1.0"
    assert report.tier_one_violations == []
    assert report.slice_scores == {"k": SliceScore(precision=0.9, recall=0.85, n=10)}


@pytest.mark.asyncio
async def test_harness_fail_fast_on_tier_one_violation(tmp_path: Path) -> None:
    harness = EvalHarness(
        pair=_pair(checks=[_FailingTierOneCheck()], scorer=_FakeScorerOneSlice()),
        baseline_store=JSONFileBaselineStore(root=tmp_path),
        slice_gate=DefaultSliceGate(),
        graph_store_factory=lambda: InMemoryGraphStore(),
        harness_version="0.1.0",
    )
    report = await harness.run()
    assert report.passed is False
    assert len(report.tier_one_violations) == 1
    assert report.slice_scores == {}, "T2 scoring should NOT run after T1 violation"
    assert report.gate_failures == []

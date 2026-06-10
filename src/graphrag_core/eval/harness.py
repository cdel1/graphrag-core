"""EvalHarness orchestrator — wires Pair + BaselineStore + SliceGate together.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.2 + §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from graphrag_core.interfaces import GraphStore

from graphrag_core.eval.models import RunReport, SliceKey, SliceScore
from graphrag_core.eval.protocols import (
    BaselineStore,
    Manifest,
    PipelineRunner,
    Scorer,
    SliceGate,
    TierOneCheck,
)


@dataclass
class ManifestScorerPair:
    manifest: Manifest
    corpus_path: Path
    pipeline_runner: PipelineRunner
    tier_one_checks: list[TierOneCheck]
    scorer: Scorer


class EvalHarness:
    def __init__(
        self,
        pair: ManifestScorerPair,
        baseline_store: BaselineStore,
        slice_gate: SliceGate,
        graph_store_factory: Callable[[], GraphStore],
        harness_version: str,
    ) -> None:
        self.pair = pair
        self.baseline_store = baseline_store
        self.slice_gate = slice_gate
        self.graph_store_factory = graph_store_factory
        self.harness_version = harness_version

    async def run(self) -> RunReport:
        graph_store = self.graph_store_factory()
        await self.pair.pipeline_runner.run(self.pair.corpus_path, graph_store)

        violations = []
        for check in self.pair.tier_one_checks:
            violations.extend(await check.check(graph_store, self.pair.manifest))

        if violations:
            return RunReport(
                manifest_version=self.pair.manifest.version,
                harness_version=self.harness_version,
                tier_one_violations=violations,
                slice_scores={},
                gate_failures=[],
                passed=False,
            )

        scores: dict[SliceKey, SliceScore] = await self.pair.scorer.score(
            graph_store, self.pair.manifest
        )

        baseline = self.baseline_store.read(
            self.pair.manifest.version, self.harness_version
        )
        gate_failures = self.slice_gate.evaluate(scores, baseline)

        return RunReport(
            manifest_version=self.pair.manifest.version,
            harness_version=self.harness_version,
            tier_one_violations=[],
            slice_scores=scores,
            gate_failures=gate_failures,
            passed=(len(gate_failures) == 0),
        )

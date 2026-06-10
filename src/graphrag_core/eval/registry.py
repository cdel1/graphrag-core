"""Pair registry — pairs register themselves into the harness via entry points
or programmatic registration.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.1.
"""

from __future__ import annotations

import datetime as dt
from importlib import metadata
from pathlib import Path
from typing import Callable

from graphrag_core.eval.baseline_store import JSONFileBaselineStore
from graphrag_core.eval.harness import EvalHarness, ManifestScorerPair
from graphrag_core.eval.models import BaselineFile, RunReport
from graphrag_core.eval.slice_gate import DefaultSliceGate
from graphrag_core.graph import InMemoryGraphStore

_PAIRS: dict[str, Callable[[], ManifestScorerPair]] = {}


def register_pair(name: str) -> Callable:
    def decorator(factory: Callable[[], ManifestScorerPair]) -> Callable:
        _PAIRS[name] = factory
        return factory
    return decorator


def get_pair(name: str) -> ManifestScorerPair:
    _discover_entry_points()
    if name not in _PAIRS:
        raise KeyError(f"Unknown pair: {name}. Registered: {list_registered()}")
    return _PAIRS[name]()


def list_registered() -> list[str]:
    _discover_entry_points()
    return sorted(_PAIRS)


def _discover_entry_points() -> None:
    for ep in metadata.entry_points(group="graphrag_core.eval_pairs"):
        ep.load()  # importing triggers @register_pair


def build_default_components(
    pair: ManifestScorerPair,
    baseline_root: Path | None = None,
) -> tuple[EvalHarness, JSONFileBaselineStore]:
    baseline_store = JSONFileBaselineStore(root=baseline_root or Path("eval/baselines"))
    harness = EvalHarness(
        pair=pair,
        baseline_store=baseline_store,
        slice_gate=DefaultSliceGate(),
        graph_store_factory=lambda: InMemoryGraphStore(),
        harness_version="0.1.0",
    )
    return harness, baseline_store


def persist_as_baseline(
    report: RunReport,
    pair: ManifestScorerPair,
    baseline_store: JSONFileBaselineStore,
) -> Path:
    baseline = BaselineFile(
        harness_version=report.harness_version,
        manifest_version=pair.manifest.version,
        created_at=dt.datetime.utcnow().isoformat() + "Z",
        created_by="rebaseline",
        model_pin=pair.manifest.model_pin,
        slice_scores=report.slice_scores,
        slice_gates={},
    )
    return baseline_store.write(baseline)

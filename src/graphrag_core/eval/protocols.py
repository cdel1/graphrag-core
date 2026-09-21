"""Protocol surfaces for the eval harness.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from graphrag_core.interfaces import GraphStore

from graphrag_core.eval.models import (
    BaselineFile,
    GateFailure,
    SliceGateRule,
    SliceKey,
    SliceScore,
    Violation,
)


@runtime_checkable
class Manifest(Protocol):
    # Read-only properties, not plain attributes: the harness only reads a
    # manifest, and plain Protocol attributes demand settability — which
    # rejects frozen/immutable implementations (e.g. a frozen Pydantic model).
    @property
    def version(self) -> str: ...

    @property
    def slice_axes(self) -> list[str]: ...

    @property
    def token_budget(self) -> int: ...

    @property
    def model_pin(self) -> dict[str, Any]: ...


@runtime_checkable
class ManifestLoader(Protocol):
    def load(self, path: Path) -> Manifest: ...


@runtime_checkable
class PipelineRunner(Protocol):
    """Runs a corpus through ingest+detect and yields the produced graph."""

    async def run(self, corpus_path: Path, graph_store: GraphStore) -> None: ...


@runtime_checkable
class TierOneCheck(Protocol):
    async def check(
        self, graph_store: GraphStore, manifest: Manifest
    ) -> list[Violation]: ...


@runtime_checkable
class Scorer(Protocol):
    async def score(
        self, graph_store: GraphStore, manifest: Manifest
    ) -> dict[SliceKey, SliceScore]: ...


@runtime_checkable
class BaselineStore(Protocol):
    def read(
        self, manifest_version: str, harness_version: str
    ) -> BaselineFile | None: ...
    def write(self, baseline: BaselineFile) -> Path: ...
    def list_versions(self, manifest_version: str) -> list[str]: ...


@runtime_checkable
class SliceGate(Protocol):
    def evaluate(
        self,
        current: dict[SliceKey, SliceScore],
        baseline: BaselineFile | None,
    ) -> list[GateFailure]: ...

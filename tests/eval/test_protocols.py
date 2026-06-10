from typing import Protocol

from graphrag_core.eval.protocols import (
    BaselineStore,
    Manifest,
    ManifestLoader,
    PipelineRunner,
    Scorer,
    SliceGate,
    TierOneCheck,
)


def test_each_eval_protocol_is_a_runtime_checkable_protocol() -> None:
    for p in (Manifest, ManifestLoader, PipelineRunner, Scorer, TierOneCheck, BaselineStore, SliceGate):
        assert issubclass(p, Protocol)

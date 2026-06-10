from graphrag_core.eval.models import BaselineFile, SliceGateRule, SliceScore
from graphrag_core.eval.slice_gate import DefaultSliceGate


def _baseline(scores: dict, gates: dict) -> BaselineFile:
    return BaselineFile(
        harness_version="0.1.0",
        manifest_version="m@v",
        created_at="2026-06-10T00:00:00Z",
        created_by="rebaseline-abc",
        model_pin={},
        slice_scores=scores,
        slice_gates=gates,
    )


def test_no_baseline_emits_no_failures() -> None:
    gate = DefaultSliceGate()
    current = {"k": SliceScore(precision=0.5, recall=0.5, n=10)}
    assert gate.evaluate(current, baseline=None) == []


def test_passing_slice_emits_no_failure() -> None:
    gate = DefaultSliceGate()
    current = {"k": SliceScore(precision=0.9, recall=0.85, n=10)}
    baseline = _baseline(
        scores=current,
        gates={"k": SliceGateRule(precision_min=0.85, recall_min=0.80)},
    )
    assert gate.evaluate(current, baseline) == []


def test_regressing_precision_emits_failure() -> None:
    gate = DefaultSliceGate()
    current = {"k": SliceScore(precision=0.70, recall=0.85, n=10)}
    baseline = _baseline(
        scores={"k": SliceScore(precision=0.9, recall=0.85, n=10)},
        gates={"k": SliceGateRule(precision_min=0.85, recall_min=0.80)},
    )
    failures = gate.evaluate(current, baseline)
    assert len(failures) == 1
    assert failures[0].metric == "precision"
    assert failures[0].observed == 0.70
    assert failures[0].threshold == 0.85


def test_slice_missing_from_baseline_is_not_a_regression() -> None:
    gate = DefaultSliceGate()
    current = {"new_slice": SliceScore(precision=0.5, recall=0.5, n=5)}
    baseline = _baseline(scores={}, gates={})
    assert gate.evaluate(current, baseline) == []

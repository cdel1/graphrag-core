import pytest

from graphrag_core.eval.models import (
    BaselineFile,
    RunReport,
    SliceGateRule,
    SliceScore,
    Violation,
)


def test_slice_score_records_metrics_and_n() -> None:
    s = SliceScore(precision=0.91, recall=0.86, n=42)
    assert s.precision == 0.91
    assert s.n == 42


def test_slice_gate_rule_has_min_thresholds() -> None:
    g = SliceGateRule(precision_min=0.85, recall_min=0.80)
    assert g.precision_min == 0.85


def test_violation_carries_slice_key_and_message() -> None:
    v = Violation(check="provenance_completeness", slice_key="document_class=oba", message="3 claims missing SOURCED_FROM")
    assert v.check == "provenance_completeness"


def test_baseline_file_roundtrip_json() -> None:
    bf = BaselineFile(
        harness_version="0.1.0",
        manifest_version="phase_0_synthetic@2026-06-10",
        created_at="2026-06-10T00:00:00Z",
        created_by="rebaseline-abc1234",
        model_pin={"extraction": "gpt-4o@2026-05", "seed": 42},
        slice_scores={"k": SliceScore(precision=0.9, recall=0.85, n=10)},
        slice_gates={"k": SliceGateRule(precision_min=0.85, recall_min=0.80)},
    )
    s = bf.model_dump_json()
    assert BaselineFile.model_validate_json(s) == bf


def test_run_report_has_violations_and_slice_scores_and_passed() -> None:
    r = RunReport(
        manifest_version="docred@2026-06-10",
        harness_version="0.1.0",
        tier_one_violations=[],
        slice_scores={},
        gate_failures=[],
        passed=True,
    )
    assert r.passed is True

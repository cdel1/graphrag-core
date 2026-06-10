"""Pydantic models for the eval harness.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §3.6 + §4.3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

SliceKey = str  # "axis1=value1/axis2=value2"


class SliceScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: float | None = None
    recall: float | None = None
    false_positive_rate: float | None = None
    n: int


class SliceGateRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision_min: float | None = None
    recall_min: float | None = None
    false_positive_rate_max: float | None = None


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    check: str
    slice_key: SliceKey | None = None
    message: str


class BaselineFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    harness_version: str
    manifest_version: str
    created_at: str
    created_by: str
    model_pin: dict[str, Any]
    slice_scores: dict[SliceKey, SliceScore]
    slice_gates: dict[SliceKey, SliceGateRule]


class GateFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    slice_key: SliceKey
    metric: str
    observed: float
    threshold: float


class RunReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: str
    harness_version: str
    tier_one_violations: list[Violation]
    slice_scores: dict[SliceKey, SliceScore]
    gate_failures: list[GateFailure]
    passed: bool

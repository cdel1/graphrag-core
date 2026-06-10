"""Default SliceGate impl — fail when a slice's metric falls below its gate.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §3.6.
"""

from __future__ import annotations

from graphrag_core.eval.models import (
    BaselineFile,
    GateFailure,
    SliceKey,
    SliceScore,
)


class DefaultSliceGate:
    def evaluate(
        self,
        current: dict[SliceKey, SliceScore],
        baseline: BaselineFile | None,
    ) -> list[GateFailure]:
        if baseline is None:
            return []
        failures: list[GateFailure] = []
        for slice_key, score in current.items():
            rule = baseline.slice_gates.get(slice_key)
            if rule is None:
                continue
            failures.extend(self._check_metric(slice_key, "precision", score.precision, rule.precision_min))
            failures.extend(self._check_metric(slice_key, "recall", score.recall, rule.recall_min))
            # false_positive_rate is "lower is better" — invert comparison
            if rule.false_positive_rate_max is not None and score.false_positive_rate is not None:
                if score.false_positive_rate > rule.false_positive_rate_max:
                    failures.append(GateFailure(
                        slice_key=slice_key,
                        metric="false_positive_rate",
                        observed=score.false_positive_rate,
                        threshold=rule.false_positive_rate_max,
                    ))
        return failures

    @staticmethod
    def _check_metric(
        slice_key: SliceKey,
        metric_name: str,
        observed: float | None,
        threshold: float | None,
    ) -> list[GateFailure]:
        if observed is None or threshold is None:
            return []
        if observed >= threshold:
            return []
        return [GateFailure(
            slice_key=slice_key,
            metric=metric_name,
            observed=observed,
            threshold=threshold,
        )]

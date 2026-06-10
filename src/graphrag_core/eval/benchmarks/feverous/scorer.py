"""FEVEROUS Scorer — slice precision/recall per label and per challenge.

A produced :Claim node with property `verdict` in {SUPPORTS, REFUTES, NOT ENOUGH INFO}
is the prediction. Slice axes declared by the manifest are scored independently;
the harness reports per declared axis.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.5.
"""

from __future__ import annotations

from collections import defaultdict

from graphrag_core.interfaces import GraphStore

from graphrag_core.eval.benchmarks.feverous.manifest import FEVEROUSManifest
from graphrag_core.eval.models import SliceKey, SliceScore


class FEVEROUSScorer:
    async def score(
        self, graph_store: GraphStore, manifest: FEVEROUSManifest
    ) -> dict[SliceKey, SliceScore]:
        # collect predictions: claim_id -> predicted_label
        predicted: dict[str, str] = {}
        for node in await graph_store.list_nodes():
            if node.label != "Claim":
                continue
            verdict = node.properties.get("verdict")
            if verdict is not None:
                predicted[node.id] = verdict

        # Per-label slice
        by_label_gold: dict[str, set[str]] = defaultdict(set)
        by_label_correct: dict[str, int] = defaultdict(int)
        by_label_predicted: dict[str, int] = defaultdict(int)
        # Per-challenge slice
        by_challenge_gold: dict[str, set[str]] = defaultdict(set)
        by_challenge_correct: dict[str, int] = defaultdict(int)
        by_challenge_predicted: dict[str, int] = defaultdict(int)

        for claim in manifest.gold_claims:
            by_label_gold[claim.label].add(claim.id)
            by_challenge_gold[claim.challenge].add(claim.id)
            if claim.id in predicted:
                if predicted[claim.id] == claim.label:
                    by_label_correct[claim.label] += 1
                    by_challenge_correct[claim.challenge] += 1

        # Predicted counts per label (for precision)
        for cid, pred in predicted.items():
            by_label_predicted[pred] += 1
        # For challenge, predicted-per-bucket only counts claims that exist in gold
        # (we don't know the challenge bucket of a predicted-but-not-in-gold claim)
        for claim in manifest.gold_claims:
            if claim.id in predicted:
                by_challenge_predicted[claim.challenge] += 1

        scores: dict[SliceKey, SliceScore] = {}
        for label, gold_set in by_label_gold.items():
            tp = by_label_correct[label]
            predicted_count = by_label_predicted[label]
            precision = tp / predicted_count if predicted_count > 0 else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            scores[f"label={label}"] = SliceScore(
                precision=precision, recall=recall, n=len(gold_set)
            )
        for challenge, gold_set in by_challenge_gold.items():
            tp = by_challenge_correct[challenge]
            predicted_count = by_challenge_predicted[challenge]
            precision = tp / predicted_count if predicted_count > 0 else 0.0
            recall = tp / len(gold_set) if gold_set else 0.0
            scores[f"challenge={challenge}"] = SliceScore(
                precision=precision, recall=recall, n=len(gold_set)
            )
        return scores

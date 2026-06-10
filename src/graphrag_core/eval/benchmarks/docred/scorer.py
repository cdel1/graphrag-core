"""DocRED Scorer — slice precision/recall per relation_type.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.5.
"""

from __future__ import annotations

from collections import defaultdict

from graphrag_core.interfaces import GraphStore

from graphrag_core.eval.benchmarks.docred.manifest import DocREDManifest
from graphrag_core.eval.benchmarks.docred.relation_mapping import (
    graph_edge_to_docred_relation,
)
from graphrag_core.eval.models import SliceKey, SliceScore


class DocREDScorer:
    async def score(
        self, graph_store: GraphStore, manifest: DocREDManifest
    ) -> dict[SliceKey, SliceScore]:
        gold_by_type: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for r in manifest.gold_relations:
            gold_by_type[r.relation_type].add((r.head_id, r.tail_id))

        produced_by_type: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for edge in await graph_store.list_relationships():
            rel = graph_edge_to_docred_relation(edge.type)
            if rel is None:
                continue
            produced_by_type[rel].add((edge.source_id, edge.target_id))

        scores: dict[SliceKey, SliceScore] = {}
        for rel_type, gold in gold_by_type.items():
            produced = produced_by_type.get(rel_type, set())
            tp = len(gold & produced)
            fp = len(produced - gold)
            fn = len(gold - produced)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            scores[f"relation_type={rel_type}"] = SliceScore(
                precision=precision, recall=recall, n=len(gold)
            )
        return scores

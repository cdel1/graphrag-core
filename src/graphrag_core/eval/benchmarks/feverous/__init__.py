"""FEVEROUS pair registration."""

from pathlib import Path

from graphrag_core.eval.benchmarks.feverous.manifest import FEVEROUSManifestLoader
from graphrag_core.eval.benchmarks.feverous.runner import FEVEROUSPipelineRunner
from graphrag_core.eval.benchmarks.feverous.scorer import FEVEROUSScorer
from graphrag_core.eval.harness import ManifestScorerPair
from graphrag_core.eval.registry import register_pair
from graphrag_core.eval.tier_one import (
    NoOrphanIntelligenceCheck,
    ProvenanceCompletenessCheck,
    SchemaConformanceCheck,
)

# repo-root anchor: eval/fixtures/feverous/sample.jsonl
# module path: src/graphrag_core/eval/benchmarks/feverous/__init__.py
_FIXTURE_PATH = Path(__file__).resolve().parents[5] / "eval/fixtures/feverous/sample.jsonl"


@register_pair("feverous")
def feverous_pair() -> ManifestScorerPair:
    manifest = FEVEROUSManifestLoader().load(_FIXTURE_PATH)
    return ManifestScorerPair(
        manifest=manifest,
        corpus_path=_FIXTURE_PATH,
        pipeline_runner=FEVEROUSPipelineRunner(),
        tier_one_checks=[
            ProvenanceCompletenessCheck(),
            NoOrphanIntelligenceCheck(),
            SchemaConformanceCheck(
                allowed_labels={"Claim", "Entity", "Document", "Chunk", "Stakeholder"}
            ),
        ],
        scorer=FEVEROUSScorer(),
    )

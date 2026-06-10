"""DocRED pair registration."""

from pathlib import Path

from graphrag_core.eval.benchmarks.docred.manifest import DocREDManifestLoader
from graphrag_core.eval.benchmarks.docred.runner import DocREDPipelineRunner
from graphrag_core.eval.benchmarks.docred.scorer import DocREDScorer
from graphrag_core.eval.harness import ManifestScorerPair
from graphrag_core.eval.registry import register_pair
from graphrag_core.eval.tier_one import (
    NoOrphanIntelligenceCheck,
    ProvenanceCompletenessCheck,
    SchemaConformanceCheck,
)

# repo-root anchor: eval/fixtures/docred/sample.jsonl
# module path: src/graphrag_core/eval/benchmarks/docred/__init__.py
_FIXTURE_PATH = Path(__file__).resolve().parents[5] / "eval/fixtures/docred/sample.jsonl"


@register_pair("docred")
def docred_pair() -> ManifestScorerPair:
    manifest = DocREDManifestLoader().load(_FIXTURE_PATH)
    return ManifestScorerPair(
        manifest=manifest,
        corpus_path=_FIXTURE_PATH,
        pipeline_runner=DocREDPipelineRunner(),
        tier_one_checks=[
            ProvenanceCompletenessCheck(),
            NoOrphanIntelligenceCheck(),
            SchemaConformanceCheck(
                allowed_labels={"Claim", "Entity", "Document", "Chunk", "Stakeholder"}
            ),
        ],
        scorer=DocREDScorer(),
    )

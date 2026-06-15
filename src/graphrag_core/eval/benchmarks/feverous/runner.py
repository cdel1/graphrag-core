"""FEVEROUS PipelineRunner — LLM-driven verdict classification.

Reads each entry of the FEVEROUS JSONL fixture, asks the configured
LLMClient to classify the claim as SUPPORTS / REFUTES / NOT ENOUGH INFO,
and emits one :Claim node per fixture entry with the predicted verdict
as a property. The FEVEROUSScorer reads ``node.properties["verdict"]``
and compares it against the gold label for per-label and per-challenge
slice scores.

The runner takes no retrieval action — it relies on the model's own
world knowledge. That is intentional for an L1 example benchmark; it
keeps the substrate dependency surface to ``LLMClient`` only and lets
each pair register a different provider for portability comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from graphrag_core.interfaces import GraphStore
from graphrag_core.llm.base import BaseLLMClient
from graphrag_core.models import GraphNode, GraphRelationship

_SYSTEM_PROMPT = (
    "You are a claim-verification engine. For each claim, decide whether "
    "the claim is supported, refuted, or whether there is not enough "
    "information to decide, based on your world knowledge. Respond with "
    "JSON matching the provided schema."
)


class _Verdict(BaseModel):
    """Structured-output target for the LLM's classification."""

    label: Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class FEVEROUSPipelineRunner:
    """LLM-driven FEVEROUS pipeline.

    Parameters
    ----------
    llm:
        The :class:`graphrag_core.llm.base.BaseLLMClient` (or any
        Protocol-compatible subclass) used to classify each claim.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    async def run(self, corpus_path: Path, graph_store: GraphStore) -> None:
        with corpus_path.open() as f:
            for line in f:
                entry = json.loads(line)
                if "id" not in entry or "claim" not in entry:
                    continue
                claim_id = str(entry["id"])
                claim_text = entry["claim"]
                verdict = await self._classify(claim_text)
                doc_id = f"feverous-doc:{claim_id}"
                await graph_store.merge_node(
                    GraphNode(
                        id=doc_id,
                        label="Document",
                        properties={"title": f"FEVEROUS claim {claim_id}"},
                    ),
                    import_run_id="feverous-eval",
                )
                await graph_store.merge_node(
                    GraphNode(
                        id=claim_id,
                        label="Claim",
                        properties={"text": claim_text, "verdict": verdict},
                    ),
                    import_run_id="feverous-eval",
                )
                await graph_store.merge_relationship(
                    GraphRelationship(
                        source_id=claim_id,
                        target_id=doc_id,
                        type="SOURCED_FROM",
                        properties={},
                    ),
                    import_run_id="feverous-eval",
                )

    async def _classify(self, claim: str) -> str:
        result = await self._llm.complete_json(
            messages=[{"role": "user", "content": f"Claim: {claim}"}],
            schema=_Verdict,
            system=_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return result.label

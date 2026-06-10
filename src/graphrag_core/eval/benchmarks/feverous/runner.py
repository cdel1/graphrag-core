"""FEVEROUS PipelineRunner — minimal Band-3 stub.

Streams the FEVEROUS JSONL fixture and emits one Document node per claim.
Score will be near-zero until a real verification pipeline lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphrag_core.interfaces import GraphStore
from graphrag_core.models import GraphNode


class FEVEROUSPipelineRunner:
    async def run(self, corpus_path: Path, graph_store: GraphStore) -> None:
        with corpus_path.open() as f:
            for line in f:
                entry = json.loads(line)
                if "id" not in entry:
                    continue
                claim_id = str(entry["id"])
                await graph_store.merge_node(
                    GraphNode(label="Document", id=f"feverous:{claim_id}", properties={"claim": entry.get("claim", "")}),
                    import_run_id="feverous-eval",
                )

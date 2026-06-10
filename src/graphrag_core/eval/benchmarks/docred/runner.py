"""DocRED PipelineRunner — minimal Band-3 stub.

Streams the DocRED JSONL fixture and emits one Document node per entry.
Does NOT extract relations — that's the framework's default extraction's
job, and the score will be near-zero until that lands. The harness loop
runs end-to-end so baselines can be written.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphrag_core.interfaces import GraphStore
from graphrag_core.models import GraphNode


class DocREDPipelineRunner:
    async def run(self, corpus_path: Path, graph_store: GraphStore) -> None:
        with corpus_path.open() as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry["title"]
                await graph_store.merge_node(
                    GraphNode(label="Document", id=doc_id, properties={"title": doc_id}),
                    import_run_id="docred-eval",
                )

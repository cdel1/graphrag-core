"""DocRED PipelineRunner — LLM-driven relation extraction.

For each entry in the DocRED JSONL fixture, the runner constructs a
prompt from the document text and the list of entity names taken from
the gold vertex set, then asks the configured LLMClient to emit typed
relations between those entities. Each extracted relation is written
to the graph as two :Entity endpoints plus a relationship whose ``type``
is the DocRED relation identifier (Wikidata PIDs such as ``P17``,
``P127``). The DocRED scorer compares the produced (head_id, tail_id)
tuples per relation type against the gold set.

Endpoint ids are namespaced by document title (``"{doc_id}:{name}"``)
so the same surface form appearing in two different documents is
treated as two distinct nodes by the scorer.

Quality on this benchmark is bounded by the model's ability to produce
Wikidata-style PIDs from a single prompt and by surface-form matches
against the gold mention names; absolute precision/recall is expected
to be low. The benchmark's purpose at L1 is harness-wiring + provider
side-by-side comparability rather than state-of-the-art extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from graphrag_core.interfaces import GraphStore
from graphrag_core.llm.base import BaseLLMClient
from graphrag_core.models import GraphNode, GraphRelationship

_SYSTEM_PROMPT = (
    "You are a relation-extraction engine for the DocRED schema. "
    "Given a document and a list of mentioned entities, return typed "
    "relations between pairs of those entities using Wikidata property "
    "identifiers (PIDs) such as P17 (country), P127 (owned by), "
    "P577 (publication date), P150 (contains administrative entity), "
    "etc. Use the entity names exactly as given. Emit only relations "
    "that are explicitly supported by the document text. Respond with "
    "JSON matching the provided schema."
)

_DOC_TEXT_CHAR_LIMIT = 8000


class _Relation(BaseModel):
    head: str
    tail: str
    relation_type: str


class _Result(BaseModel):
    relations: list[_Relation]


class DocREDPipelineRunner:
    """LLM-driven DocRED pipeline.

    Parameters
    ----------
    llm:
        The :class:`graphrag_core.llm.base.BaseLLMClient` (or any
        Protocol-compatible subclass) used to extract relations.
    """

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    async def run(self, corpus_path: Path, graph_store: GraphStore) -> None:
        with corpus_path.open() as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry["title"]
                entity_names = self._entity_names(entry)
                text = self._flatten_text(entry["sents"])
                relations = await self._extract(text, entity_names)
                await self._emit(graph_store, doc_id, entity_names, relations)

    @staticmethod
    def _entity_names(entry: dict) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for vertex in entry.get("vertexSet", []):
            if not vertex:
                continue
            name = vertex[0].get("name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    @staticmethod
    def _flatten_text(sents: list[list[str]]) -> str:
        flat = " ".join(" ".join(s) for s in sents)
        return flat[:_DOC_TEXT_CHAR_LIMIT]

    async def _extract(
        self, text: str, entity_names: list[str]
    ) -> list[_Relation]:
        entities_block = ", ".join(entity_names)
        user_prompt = (
            f"Document text:\n{text}\n\n"
            f"Mentioned entities: {entities_block}\n\n"
            "Return typed relations between these entities."
        )
        try:
            result = await self._llm.complete_json(
                messages=[{"role": "user", "content": user_prompt}],
                schema=_Result,
                system=_SYSTEM_PROMPT,
                temperature=0.0,
            )
        except Exception:
            return []
        return result.relations

    @staticmethod
    async def _emit(
        graph_store: GraphStore,
        doc_id: str,
        entity_names: list[str],
        relations: list[_Relation],
    ) -> None:
        valid = set(entity_names)
        emitted_endpoints: set[str] = set()
        for rel in relations:
            if rel.head not in valid or rel.tail not in valid:
                continue
            head_id = f"{doc_id}:{rel.head}"
            tail_id = f"{doc_id}:{rel.tail}"
            if head_id not in emitted_endpoints:
                await graph_store.merge_node(
                    GraphNode(
                        id=head_id,
                        label="Entity",
                        properties={"name": rel.head, "doc_id": doc_id},
                    ),
                    import_run_id="docred-eval",
                )
                emitted_endpoints.add(head_id)
            if tail_id not in emitted_endpoints:
                await graph_store.merge_node(
                    GraphNode(
                        id=tail_id,
                        label="Entity",
                        properties={"name": rel.tail, "doc_id": doc_id},
                    ),
                    import_run_id="docred-eval",
                )
                emitted_endpoints.add(tail_id)
            await graph_store.merge_relationship(
                GraphRelationship(
                    source_id=head_id,
                    target_id=tail_id,
                    type=rel.relation_type,
                    properties={},
                ),
                import_run_id="docred-eval",
            )

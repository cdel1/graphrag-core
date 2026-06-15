"""Unit tests for DocREDPipelineRunner (LLM-driven relation extraction)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_core.eval.benchmarks.docred.runner import (
    DocREDPipelineRunner,
    _Relation,
    _Result,
)
from graphrag_core.graph import InMemoryGraphStore


class _StubLLM:
    """A no-network LLM that returns queued relation lists."""

    def __init__(self, relations_per_doc: list[list[tuple[str, str, str]]]) -> None:
        self._queued = [list(rels) for rels in relations_per_doc]
        self.calls: list[dict] = []

    async def complete(self, *args, **kwargs):
        raise NotImplementedError

    async def complete_json(self, *, messages, schema, system, temperature=0.0, max_tokens=4096):
        self.calls.append({"messages": messages})
        if not self._queued:
            raise RuntimeError("StubLLM exhausted")
        rels = self._queued.pop(0)
        return _Result(
            relations=[_Relation(head=h, tail=t, relation_type=rt) for h, t, rt in rels]
        )


def _make_fixture(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "docred.jsonl"
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.mark.asyncio
async def test_runner_emits_entities_and_relationships(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        [
            {
                "title": "Doc1",
                "sents": [["Alice", "works", "at", "Acme", "."]],
                "vertexSet": [
                    [{"name": "Alice", "type": "PER", "pos": [0, 1], "sent_id": 0}],
                    [{"name": "Acme", "type": "ORG", "pos": [3, 4], "sent_id": 0}],
                ],
            }
        ],
    )
    llm = _StubLLM(relations_per_doc=[[("Alice", "Acme", "P108")]])
    store = InMemoryGraphStore()

    runner = DocREDPipelineRunner(llm=llm)
    await runner.run(fixture, store)

    nodes = await store.list_nodes()
    entity_nodes = [n for n in nodes if n.label == "Entity"]
    assert {n.id for n in entity_nodes} == {"Doc1:Alice", "Doc1:Acme"}

    rels = await store.list_relationships()
    assert len(rels) == 1
    r = rels[0]
    assert r.source_id == "Doc1:Alice"
    assert r.target_id == "Doc1:Acme"
    assert r.type == "P108"


@pytest.mark.asyncio
async def test_runner_drops_relations_referencing_unknown_entities(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        [
            {
                "title": "Doc1",
                "sents": [["Bob", "knows", "Carol", "."]],
                "vertexSet": [
                    [{"name": "Bob", "type": "PER", "pos": [0, 1], "sent_id": 0}],
                    [{"name": "Carol", "type": "PER", "pos": [2, 3], "sent_id": 0}],
                ],
            }
        ],
    )
    # The LLM hallucinates a relation involving "Mallory" who isn't in the doc.
    llm = _StubLLM(relations_per_doc=[[("Bob", "Mallory", "P26"), ("Bob", "Carol", "P26")]])
    store = InMemoryGraphStore()

    runner = DocREDPipelineRunner(llm=llm)
    await runner.run(fixture, store)

    rels = await store.list_relationships()
    assert len(rels) == 1
    assert rels[0].source_id == "Doc1:Bob"
    assert rels[0].target_id == "Doc1:Carol"


@pytest.mark.asyncio
async def test_runner_swallows_llm_errors_and_continues(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        [
            {
                "title": "Doc1",
                "sents": [["X."]],
                "vertexSet": [[{"name": "X", "type": "PER", "pos": [0, 1], "sent_id": 0}]],
            },
            {
                "title": "Doc2",
                "sents": [["Y.", "Z."]],
                "vertexSet": [
                    [{"name": "Y", "type": "PER", "pos": [0, 1], "sent_id": 0}],
                    [{"name": "Z", "type": "PER", "pos": [0, 1], "sent_id": 1}],
                ],
            },
        ],
    )

    class _RaisingThenOK:
        def __init__(self) -> None:
            self.n = 0

        async def complete_json(self, **_kwargs):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient")
            return _Result(relations=[_Relation(head="Y", tail="Z", relation_type="P26")])

    llm = _RaisingThenOK()
    store = InMemoryGraphStore()

    runner = DocREDPipelineRunner(llm=llm)
    await runner.run(fixture, store)

    rels = await store.list_relationships()
    assert len(rels) == 1
    assert rels[0].source_id == "Doc2:Y"
    assert rels[0].target_id == "Doc2:Z"

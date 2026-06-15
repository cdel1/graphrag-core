"""Unit tests for FEVEROUSPipelineRunner (LLM-driven verdict classification)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from graphrag_core.eval.benchmarks.feverous.runner import (
    FEVEROUSPipelineRunner,
    _Verdict,
)
from graphrag_core.graph import InMemoryGraphStore


class _StubLLM:
    """A no-network LLM that returns a queued sequence of verdicts."""

    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[dict] = []

    async def complete(self, *args, **kwargs):
        raise NotImplementedError

    async def complete_json(self, *, messages, schema, system, temperature=0.0, max_tokens=4096):
        self.calls.append({"messages": messages, "system": system})
        if not self._verdicts:
            raise RuntimeError("StubLLM exhausted")
        label = self._verdicts.pop(0)
        return schema(label=label)


def _make_fixture(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "feverous.jsonl"
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


@pytest.mark.asyncio
async def test_runner_emits_one_claim_node_per_entry(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        [
            {"id": 101, "claim": "The sky is blue.", "label": "SUPPORTS"},
            {"id": 202, "claim": "Water is dry.", "label": "REFUTES"},
        ],
    )
    llm = _StubLLM(verdicts=["SUPPORTS", "REFUTES"])
    store = InMemoryGraphStore()

    runner = FEVEROUSPipelineRunner(llm=llm)
    await runner.run(fixture, store)

    nodes = await store.list_nodes()
    claim_nodes = [n for n in nodes if n.label == "Claim"]
    assert len(claim_nodes) == 2
    by_id = {n.id: n for n in claim_nodes}
    assert by_id["101"].properties["verdict"] == "SUPPORTS"
    assert by_id["202"].properties["verdict"] == "REFUTES"
    assert by_id["101"].properties["text"] == "The sky is blue."
    assert len(llm.calls) == 2

    # Provenance: every Claim should have a SOURCED_FROM edge to a Document.
    doc_nodes = {n.id: n for n in nodes if n.label == "Document"}
    assert len(doc_nodes) == 2
    rels = await store.list_relationships()
    sourced_from = [r for r in rels if r.type == "SOURCED_FROM"]
    assert {(r.source_id, r.target_id) for r in sourced_from} == {
        ("101", "feverous-doc:101"),
        ("202", "feverous-doc:202"),
    }


@pytest.mark.asyncio
async def test_runner_skips_entries_missing_claim_or_id(tmp_path: Path) -> None:
    fixture = _make_fixture(
        tmp_path,
        [
            {"id": 1, "claim": "A."},
            {"id": 2},  # no claim — should be skipped
            {"claim": "C."},  # no id — should be skipped
            {"id": 3, "claim": "D."},
        ],
    )
    llm = _StubLLM(verdicts=["SUPPORTS", "REFUTES"])
    store = InMemoryGraphStore()

    runner = FEVEROUSPipelineRunner(llm=llm)
    await runner.run(fixture, store)

    nodes = await store.list_nodes()
    claim_nodes = [n for n in nodes if n.label == "Claim"]
    assert {n.id for n in claim_nodes} == {"1", "3"}
    assert len(llm.calls) == 2


def test_verdict_schema_enforces_three_labels() -> None:
    _Verdict(label="SUPPORTS")
    _Verdict(label="REFUTES")
    _Verdict(label="NOT ENOUGH INFO")
    with pytest.raises(Exception):
        _Verdict(label="MAYBE")

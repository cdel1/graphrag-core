"""DocRED Manifest + Loader.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §4.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocREDDocument(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    title: str
    sentences: list[list[str]]  # list of tokenised sentences


class DocREDGoldRelation(BaseModel):
    model_config = ConfigDict(frozen=True)
    head_id: str
    tail_id: str
    relation_type: str
    evidence_sentence_ids: list[int]


class DocREDManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: str
    slice_axes: list[str]
    token_budget: int
    model_pin: dict[str, Any]
    documents: list[DocREDDocument]
    gold_relations: list[DocREDGoldRelation]


class DocREDManifestLoader:
    def load(
        self,
        path: Path,
        model_pin: dict[str, Any] | None = None,
    ) -> DocREDManifest:
        documents: list[DocREDDocument] = []
        gold_relations: list[DocREDGoldRelation] = []
        with path.open() as f:
            for line in f:
                entry = json.loads(line)
                doc_id = entry["title"]
                documents.append(DocREDDocument(
                    id=doc_id,
                    title=entry["title"],
                    sentences=entry["sents"],
                ))
                for label in entry.get("labels", []):
                    head_v = entry["vertexSet"][label["h"]]
                    tail_v = entry["vertexSet"][label["t"]]
                    gold_relations.append(DocREDGoldRelation(
                        head_id=f"{doc_id}:{head_v[0]['name']}",
                        tail_id=f"{doc_id}:{tail_v[0]['name']}",
                        relation_type=label["r"],
                        evidence_sentence_ids=label.get("evidence", []),
                    ))
        return DocREDManifest(
            version="docred@2026-06-10",
            slice_axes=["relation_type"],
            token_budget=200_000,
            model_pin=model_pin if model_pin is not None else {"extraction": "gpt-4o-mini", "seed": 42},
            documents=documents,
            gold_relations=gold_relations,
        )

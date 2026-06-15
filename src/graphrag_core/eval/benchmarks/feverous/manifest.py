"""FEVEROUS Manifest + Loader.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §3.2 + §4.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class FEVEROUSGoldClaim(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    label: str
    claim_text: str
    challenge: str
    evidence_ids: list[str]


class FEVEROUSManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: str
    slice_axes: list[str]
    token_budget: int
    model_pin: dict[str, Any]
    gold_claims: list[FEVEROUSGoldClaim]


class FEVEROUSManifestLoader:
    def load(
        self,
        path: Path,
        model_pin: dict[str, Any] | None = None,
    ) -> FEVEROUSManifest:
        gold_claims: list[FEVEROUSGoldClaim] = []
        with path.open() as f:
            for line in f:
                entry = json.loads(line)
                if "id" not in entry or "label" not in entry:
                    continue  # skip header/metadata records
                evidence_ids: list[str] = []
                for ev_set in entry.get("evidence", []):
                    evidence_ids.extend(ev_set.get("content", []))
                gold_claims.append(FEVEROUSGoldClaim(
                    id=str(entry["id"]),
                    label=entry["label"],
                    claim_text=entry["claim"],
                    challenge=entry.get("challenge", "Other"),
                    evidence_ids=evidence_ids,
                ))
        return FEVEROUSManifest(
            version="feverous@2026-06-10",
            slice_axes=["label", "challenge"],
            token_budget=200_000,
            model_pin=model_pin if model_pin is not None else {"extraction": "gpt-4o-mini", "seed": 42},
            gold_claims=gold_claims,
        )

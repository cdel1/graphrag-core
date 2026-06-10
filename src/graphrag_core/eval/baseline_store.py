"""JSON-file BaselineStore — default impl sharded per manifest.

See docs/superpowers/specs/2026-06-10-eval-harness-design.md §3.6.
"""

from __future__ import annotations

from pathlib import Path

from graphrag_core.eval.models import BaselineFile


class JSONFileBaselineStore:
    """Per-manifest sharded directories; per-harness-version JSON files."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, manifest_version: str, harness_version: str) -> Path:
        return self.root / manifest_version / f"{harness_version}.json"

    def write(self, baseline: BaselineFile) -> Path:
        path = self._path(baseline.manifest_version, baseline.harness_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(baseline.model_dump_json(indent=2))
        return path

    def read(
        self, manifest_version: str, harness_version: str
    ) -> BaselineFile | None:
        path = self._path(manifest_version, harness_version)
        if not path.exists():
            return None
        return BaselineFile.model_validate_json(path.read_text())

    def list_versions(self, manifest_version: str) -> list[str]:
        d = self.root / manifest_version
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

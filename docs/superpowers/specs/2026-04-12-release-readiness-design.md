# Release Readiness: graphrag-core v0.2.0

**Date:** 2026-04-12
**Status:** Approved
**Goal:** Prepare graphrag-core for PyPI publication as v0.2.0 — metadata, README, CHANGELOG, CI/CD, type marker.

---

## Decisions

| Decision | Rationale |
|---|---|
| Version bump to 0.2.0 | BB1-BB8 complete is a major milestone past the 0.1.0 interfaces-only commit |
| Two CI workflows (test + release) | Test on every push, publish on tag. Minimal infrastructure. |
| Unit tests only in CI | Integration tests require Neo4j Docker — too slow/complex for CI now. Run locally. |
| Trusted publisher for PyPI | OIDC flow, no API token stored. GitHub Actions native. |
| No docs site | README is the docs for v0.2.0. Full docs site is a future concern. |
| No version tag in this sprint | Tag manually after reviewing README on GitHub. |

---

## Section 1: pyproject.toml — PyPI metadata

Add to the existing `[project]` table:

- `authors = [{name = "...", email = "..."}]`
- `readme = "README.md"`
- `classifiers` — Python version, license, topic
- `[project.urls]` — Homepage, Repository, Issues

Version bump: `version = "0.2.0"`

---

## Section 2: README overhaul

Replace the current stale README with:

1. **One-liner + architecture diagram** — Keep existing diagram, update to show all 8 BBs
2. **Install** — `pip install graphrag-core` + extras (`[neo4j]`, `[anthropic]`, `[all]`)
3. **Quick start** — Minimal working example: parse, extract, store, search
4. **Building blocks table** — All 8, showing what's implemented vs Protocol-only
5. **Extension pattern** — Keep existing example, it's good
6. **Development** — Run tests, start Neo4j, integration tests

---

## Section 3: CHANGELOG

`CHANGELOG.md` following Keep a Changelog format:

- **v0.2.0** — All 8 building blocks with implementations
- **v0.1.0** — Initial commit, BB1-BB4 Protocol interfaces and models only

---

## Section 4: CI/CD — GitHub Actions

### `.github/workflows/test.yml`

Triggers: push and PR to `main`.

Steps:
1. Checkout
2. Setup Python 3.12 + uv
3. `uv sync --all-extras`
4. `uv run pytest tests/ -x -q` (unit tests only)
5. Boundary check: grep for domain-specific terms in `src/`

### `.github/workflows/release.yml`

Triggers: push tag `v*`.

Steps:
1. Checkout
2. Setup Python 3.12 + uv
3. `uv build`
4. Publish to PyPI via `pypa/gh-action-pypi-publish` (trusted publisher)

---

## Section 5: Final items

- `src/graphrag_core/py.typed` — Empty marker file for type checker support
- `.gitignore` audit — Verify `dist/`, `*.egg-info/` excluded
- `pyproject.toml` readme field

---

## New File Tree (additions only)

```
graphrag-core/
├── .github/
│   └── workflows/
│       ├── test.yml              # CI: lint + unit tests + boundary check
│       └── release.yml           # CD: build + publish to PyPI on tag
├── src/graphrag_core/
│   └── py.typed                  # PEP 561 type marker
├── CHANGELOG.md
├── README.md                     # Overhauled
└── pyproject.toml                # Updated metadata + version
```

---

## Not Included

- Docs site (Sphinx, MkDocs, etc.)
- Integration tests in CI (requires Neo4j Docker)
- Automated version bumping
- Pre-commit hooks

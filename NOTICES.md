# Third-Party Notices

`graphrag-core` is MIT-licensed (see [`LICENSE`](LICENSE)).

The framework bundles a small number of third-party data fixtures used by the
`graphrag_core.eval` harness for benchmark scoring. Those fixtures retain
their own licenses and attribution requirements, distinct from the MIT
license that applies to the code. This file is the canonical index.

If you redistribute any file under `eval/fixtures/`, you must comply with
that fixture's own license — preserving attribution and (where applicable)
re-licensing your redistribution under the same terms.

---

## DocRED — `eval/fixtures/docred/`

| Field | Value |
|---|---|
| **Asset** | 50-document subsample of the DocRED dev split (JSONL) |
| **Source** | https://huggingface.co/datasets/thunlp/docred (HuggingFace mirror of the original `thunlp/DocRED` dataset) |
| **Paper** | Yao et al., *DocRED: A Large-Scale Document-Level Relation Extraction Dataset*, ACL 2019 — https://arxiv.org/abs/1906.06127 |
| **License** | MIT (same as graphrag-core) |
| **Attribution required** | Yes — credit `thunlp` (Tsinghua NLP) and cite the ACL 2019 paper if you publish results derived from this fixture |
| **Files** | `eval/fixtures/docred/sample.jsonl`, `eval/fixtures/docred/README.md` |

Because DocRED is MIT-licensed and graphrag-core is MIT-licensed, this fixture
adds no obligations beyond attribution.

---

## FEVEROUS — `eval/fixtures/feverous/`

| Field | Value |
|---|---|
| **Asset** | 50-claim subsample of the FEVEROUS dev split (JSONL) |
| **Source** | https://fever.ai/dataset/feverous.html — `feverous_dev_challenges.jsonl` |
| **Paper** | Aly et al., *FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information*, NeurIPS 2021 — https://arxiv.org/abs/2106.05707 |
| **License** | **CC-BY-SA 3.0** (Creative Commons Attribution-ShareAlike 3.0 Unported) — different from the repo's MIT license |
| **Attribution required** | Yes — credit the FEVEROUS authors + the underlying Wikipedia contributors |
| **ShareAlike obligation** | Yes — if you redistribute this fixture (or a derivative), you must do so under CC-BY-SA 3.0 (or a compatible license) |
| **Files** | `eval/fixtures/feverous/sample.jsonl`, `eval/fixtures/feverous/README.md`, `eval/fixtures/feverous/LICENSE` (full CC-BY-SA 3.0 text) |

### What CC-BY-SA 3.0 means in practice

Three common cases for graphrag-core consumers:

| Scenario | CC-BY-SA 3.0 obligation? |
|---|---|
| `pip install graphrag-core` and use the harness against FEVEROUS internally | None — internal use is unrestricted |
| Build your own wheel that bundles `sample.jsonl` and ship it to customers | Yes — preserve attribution; that file remains CC-BY-SA 3.0 |
| Derive a new annotated dataset from FEVEROUS and publish it | Yes — the derivative dataset must also be CC-BY-SA 3.0 |

The MIT license that covers the rest of `graphrag-core` (including all code in
`src/`, `tests/`, and the loader / scorer modules under
`src/graphrag_core/eval/benchmarks/feverous/`) is **not** affected by the
CC-BY-SA 3.0 license on the fixture file. Code and data are separate works.

The underlying Wikipedia articles referenced by `evidence[*].content` entries
in `sample.jsonl` are © Wikipedia contributors under CC-BY-SA 3.0.

---

## Updating this file

When adding a new fixture under `eval/fixtures/`:

1. Add the fixture-local `README.md` with source / paper / license / seed.
2. Add the fixture-local `LICENSE` (full upstream license text) if it isn't
   MIT or another permissive license already represented at the repo root.
3. Add a row in this file under the appropriate section, or a new section
   if a new license family appears.

If a fixture introduces a license family not already documented here
(e.g., GPL, MPL, an EULA-style data license), open an issue first — those
families have stronger obligations and may not be a good fit for the repo's
permissive-license posture.

# DocRED fixture

A 50-document subsample of DocRED dev split, JSONL-formatted (one document per line).

**Source:** https://huggingface.co/datasets/thunlp/docred — `data/dev.json.gz` (HuggingFace mirror of the original thunlp/DocRED dataset; MIT license). The canonical repo https://github.com/thunlp/DocRED no longer ships `data/dev.json` inline (it now points to Google Drive); the HuggingFace mirror is published by the same authors (`thunlp`) and preserves the original schema (`h`, `t`, `r`, `evidence`).
**Paper:** Yao et al., *DocRED: A Large-Scale Document-Level Relation Extraction Dataset*, ACL 2019 — https://arxiv.org/abs/1906.06127
**Random seed:** 42 (see `subsample.py` script in the [eval-fixture handoff doc](../../../../docs/superpowers/handoffs/2026-06-10-eval-fixture-acquisition.md))
**Subsample date:** 2026-06-10
**Distribution:** 50 documents bucketed by max-entity-pair-distance (short / medium / long). Covers 65 distinct Wikidata relation types — well above the ≥6 minimum needed for slice gating.

## Schema

Each line is one DocRED entry with fields `title`, `sents`, `vertexSet`, `labels`.
See `src/graphrag_core/eval/benchmarks/docred/manifest.py` for the loader (added in a follow-up PR).

## Regenerate

```bash
mkdir -p /tmp/docred && cd /tmp/docred
curl -sSL https://huggingface.co/datasets/thunlp/docred/resolve/main/data/dev.json.gz -o dev.json.gz
gunzip -f dev.json.gz
# copy `subsample.py` from the handoff doc, then:
python3 subsample.py
cp sample.jsonl <repo-root>/eval/fixtures/docred/sample.jsonl
```

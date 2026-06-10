# FEVEROUS fixture

A 50-claim subsample of FEVEROUS dev split, stratified by (label x challenge).

**Source:** https://fever.ai/dataset/feverous.html — `feverous_dev_challenges.jsonl` (CC-BY-SA 3.0, Wikipedia-derived)
**Paper:** Aly et al., *FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information*, NeurIPS 2021 — https://arxiv.org/abs/2106.05707
**Random seed:** 42 (see the [eval-fixture handoff doc](../../../../docs/superpowers/handoffs/2026-06-10-eval-fixture-acquisition.md))
**Subsample date:** 2026-06-10
**Distribution:** 50 claims stratified across `label` x `challenge`.
- Labels: `SUPPORTS=18, REFUTES=16, NOT ENOUGH INFO=16`
- Challenges: `Search terms not in claim=9, Entity Disambiguation=9, Numerical Reasoning=8, Other=8, Combining Tables and Text=8, Multi-hop Reasoning=8`

Note: the upstream `feverous_dev_challenges.jsonl` ships with a first-line dummy/header record (all fields empty); the subsample script skips it.

## Schema

Each line is one FEVEROUS entry with fields `id`, `label`, `claim`, `evidence`, `challenge`, `annotator_operations`.
See `src/graphrag_core/eval/benchmarks/feverous/manifest.py` for the loader (added in a follow-up PR).

## Regenerate

```bash
mkdir -p /tmp/feverous && cd /tmp/feverous
curl -sSL https://fever.ai/download/feverous/feverous_dev_challenges.jsonl -o dev.jsonl
# copy `subsample.py` from the handoff doc, then:
python3 subsample.py
cp sample.jsonl <repo-root>/eval/fixtures/feverous/sample.jsonl
```

## License attribution

FEVEROUS is released under CC-BY-SA 3.0. This fixture redistributes a subsample under the same license. The Wikipedia source articles referenced by `evidence[*].content` are (c) Wikipedia contributors under CC-BY-SA 3.0.

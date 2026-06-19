# `curation/` — INTERFACE (BB5)

**Substrate role:** Layer-3 attestation contract (text doctrine).
**Python surface:** none — this package ships no Protocols, models, or implementations.
**Doctrine:** ADR-0035 (BB5 substrate); ADR-0038 (Python-removal amendment).

---

## Layer 3 — the attestation contract

Layer 3 is a **contract**, not a Protocol shape. Every Layer-3 mutation must produce a promotion event recording: attestor kind (distinguishing human vs. agent), attestor id, rationale, supporting excerpts, and timestamp; plus an immutable audit record of the mutation itself.

Consumers may implement any surface that produces a contract-satisfying promotion event (e.g., batch-and-apply review, continuous editing, async queue, agent-driven review); `graphrag-core` takes no opinion on which shape and ships no reference Protocol.

## What this package does NOT contain

- No `DetectionLayer` / `LLMCurationLayer` / `ApprovalGateway` Protocols.
- No `CurationIssue` / `CurationReport` / `ApprovalBatch` / `ApplyResult` models.
- No `DeterministicDetectionLayer` or `CurationPipeline`.

Graph-quality concerns (duplicate / orphan / schema-violation detection, telemetry, ingest-time deduplication, candidate review) are play-shaped and live at Layer 2 in consuming applications. The pre-0.12.0 surface that attempted to ship these at L1 had zero consumers in production usage and was removed per ADR-0038.

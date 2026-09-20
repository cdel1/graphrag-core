"""BB2: LLM-powered schema-guided entity extraction engine."""

from __future__ import annotations

import logging

import pydantic

from graphrag_core.interfaces import ExtractionPromptBuilder, LLMClient
from graphrag_core.models import (
    ChunkExtractionResult,
    Chunk,
    ExtractedNode,
    ExtractedRelationship,
    ExtractionResult,
    ImportRun,
    OntologySchema,
    ProvenanceLink,
    RejectedNode,
    RejectedRelationship,
    RejectionReason,
    SchemaAdmission,
)

logger = logging.getLogger(__name__)


def validate_extraction(
    nodes: list[ExtractedNode],
    rels: list[ExtractedRelationship],
    schema: OntologySchema,
    chunk_id: str | None = None,
) -> SchemaAdmission:
    """Split extracted nodes and relationships into admitted and rejected.

    The schema decides what enters the typed graph; nothing the extractor
    emitted is discarded. A rejected emission comes back verbatim — properties
    included, so the passage that justified it survives — typed by the reason
    it was not admitted.

    Args:
        nodes: Nodes the extractor emitted.
        rels: Relationships the extractor emitted.
        schema: The ontology schema that decides admission.
        chunk_id: Chunk the emissions came from, recorded on each rejection so
            an un-admitted node's chunk link survives with it. ``None`` when
            validating outside a chunk context.

    Returns:
        The admitted nodes and relationships plus every rejection, typed by
        :class:`RejectionReason`.
    """
    allowed_labels = {nt.label for nt in schema.node_types}
    rel_constraints = {
        rt.type: (set(rt.source_types), set(rt.target_types))
        for rt in schema.relationship_types
    }

    admitted_nodes, rejected_nodes = _admit_nodes(nodes, allowed_labels, chunk_id)
    node_labels = {n.id: n.label for n in admitted_nodes}
    admitted_rels, rejected_rels = _admit_relationships(
        rels, node_labels, rel_constraints, chunk_id
    )

    return SchemaAdmission(
        nodes=admitted_nodes,
        relationships=admitted_rels,
        rejected_nodes=rejected_nodes,
        rejected_relationships=rejected_rels,
    )


def _admit_nodes(
    nodes: list[ExtractedNode],
    allowed_labels: set[str],
    chunk_id: str | None,
) -> tuple[list[ExtractedNode], list[RejectedNode]]:
    """Split nodes on whether the schema declares their label."""
    admitted: list[ExtractedNode] = []
    rejected: list[RejectedNode] = []
    for node in nodes:
        if node.label in allowed_labels:
            admitted.append(node)
        else:
            rejected.append(
                RejectedNode(
                    node=node,
                    reason=RejectionReason.UNDECLARED_NODE_LABEL,
                    chunk_id=chunk_id,
                )
            )
    return admitted, rejected


def _admit_relationships(
    rels: list[ExtractedRelationship],
    node_labels: dict[str, str],
    rel_constraints: dict[str, tuple[set[str], set[str]]],
    chunk_id: str | None,
) -> tuple[list[ExtractedRelationship], list[RejectedRelationship]]:
    """Split relationships on the first admission check each one fails."""
    admitted: list[ExtractedRelationship] = []
    rejected: list[RejectedRelationship] = []
    for rel in rels:
        reason = _rejection_reason(rel, node_labels, rel_constraints)
        if reason is None:
            admitted.append(rel)
        else:
            rejected.append(
                RejectedRelationship(relationship=rel, reason=reason, chunk_id=chunk_id)
            )
    return admitted, rejected


def _rejection_reason(
    rel: ExtractedRelationship,
    node_labels: dict[str, str],
    rel_constraints: dict[str, tuple[set[str], set[str]]],
) -> RejectionReason | None:
    """Return why the schema rejects this relationship, or None if it admits it.

    Checks run in order and the first failure wins: an undeclared type is
    reported as such even when its endpoints are also unresolvable.
    """
    if rel.type not in rel_constraints:
        return RejectionReason.UNDECLARED_RELATIONSHIP_TYPE
    if rel.source_id not in node_labels or rel.target_id not in node_labels:
        return RejectionReason.DANGLING_ENDPOINT
    source_types, target_types = rel_constraints[rel.type]
    if (
        node_labels[rel.source_id] not in source_types
        or node_labels[rel.target_id] not in target_types
    ):
        return RejectionReason.ENDPOINT_TYPE_VIOLATION
    return None


def _assemble_result(
    per_chunk: list[tuple[str, SchemaAdmission]],
    malformed_chunk_extractions: int,
) -> ExtractionResult:
    """Flatten per-chunk admissions into one run-level result.

    Provenance is built from the admitted nodes only; a rejected node's chunk
    link travels on the rejection itself.
    """
    return ExtractionResult(
        nodes=[node for _, a in per_chunk for node in a.nodes],
        relationships=[rel for _, a in per_chunk for rel in a.relationships],
        provenance=[
            ProvenanceLink(chunk_id=chunk_id, node_id=node.id, confidence=1.0)
            for chunk_id, a in per_chunk
            for node in a.nodes
        ],
        rejected_nodes=[r for _, a in per_chunk for r in a.rejected_nodes],
        rejected_relationships=[
            r for _, a in per_chunk for r in a.rejected_relationships
        ],
        quality_signals=(
            {"malformed_chunk_extractions": malformed_chunk_extractions}
            if malformed_chunk_extractions
            else None
        ),
    )


def _log_rejections(chunk_id: str, admission: SchemaAdmission) -> None:
    """Report what a chunk's emissions failed on — names, not just counts."""
    if not admission.rejected_nodes and not admission.rejected_relationships:
        return
    logger.info(
        "Chunk %s: schema did not admit %d node(s) %s and %d relationship(s) %s",
        chunk_id,
        len(admission.rejected_nodes),
        sorted({r.node.label for r in admission.rejected_nodes}),
        len(admission.rejected_relationships),
        sorted({r.relationship.type for r in admission.rejected_relationships}),
    )


class DefaultPromptBuilder:
    """Builds the default system prompt for LLM-based entity extraction."""

    def build_system_prompt(self, schema: OntologySchema) -> str:
        node_descriptions = []
        for nt in schema.node_types:
            props = ", ".join(
                f"{p.name} ({p.type}{', required' if p.required else ''})"
                for p in nt.properties
            )
            line = f"- {nt.label}: properties=[{props}]"
            if nt.description:
                line += f" \u2014 {nt.description}"
            node_descriptions.append(line)

        rel_descriptions = []
        for rt in schema.relationship_types:
            line = f"- {rt.type}: {rt.source_types} -> {rt.target_types}"
            if rt.description:
                line += f" \u2014 {rt.description}"
            rel_descriptions.append(line)

        return (
            "You are an entity extraction engine. Extract entities and relationships "
            "from the provided text according to this schema.\n\n"
            "ALLOWED NODE TYPES:\n"
            + "\n".join(node_descriptions)
            + "\n\nALLOWED RELATIONSHIP TYPES:\n"
            + "\n".join(rel_descriptions)
            + "\n\nDo not extract entities or relationships not listed above.\n\n"
            "Rules:\n"
            "- Every node id must be unique and descriptive (e.g., 'person-alice', 'company-acme')\n"
            "- Only use node types and relationship types listed above\n"
            "- Include all required properties for each node type\n"
            "- Return empty arrays if no entities are found"
        )


class LLMExtractionEngine:
    """Extracts entities and relationships from text using an LLM, guided by an ontology schema."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: ExtractionPromptBuilder | None = None,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder or DefaultPromptBuilder()

    async def extract(
        self,
        chunks: list[Chunk],
        schema: OntologySchema,
        import_run: ImportRun,
    ) -> ExtractionResult:
        per_chunk: list[tuple[str, SchemaAdmission]] = []
        malformed_chunk_extractions = 0

        system_prompt = self._prompt_builder.build_system_prompt(schema)

        for chunk in chunks:
            try:
                nodes, rels = await self._extract_chunk(chunk, system_prompt)
            except pydantic.ValidationError:
                logger.warning(
                    "Skipping chunk %s: LLM response failed schema validation",
                    chunk.id,
                )
                malformed_chunk_extractions += 1
                continue

            admission = self._validate(nodes, rels, schema, chunk.id)
            _log_rejections(chunk.id, admission)
            per_chunk.append((chunk.id, admission))

        return _assemble_result(per_chunk, malformed_chunk_extractions)

    async def _extract_chunk(
        self, chunk: Chunk, system_prompt: str
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        result = await self._llm.complete_json(
            messages=[{"role": "user", "content": chunk.text}],
            schema=ChunkExtractionResult,
            system=system_prompt,
            temperature=0.0,
        )
        return result.nodes, result.relationships

    def _validate(
        self,
        nodes: list[ExtractedNode],
        rels: list[ExtractedRelationship],
        schema: OntologySchema,
        chunk_id: str | None = None,
    ) -> SchemaAdmission:
        return validate_extraction(nodes, rels, schema, chunk_id)

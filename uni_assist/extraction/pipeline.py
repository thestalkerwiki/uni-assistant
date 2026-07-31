"""Orchestrate the deterministic evidence extraction pipeline."""

from typing import Any

from uni_assist.extraction.context_builder import (
    build_contextual_blocks,
)
from uni_assist.extraction.evidence_candidate import (
    build_evidence_candidates,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
    GroundedEvidenceCandidate,
    ground_evidence_candidates,
)


def run_grounded_evidence_pipeline(
    structured_blocks: list[dict[str, Any]],
    source: EvidenceSourceContext,
) -> list[GroundedEvidenceCandidate]:
    """
    Build grounded evidence candidates from structured source blocks.

    The pipeline performs no semantic category inference.
    """

    contextual_blocks = build_contextual_blocks(
        structured_blocks
    )

    evidence_candidates = build_evidence_candidates(
        contextual_blocks
    )

    return ground_evidence_candidates(
        candidates=evidence_candidates,
        source=source,
    )

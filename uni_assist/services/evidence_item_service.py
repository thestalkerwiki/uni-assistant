"""Build domain evidence items from classified source facts."""

from pydantic import BaseModel, ConfigDict

from uni_assist.domain.evidence import (
    EvidenceConfidence,
    EvidenceItem,
)
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
)
from uni_assist.extraction.classification_join import (
    ClassifiedEvidenceCandidate,
)


class EvidenceBuildResult(BaseModel):
    """
    Result of converting classified candidates into domain evidence.

    Unresolved candidates remain available for later analysis.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    evidence_items: list[EvidenceItem]
    unresolved_candidates: list[
        ClassifiedEvidenceCandidate
    ]


def build_evidence_items(
    candidates: list[ClassifiedEvidenceCandidate],
    confidence: EvidenceConfidence,
) -> EvidenceBuildResult:
    """
    Convert resolved classified candidates into EvidenceItem objects.

    Source-derived text and provenance are preserved unchanged.
    Unresolved candidates are not converted or discarded.
    """

    evidence_items: list[EvidenceItem] = []
    unresolved_candidates: list[
        ClassifiedEvidenceCandidate
    ] = []

    for candidate in candidates:
        if (
            candidate.classification_status
            == ClassificationStatus.UNRESOLVED
        ):
            unresolved_candidates.append(candidate)
            continue

        if candidate.category is None:
            raise ValueError(
                "Resolved candidate requires a category."
            )

        evidence_items.append(
            EvidenceItem(
                id=candidate.candidate_id,
                category=candidate.category,
                label=candidate.raw_label,
                value=candidate.raw_value,
                confidence=confidence,
                source_id=candidate.source_id,
                source_url=candidate.source_url,
                source_type=candidate.source_type,
                source_language=(
                    candidate.source_language
                ),
                raw_text=candidate.raw_text,
                source_locator=(
                    candidate.source_locator
                ),
            )
        )

    return EvidenceBuildResult(
        evidence_items=evidence_items,
        unresolved_candidates=unresolved_candidates,
    )

import pytest
from pydantic import ValidationError

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
    validate_classification_batch,
)
from uni_assist.extraction.evidence_candidate import (
    EvidenceCandidate,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
    ground_evidence_candidates,
)


def build_grounded_candidates():
    candidates = [
        EvidenceCandidate(
            raw_label="Study duration",
            raw_value="6 semesters",
            raw_text="Study duration: 6 semesters",
            section_path=[
                "Programme overview",
            ],
            source_index=2,
        ),
        EvidenceCandidate(
            raw_label="Additional condition",
            raw_value="See programme regulations",
            raw_text=(
                "Additional condition: "
                "See programme regulations"
            ),
            section_path=[
                "Admission",
            ],
            source_index=5,
        ),
    ]

    source = EvidenceSourceContext(
        source_id="source-123",
        source_url="https://example.edu/programme",
        source_type="webpage",
        source_language="en",
    )

    return ground_evidence_candidates(
        candidates=candidates,
        source=source,
    )


def test_accepts_resolved_and_unresolved_results() -> None:
    candidates = build_grounded_candidates()

    classifications = [
        EvidenceClassification(
            candidate_id=candidates[0].candidate_id,
            status=ClassificationStatus.RESOLVED,
            category=EvidenceCategory.DURATION_WORKLOAD,
        ),
        EvidenceClassification(
            candidate_id=candidates[1].candidate_id,
            status=ClassificationStatus.UNRESOLVED,
        ),
    ]

    validated = validate_classification_batch(
        candidates=candidates,
        classifications=classifications,
    )

    assert validated == classifications


def test_rejects_invalid_or_incomplete_results() -> None:
    candidates = build_grounded_candidates()

    with pytest.raises(ValidationError):
        EvidenceClassification(
            candidate_id=candidates[0].candidate_id,
            status=ClassificationStatus.RESOLVED,
        )

    with pytest.raises(ValidationError):
        EvidenceClassification(
            candidate_id=candidates[0].candidate_id,
            status=ClassificationStatus.UNRESOLVED,
            category=EvidenceCategory.DURATION_WORKLOAD,
        )

    incomplete_results = [
        EvidenceClassification(
            candidate_id=candidates[0].candidate_id,
            status=ClassificationStatus.RESOLVED,
            category=EvidenceCategory.DURATION_WORKLOAD,
        ),
    ]

    with pytest.raises(ValueError):
        validate_classification_batch(
            candidates=candidates,
            classifications=incomplete_results,
        )

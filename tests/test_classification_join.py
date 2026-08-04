import pytest

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.extraction.classification_join import (
    attach_classifications,
)
from uni_assist.extraction.evidence_candidate import (
    EvidenceCandidate,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
    ground_evidence_candidates,
)


def build_candidates():
    source = EvidenceSourceContext(
        source_id="source-123",
        source_url="https://example.edu/programme",
        source_type="webpage",
        source_language="de",
    )

    candidates = [
        EvidenceCandidate(
            raw_label="Studiendauer",
            raw_value="6 Semester",
            raw_text="Studiendauer: 6 Semester",
            section_path=["Studienübersicht"],
            source_index=2,
        ),
        EvidenceCandidate(
            raw_label="Weitere Information",
            raw_value="Siehe Studienplan",
            raw_text=(
                "Weitere Information: "
                "Siehe Studienplan"
            ),
            section_path=["Zulassung"],
            source_index=5,
        ),
    ]

    return ground_evidence_candidates(
        candidates=candidates,
        source=source,
    )


def test_attaches_results_by_candidate_id() -> None:
    candidates = build_candidates()

    # Deliberately return classifications in reverse order.
    classifications = [
        EvidenceClassification(
            candidate_id=candidates[1].candidate_id,
            status=ClassificationStatus.UNRESOLVED,
        ),
        EvidenceClassification(
            candidate_id=candidates[0].candidate_id,
            status=ClassificationStatus.RESOLVED,
            category=EvidenceCategory.DURATION_WORKLOAD,
        ),
    ]

    results = attach_classifications(
        candidates=candidates,
        classifications=classifications,
    )

    assert results[0].candidate_id == (
        candidates[0].candidate_id
    )
    assert results[0].category == (
        EvidenceCategory.DURATION_WORKLOAD
    )

    assert results[1].candidate_id == (
        candidates[1].candidate_id
    )
    assert results[1].classification_status == (
        ClassificationStatus.UNRESOLVED
    )
    assert results[1].category is None

    # Source facts remain untouched.
    assert results[0].raw_value == "6 Semester"
    assert results[0].raw_text == (
        "Studiendauer: 6 Semester"
    )
    assert results[0].source_id == "source-123"


def test_rejects_incomplete_classification_batch() -> None:
    candidates = build_candidates()

    incomplete = [
        EvidenceClassification(
            candidate_id=candidates[0].candidate_id,
            status=ClassificationStatus.RESOLVED,
            category=EvidenceCategory.DURATION_WORKLOAD,
        ),
    ]

    with pytest.raises(ValueError):
        attach_classifications(
            candidates=candidates,
            classifications=incomplete,
        )

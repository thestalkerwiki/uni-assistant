import pytest

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.extraction.classification_request import (
    EvidenceClassificationRequest,
)
from uni_assist.extraction.evidence_candidate import (
    EvidenceCandidate,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
    ground_evidence_candidates,
)
from uni_assist.services.classification_service import (
    classify_grounded_candidates,
)


def build_grounded_candidates():
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
            section_path=[
                "Studienübersicht",
            ],
            source_index=2,
        ),
        EvidenceCandidate(
            raw_label="Zusätzliche Information",
            raw_value="Siehe Studienplan",
            raw_text=(
                "Zusätzliche Information: "
                "Siehe Studienplan"
            ),
            section_path=[
                "Zulassung",
            ],
            source_index=5,
        ),
    ]

    return ground_evidence_candidates(
        candidates=candidates,
        source=source,
    )


class FakeClassifier:
    def __init__(self) -> None:
        self.received_request = None

    def classify(
        self,
        request: EvidenceClassificationRequest,
    ) -> list[EvidenceClassification]:
        self.received_request = request

        return [
            EvidenceClassification(
                candidate_id=(
                    request.candidates[0].candidate_id
                ),
                status=ClassificationStatus.RESOLVED,
                category=(
                    EvidenceCategory.DURATION_WORKLOAD
                ),
            ),
            EvidenceClassification(
                candidate_id=(
                    request.candidates[1].candidate_id
                ),
                status=ClassificationStatus.UNRESOLVED,
            ),
        ]


class IncompleteClassifier:
    def classify(
        self,
        request: EvidenceClassificationRequest,
    ) -> list[EvidenceClassification]:
        return [
            EvidenceClassification(
                candidate_id=(
                    request.candidates[0].candidate_id
                ),
                status=ClassificationStatus.RESOLVED,
                category=(
                    EvidenceCategory.DURATION_WORKLOAD
                ),
            ),
        ]


def test_classifies_candidates_through_complete_service() -> None:
    candidates = build_grounded_candidates()
    classifier = FakeClassifier()

    results = classify_grounded_candidates(
        candidates=candidates,
        classifier=classifier,
    )

    assert len(results) == 2

    assert results[0].category == (
        EvidenceCategory.DURATION_WORKLOAD
    )

    assert results[1].status == (
        ClassificationStatus.UNRESOLVED
    )

    assert classifier.received_request is not None
    assert len(
        classifier.received_request.candidates
    ) == 2


def test_rejects_incomplete_classifier_response() -> None:
    candidates = build_grounded_candidates()

    with pytest.raises(ValueError):
        classify_grounded_candidates(
            candidates=candidates,
            classifier=IncompleteClassifier(),
        )

from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
    EvidenceImportance,
    EvidenceStage,
)
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
from uni_assist.services.evidence_item_service import (
    build_evidence_items,
)


def build_classified_candidates():
    source = EvidenceSourceContext(
        source_id="source-123",
        source_url="https://example.edu/programme",
        source_type="webpage",
        source_language="de",
    )

    raw_candidates = [
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

    grounded = ground_evidence_candidates(
        candidates=raw_candidates,
        source=source,
    )

    classifications = [
        EvidenceClassification(
            candidate_id=grounded[0].candidate_id,
            status=ClassificationStatus.RESOLVED,
            category=(
                EvidenceCategory.DURATION_WORKLOAD
            ),
        ),
        EvidenceClassification(
            candidate_id=grounded[1].candidate_id,
            status=ClassificationStatus.UNRESOLVED,
        ),
    ]

    return attach_classifications(
        candidates=grounded,
        classifications=classifications,
    )


def test_builds_evidence_item_without_changing_fact() -> None:
    candidates = build_classified_candidates()

    result = build_evidence_items(
        candidates=candidates,
        confidence=EvidenceConfidence.HIGH,
    )

    assert len(result.evidence_items) == 1

    evidence = result.evidence_items[0]
    candidate = candidates[0]

    assert evidence.id == candidate.candidate_id
    assert evidence.category == (
        EvidenceCategory.DURATION_WORKLOAD
    )

    assert evidence.label == "Studiendauer"
    assert evidence.value == "6 Semester"
    assert evidence.raw_text == (
        "Studiendauer: 6 Semester"
    )

    assert evidence.source_id == "source-123"
    assert evidence.source_locator.block_index == 2

    assert evidence.confidence == (
        EvidenceConfidence.HIGH
    )

    assert evidence.stage == (
        EvidenceStage.NOT_SPECIFIED
    )

    assert evidence.importance == (
        EvidenceImportance.MEDIUM
    )

    assert evidence.applicant_action is None


def test_preserves_unresolved_candidate_separately() -> None:
    candidates = build_classified_candidates()

    result = build_evidence_items(
        candidates=candidates,
        confidence=EvidenceConfidence.HIGH,
    )

    assert len(result.evidence_items) == 1
    assert len(result.unresolved_candidates) == 1

    unresolved = result.unresolved_candidates[0]

    assert unresolved.raw_label == (
        "Weitere Information"
    )

    assert unresolved.classification_status == (
        ClassificationStatus.UNRESOLVED
    )

    assert unresolved.category is None

from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
)
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.extraction.classification_request import (
    EvidenceClassificationRequest,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.services.evidence_pipeline_service import (
    build_evidence_from_structured_blocks,
)


class FakeClassifier:
    """
    Test classifier that returns controlled semantic results.
    """

    def classify(
        self,
        request: EvidenceClassificationRequest,
    ) -> list[EvidenceClassification]:
        results: list[EvidenceClassification] = []

        for candidate in request.candidates:
            if candidate.raw_label == "Studiendauer":
                results.append(
                    EvidenceClassification(
                        candidate_id=candidate.candidate_id,
                        status=ClassificationStatus.RESOLVED,
                        category=(
                            EvidenceCategory.DURATION_WORKLOAD
                        ),
                    )
                )
            else:
                results.append(
                    EvidenceClassification(
                        candidate_id=candidate.candidate_id,
                        status=ClassificationStatus.UNRESOLVED,
                    )
                )

        return results


def test_builds_domain_evidence_from_structured_blocks() -> None:
    structured_blocks = [
        {
            "tag": "h1",
            "text": "Psychologie",
        },
        {
            "tag": "h2",
            "text": "Studienübersicht",
        },
        {
            "type": "label_value",
            "tag": "dl",
            "label": "Studiendauer",
            "value": "6 Semester",
            "text": "Studiendauer: 6 Semester",
        },
        {
            "tag": "h2",
            "text": "Zulassung",
        },
        {
            "type": "label_value",
            "tag": "dl",
            "label": "Weitere Information",
            "value": "Siehe Studienplan",
            "text": (
                "Weitere Information: "
                "Siehe Studienplan"
            ),
        },
    ]

    source = EvidenceSourceContext(
        source_id="source-psychology",
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="de",
    )

    result = build_evidence_from_structured_blocks(
        structured_blocks=structured_blocks,
        source=source,
        classifier=FakeClassifier(),
        confidence=EvidenceConfidence.HIGH,
    )

    assert len(result.evidence_items) == 1
    assert len(result.unresolved_candidates) == 1

    evidence = result.evidence_items[0]

    assert evidence.category == (
        EvidenceCategory.DURATION_WORKLOAD
    )
    assert evidence.label == "Studiendauer"
    assert evidence.value == "6 Semester"
    assert evidence.raw_text == (
        "Studiendauer: 6 Semester"
    )

    assert evidence.source_id == "source-psychology"
    assert evidence.source_locator.heading == (
        "Studienübersicht"
    )
    assert evidence.source_locator.block_index == 2


def test_pipeline_preserves_unresolved_source_fact() -> None:
    structured_blocks = [
        {
            "tag": "h2",
            "text": "Zulassung",
        },
        {
            "type": "label_value",
            "tag": "dl",
            "label": "Weitere Information",
            "value": "Siehe Studienplan",
            "text": (
                "Weitere Information: "
                "Siehe Studienplan"
            ),
        },
    ]

    source = EvidenceSourceContext(
        source_id="source-admission",
        source_url="https://example.edu/admission",
        source_type="webpage",
        source_language="de",
    )

    result = build_evidence_from_structured_blocks(
        structured_blocks=structured_blocks,
        source=source,
        classifier=FakeClassifier(),
        confidence=EvidenceConfidence.HIGH,
    )

    assert result.evidence_items == []
    assert len(result.unresolved_candidates) == 1

    unresolved = result.unresolved_candidates[0]

    assert unresolved.raw_label == (
        "Weitere Information"
    )
    assert unresolved.raw_value == (
        "Siehe Studienplan"
    )
    assert unresolved.source_id == (
        "source-admission"
    )

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.classification_input import (
    EvidenceClassificationInput,
)
from uni_assist.extraction.classification_request import (
    build_classification_request,
)


def build_input() -> EvidenceClassificationInput:
    return EvidenceClassificationInput(
        candidate_id="a" * 64,
        raw_label="Studiendauer",
        raw_value="6 Semester",
        raw_text="Studiendauer: 6 Semester",
        section_path=[
            "Psychologie",
            "Studienübersicht",
        ],
        source_language="de",
    )


def test_builds_request_with_all_allowed_categories() -> None:
    classification_input = build_input()

    request = build_classification_request(
        inputs=[classification_input],
    )

    assert request.allowed_categories == list(
        EvidenceCategory
    )

    assert request.candidates == [
        classification_input
    ]


def test_request_preserves_fact_and_limits_model_task() -> None:
    request = build_classification_request(
        inputs=[build_input()],
    )

    payload = request.model_dump(mode="json")
    candidate = payload["candidates"][0]

    assert candidate["candidate_id"] == "a" * 64
    assert candidate["raw_label"] == "Studiendauer"
    assert candidate["raw_value"] == "6 Semester"

    assert "source_url" not in candidate
    assert "source_id" not in candidate
    assert "value" not in payload

    assert "do not rewrite" in (
        request.instruction.casefold()
    )
    assert "unresolved" in (
        request.instruction.casefold()
    )

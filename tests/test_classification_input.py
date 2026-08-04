from uni_assist.extraction.classification_input import (
    build_classification_inputs,
)
from uni_assist.extraction.evidence_candidate import (
    EvidenceCandidate,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
    ground_evidence_candidates,
)


def build_grounded_candidate():
    candidate = EvidenceCandidate(
        raw_label="Studiendauer",
        raw_value="6 Semester",
        raw_text="Studiendauer: 6 Semester",
        section_path=[
            "Psychologie",
            "Studienübersicht",
        ],
        source_index=4,
    )

    source = EvidenceSourceContext(
        source_id="source-123",
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="de",
    )

    return ground_evidence_candidates(
        candidates=[candidate],
        source=source,
    )[0]


def test_builds_minimal_classification_input() -> None:
    candidate = build_grounded_candidate()

    result = build_classification_inputs(
        candidates=[candidate],
    )[0]

    assert result.candidate_id == candidate.candidate_id
    assert result.raw_label == "Studiendauer"
    assert result.raw_value == "6 Semester"
    assert result.section_path == [
        "Psychologie",
        "Studienübersicht",
    ]
    assert result.source_language == "de"


def test_does_not_expose_unnecessary_source_fields() -> None:
    candidate = build_grounded_candidate()

    result = build_classification_inputs(
        candidates=[candidate],
    )[0]

    payload = result.model_dump()

    assert "source_id" not in payload
    assert "source_url" not in payload
    assert "source_type" not in payload
    assert "source_locator" not in payload

from uni_assist.extraction.evidence_candidate import (
    EvidenceCandidate,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
    ground_evidence_candidates,
)


def test_attaches_source_and_locator_to_candidate() -> None:
    candidate = EvidenceCandidate(
        raw_label="Studiendauer",
        raw_value="6 Semester",
        raw_text="Studiendauer: 6 Semester",
        section_path=[
            "Psychologie",
            "Das Studium kurz und knapp",
        ],
        source_index=4,
    )

    source = EvidenceSourceContext(
        source_id="source-123",
        source_url=(
            "https://example.edu/programme/"
            "psychology"
        ),
        source_type="webpage",
        source_language="de",
    )

    grounded = ground_evidence_candidates(
        candidates=[candidate],
        source=source,
    )

    assert len(grounded) == 1

    result = grounded[0]

    assert result.raw_label == "Studiendauer"
    assert result.raw_value == "6 Semester"

    assert result.source_id == "source-123"
    assert result.source_type == "webpage"
    assert result.source_language == "de"

    assert result.source_locator.heading == (
        "Das Studium kurz und knapp"
    )

    assert result.source_locator.block_index == 4


def test_grounding_does_not_invent_missing_heading() -> None:
    candidate = EvidenceCandidate(
        raw_label="Academic degree",
        raw_value="Bachelor of Science",
        raw_text=(
            "Academic degree: "
            "Bachelor of Science"
        ),
        section_path=[],
        source_index=2,
    )

    source = EvidenceSourceContext(
        source_id="source-456",
        source_url="https://example.edu/programme",
        source_type="webpage",
        source_language="en",
    )

    grounded = ground_evidence_candidates(
        candidates=[candidate],
        source=source,
    )

    assert grounded[0].source_locator.heading is None
    assert grounded[0].source_locator.block_index == 2


def test_candidate_identity_is_deterministic() -> None:
    candidate = EvidenceCandidate(
        raw_label="Study duration",
        raw_value="6 semesters",
        raw_text="Study duration: 6 semesters",
        section_path=[
            "Psychology",
            "Programme overview",
        ],
        source_index=4,
    )

    source = EvidenceSourceContext(
        source_id="source-789",
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="en",
    )

    first_result = ground_evidence_candidates(
        candidates=[candidate],
        source=source,
    )[0]

    second_result = ground_evidence_candidates(
        candidates=[candidate],
        source=source,
    )[0]

    assert len(first_result.candidate_id) == 64
    assert (
        first_result.candidate_id
        == second_result.candidate_id
    )


def test_candidate_identity_changes_with_fact_location() -> None:
    first_candidate = EvidenceCandidate(
        raw_label="Study duration",
        raw_value="6 semesters",
        raw_text="Study duration: 6 semesters",
        section_path=["Programme overview"],
        source_index=4,
    )

    second_candidate = EvidenceCandidate(
        raw_label="Study duration",
        raw_value="6 semesters",
        raw_text="Study duration: 6 semesters",
        section_path=["Programme overview"],
        source_index=5,
    )

    source = EvidenceSourceContext(
        source_id="source-789",
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="en",
    )

    grounded = ground_evidence_candidates(
        candidates=[
            first_candidate,
            second_candidate,
        ],
        source=source,
    )

    assert (
        grounded[0].candidate_id
        != grounded[1].candidate_id
    )

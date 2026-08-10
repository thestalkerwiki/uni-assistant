"""Core invariants of the deterministic pre-LLM evidence pipeline."""

from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.extraction.pipeline import (
    run_grounded_evidence_pipeline,
)


def build_blocks() -> list[dict]:
    return [
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
    ]


def build_source(
    source_id: str = "source-123",
) -> EvidenceSourceContext:
    return EvidenceSourceContext(
        source_id=source_id,
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="de",
    )


def test_grounding_preserves_extracted_fact() -> None:
    candidates = run_grounded_evidence_pipeline(
        structured_blocks=build_blocks(),
        source=build_source(),
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.raw_label == "Studiendauer"
    assert candidate.raw_value == "6 Semester"
    assert candidate.raw_text == "Studiendauer: 6 Semester"

    assert candidate.section_path == [
        "Psychologie",
        "Studienübersicht",
    ]

    assert candidate.source_locator.heading == (
        "Studienübersicht"
    )
    assert candidate.source_locator.block_index == 2


def test_candidate_identity_is_deterministic() -> None:
    first = run_grounded_evidence_pipeline(
        structured_blocks=build_blocks(),
        source=build_source(),
    )

    second = run_grounded_evidence_pipeline(
        structured_blocks=build_blocks(),
        source=build_source(),
    )

    assert first[0].candidate_id == second[0].candidate_id


def test_same_fact_in_different_sources_has_different_identity() -> None:
    first = run_grounded_evidence_pipeline(
        structured_blocks=build_blocks(),
        source=build_source("source-A"),
    )

    second = run_grounded_evidence_pipeline(
        structured_blocks=build_blocks(),
        source=build_source("source-B"),
    )

    assert first[0].raw_label == second[0].raw_label
    assert first[0].raw_value == second[0].raw_value

    assert first[0].candidate_id != second[0].candidate_id

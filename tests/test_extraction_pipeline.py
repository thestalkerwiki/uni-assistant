from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.extraction.pipeline import (
    run_grounded_evidence_pipeline,
)


def test_builds_grounded_candidates_from_structured_blocks() -> None:
    structured_blocks = [
        {
            "tag": "h1",
            "text": "Psychology",
        },
        {
            "tag": "h2",
            "text": "Programme overview",
        },
        {
            "type": "label_value",
            "tag": "dl",
            "label": "Study duration",
            "value": "6 semesters",
            "text": "Study duration: 6 semesters",
        },
        {
            "tag": "p",
            "text": (
                "The programme provides a broad "
                "scientific education."
            ),
        },
        {
            "tag": "h2",
            "text": "Admission",
        },
        {
            "type": "label_value",
            "tag": "li",
            "label": "Application deadline",
            "value": "30 April 2026",
            "text": "Application deadline: 30 April 2026",
        },
    ]

    source = EvidenceSourceContext(
        source_id="source-psychology",
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="en",
    )

    results = run_grounded_evidence_pipeline(
        structured_blocks=structured_blocks,
        source=source,
    )

    assert len(results) == 2

    duration = results[0]
    deadline = results[1]

    assert duration.raw_label == "Study duration"
    assert duration.raw_value == "6 semesters"
    assert duration.section_path == [
        "Psychology",
        "Programme overview",
    ]
    assert duration.source_locator.heading == (
        "Programme overview"
    )
    assert duration.source_locator.block_index == 2

    assert deadline.raw_label == "Application deadline"
    assert deadline.section_path == [
        "Psychology",
        "Admission",
    ]
    assert deadline.source_locator.heading == "Admission"
    assert deadline.source_locator.block_index == 5

    assert duration.source_id == "source-psychology"
    assert deadline.source_id == "source-psychology"

    assert len(duration.candidate_id) == 64
    assert len(deadline.candidate_id) == 64
    assert duration.candidate_id != deadline.candidate_id

from uni_assist.extraction.evidence_candidate import (
    build_evidence_candidates,
)


def test_builds_candidate_from_contextual_label_value() -> None:
    contextual_blocks = [
        {
            "type": "label_value",
            "tag": "dl",
            "label": "Study duration",
            "value": "6 semesters",
            "text": "Study duration: 6 semesters",
            "source_index": 4,
            "section_path": [
                "Psychology",
                "Programme overview",
            ],
        }
    ]

    candidates = build_evidence_candidates(
        contextual_blocks
    )

    assert len(candidates) == 1

    assert candidates[0].model_dump() == {
        "raw_label": "Study duration",
        "raw_value": "6 semesters",
        "raw_text": "Study duration: 6 semesters",
        "section_path": [
            "Psychology",
            "Programme overview",
        ],
        "source_index": 4,
        "extraction_method": "html_label_value",
    }


def test_ignores_blocks_without_reliable_label_value_structure() -> None:
    contextual_blocks = [
        {
            "tag": "p",
            "text": (
                "The programme contains several "
                "different subject areas."
            ),
            "source_index": 8,
            "section_path": [
                "Programme content",
            ],
        },
        {
            "type": "label_value",
            "label": "",
            "value": "180 ECTS",
            "text": "180 ECTS",
            "source_index": 9,
            "section_path": [
                "Programme overview",
            ],
        },
    ]

    candidates = build_evidence_candidates(
        contextual_blocks
    )

    assert candidates == []

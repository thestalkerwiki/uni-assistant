from uni_assist.extraction.context_builder import (
    build_contextual_blocks,
)


def test_adds_heading_path_to_content_blocks() -> None:
    blocks = [
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
    ]

    result = build_contextual_blocks(blocks)

    assert result == [
        {
            "type": "label_value",
            "tag": "dl",
            "label": "Study duration",
            "value": "6 semesters",
            "text": "Study duration: 6 semesters",
            "source_index": 2,
            "section_path": [
                "Psychology",
                "Programme overview",
            ],
        }
    ]


def test_new_heading_replaces_deeper_section_context() -> None:
    blocks = [
        {
            "tag": "h1",
            "text": "Psychology",
        },
        {
            "tag": "h2",
            "text": "Admission",
        },
        {
            "tag": "h3",
            "text": "Deadlines",
        },
        {
            "tag": "p",
            "text": "Applications close in May.",
        },
        {
            "tag": "h2",
            "text": "Programme content",
        },
        {
            "tag": "p",
            "text": "The programme contains 180 ECTS.",
        },
    ]

    result = build_contextual_blocks(blocks)

    assert result[0]["section_path"] == [
        "Psychology",
        "Admission",
        "Deadlines",
    ]

    assert result[1]["section_path"] == [
        "Psychology",
        "Programme content",
    ]

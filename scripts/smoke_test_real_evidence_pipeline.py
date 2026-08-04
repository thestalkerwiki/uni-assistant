"""Manual smoke test for the real LLM-backed evidence pipeline."""

from uni_assist.domain import evidence
from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.integrations.openai_classifier import (
    OpenAIEvidenceClassifier,
)
from uni_assist.services.evidence_pipeline_service import (
    build_evidence_from_structured_blocks,
)


def main() -> None:
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
    ]

    source = EvidenceSourceContext(
        source_id="manual-smoke-source",
        source_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="de",
    )

    result = build_evidence_from_structured_blocks(
        structured_blocks=structured_blocks,
        source=source,
        classifier=OpenAIEvidenceClassifier(
            model_name="gpt-5.5",
        ),
        confidence=EvidenceConfidence.HIGH,
    )

    if len(result.evidence_items) != 1:
        raise RuntimeError(
            "Expected exactly one resolved EvidenceItem."
        )

    if result.unresolved_candidates:
        raise RuntimeError(
            "Expected no unresolved candidates."
        )

    evidence = result.evidence_items[0]

    if evidence.category != EvidenceCategory.DURATION_WORKLOAD:
        raise RuntimeError(
            "Unexpected category: "
            f"{evidence.category.value}"
        )

    if evidence.value != "6 Semester":
        raise RuntimeError(
            "The original source value was not preserved."
        )
        
    if evidence.raw_text != "Studiendauer: 6 Semester":
        raise RuntimeError(
            "The original raw source text was not preserved: "
            f"{evidence.raw_text!r}"
        )

    if evidence.source_id != "manual-smoke-source":
        raise RuntimeError(
            "Source provenance was not preserved."
        )
        
    print("RAW TEXT:", repr(evidence.raw_text))
    print("RAW BYTES:", list(evidence.raw_text.encode("utf-8")))

    print("REAL EVIDENCE PIPELINE: PASS")
    print(evidence.model_dump(mode="json"))


if __name__ == "__main__":
    main()

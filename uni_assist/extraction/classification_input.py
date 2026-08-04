"""Build minimal inputs for semantic evidence classification."""

from pydantic import BaseModel, ConfigDict, Field

from uni_assist.extraction.evidence_grounding import (
    GroundedEvidenceCandidate,
)


class EvidenceClassificationInput(BaseModel):
    """
    Minimal grounded data required for semantic classification.

    The model receives enough context to classify the fact,
    but it does not receive authority to modify source provenance.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    candidate_id: str = Field(
        min_length=64,
        max_length=64,
    )

    raw_label: str = Field(min_length=1)
    raw_value: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)

    section_path: list[str]
    source_language: str = Field(min_length=1)


def build_classification_inputs(
    candidates: list[GroundedEvidenceCandidate],
) -> list[EvidenceClassificationInput]:
    """
    Convert grounded candidates into minimal classification inputs.

    Candidate identity and source text are preserved unchanged.
    """

    return [
        EvidenceClassificationInput(
            candidate_id=candidate.candidate_id,
            raw_label=candidate.raw_label,
            raw_value=candidate.raw_value,
            raw_text=candidate.raw_text,
            section_path=candidate.section_path,
            source_language=candidate.source_language,
        )
        for candidate in candidates
    ]

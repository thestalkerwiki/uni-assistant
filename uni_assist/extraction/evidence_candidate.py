"""Build deterministic evidence candidates from contextual blocks."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceCandidate(BaseModel):
    """
    A source-grounded fact candidate created before semantic
    normalization into a final EvidenceItem.
    """

    raw_label: str = Field(min_length=1)
    raw_value: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)

    section_path: list[str] = Field(default_factory=list)
    source_index: int = Field(ge=0)

    extraction_method: Literal[
        "html_label_value"
    ] = "html_label_value"


def _clean_text(value: Any) -> str:
    """Convert a value to normalized single-line text."""

    if value is None:
        return ""

    return " ".join(str(value).split())


def build_evidence_candidates(
    contextual_blocks: list[dict[str, Any]],
) -> list[EvidenceCandidate]:
    """
    Convert reliable label-value blocks into evidence candidates.

    Ordinary paragraphs are deliberately ignored because they still
    require semantic interpretation.
    """

    candidates: list[EvidenceCandidate] = []

    for block in contextual_blocks:
        if block.get("type") != "label_value":
            continue

        label = _clean_text(block.get("label"))
        value = _clean_text(block.get("value"))

        if not label or not value:
            continue

        source_index = block.get("source_index")

        if (
            not isinstance(source_index, int)
            or source_index < 0
        ):
            continue

        section_path = [
            cleaned_heading
            for heading in block.get("section_path", [])
            if (cleaned_heading := _clean_text(heading))
        ]

        raw_text = _clean_text(block.get("text"))

        if not raw_text:
            raw_text = f"{label}: {value}"

        candidates.append(
            EvidenceCandidate(
                raw_label=label,
                raw_value=value,
                raw_text=raw_text,
                section_path=section_path,
                source_index=source_index,
            )
        )

    return candidates

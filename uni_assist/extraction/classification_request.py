"""Build a constrained semantic classification request."""

from pydantic import BaseModel, ConfigDict

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.classification_input import (
    EvidenceClassificationInput,
)


CLASSIFICATION_INSTRUCTION = """
Classify each evidence candidate according to its semantic meaning.

Use the raw label, raw value, raw text, section path, and source language.

For every candidate:
- preserve candidate_id exactly;
- choose one category only when the meaning is sufficiently clear;
- otherwise return unresolved;
- do not rewrite, translate, correct, or invent source facts;
- return exactly one result for every input candidate.
""".strip()


class EvidenceClassificationRequest(BaseModel):
    """
    Complete constrained input for the semantic classifier.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    instruction: str
    allowed_categories: list[EvidenceCategory]
    candidates: list[EvidenceClassificationInput]


def build_classification_request(
    inputs: list[EvidenceClassificationInput],
) -> EvidenceClassificationRequest:
    """
    Build a language-independent request with a closed category set.
    """

    return EvidenceClassificationRequest(
        instruction=CLASSIFICATION_INSTRUCTION,
        allowed_categories=list(EvidenceCategory),
        candidates=inputs,
    )

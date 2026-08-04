"""Strict contract for semantic classification of grounded evidence."""

from collections import Counter
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.evidence_grounding import (
    GroundedEvidenceCandidate,
)


class ClassificationStatus(str, Enum):
    """
    Whether the semantic category was resolved safely.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class EvidenceClassification(BaseModel):
    """
    Semantic assignment linked to an existing grounded candidate.

    The classification does not contain or rewrite factual source data.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    candidate_id: str = Field(
        min_length=64,
        max_length=64,
    )

    status: ClassificationStatus
    category: Optional[EvidenceCategory] = None

    @model_validator(mode="after")
    def validate_resolution_state(
        self,
    ) -> "EvidenceClassification":
        if (
            self.status == ClassificationStatus.RESOLVED
            and self.category is None
        ):
            raise ValueError(
                "Resolved classification requires a category."
            )

        if (
            self.status == ClassificationStatus.UNRESOLVED
            and self.category is not None
        ):
            raise ValueError(
                "Unresolved classification must not contain a category."
            )

        return self


def validate_classification_batch(
    candidates: list[GroundedEvidenceCandidate],
    classifications: list[EvidenceClassification],
) -> list[EvidenceClassification]:
    """
    Ensure that classifications correspond exactly to the input candidates.

    Every candidate must receive one result. The result may be unresolved,
    but it cannot be omitted, duplicated, or linked to an unknown candidate.
    """

    candidate_ids = {
        candidate.candidate_id
        for candidate in candidates
    }

    classification_ids = [
        classification.candidate_id
        for classification in classifications
    ]

    duplicate_ids = {
        candidate_id
        for candidate_id, count
        in Counter(classification_ids).items()
        if count > 1
    }

    if duplicate_ids:
        raise ValueError(
            "Duplicate classification candidate IDs: "
            f"{sorted(duplicate_ids)}"
        )

    unknown_ids = (
        set(classification_ids) - candidate_ids
    )

    if unknown_ids:
        raise ValueError(
            "Classification contains unknown candidate IDs: "
            f"{sorted(unknown_ids)}"
        )

    missing_ids = (
        candidate_ids - set(classification_ids)
    )

    if missing_ids:
        raise ValueError(
            "Classification is missing candidate IDs: "
            f"{sorted(missing_ids)}"
        )

    return classifications

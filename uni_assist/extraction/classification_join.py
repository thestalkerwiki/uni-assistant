"""Join semantic classifications with grounded evidence candidates."""

from typing import Optional

from pydantic import ConfigDict, model_validator

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
    validate_classification_batch,
)
from uni_assist.extraction.evidence_grounding import (
    GroundedEvidenceCandidate,
)


class ClassifiedEvidenceCandidate(
    GroundedEvidenceCandidate
):
    """
    A grounded source fact with an attached semantic result.

    The original factual and provenance fields remain unchanged.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    classification_status: ClassificationStatus
    category: Optional[EvidenceCategory] = None

    @model_validator(mode="after")
    def validate_classification_state(
        self,
    ) -> "ClassifiedEvidenceCandidate":
        if (
            self.classification_status
            == ClassificationStatus.RESOLVED
            and self.category is None
        ):
            raise ValueError(
                "Resolved candidate requires a category."
            )

        if (
            self.classification_status
            == ClassificationStatus.UNRESOLVED
            and self.category is not None
        ):
            raise ValueError(
                "Unresolved candidate must not contain a category."
            )

        return self


def attach_classifications(
    candidates: list[GroundedEvidenceCandidate],
    classifications: list[EvidenceClassification],
) -> list[ClassifiedEvidenceCandidate]:
    """
    Attach validated classifications to their original candidates.

    Candidate order and all source-derived fields are preserved.
    """

    validate_classification_batch(
        candidates=candidates,
        classifications=classifications,
    )

    classifications_by_id = {
        classification.candidate_id: classification
        for classification in classifications
    }

    results: list[ClassifiedEvidenceCandidate] = []

    for candidate in candidates:
        classification = classifications_by_id[
            candidate.candidate_id
        ]

        results.append(
            ClassifiedEvidenceCandidate(
                **candidate.model_dump(),
                classification_status=(
                    classification.status
                ),
                category=classification.category,
            )
        )

    return results

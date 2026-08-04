"""Orchestrate semantic classification of grounded evidence."""

from typing import Protocol

from uni_assist.extraction.classification_contract import (
    EvidenceClassification,
    validate_classification_batch,
)
from uni_assist.extraction.classification_input import (
    build_classification_inputs,
)
from uni_assist.extraction.classification_request import (
    EvidenceClassificationRequest,
    build_classification_request,
)
from uni_assist.extraction.evidence_grounding import (
    GroundedEvidenceCandidate,
)


class EvidenceClassifier(Protocol):
    """
    Interface implemented by any semantic classifier.

    The classifier may use an LLM or another compatible mechanism.
    """

    def classify(
        self,
        request: EvidenceClassificationRequest,
    ) -> list[EvidenceClassification]:
        ...


def classify_grounded_candidates(
    candidates: list[GroundedEvidenceCandidate],
    classifier: EvidenceClassifier,
) -> list[EvidenceClassification]:
    """
    Classify grounded candidates and validate the complete result.

    The classifier receives a constrained request. Its response is
    accepted only when every candidate has exactly one valid result.
    """

    classification_inputs = build_classification_inputs(
        candidates=candidates,
    )

    request = build_classification_request(
        inputs=classification_inputs,
    )

    classifications = classifier.classify(
        request=request,
    )

    return validate_classification_batch(
        candidates=candidates,
        classifications=classifications,
    )

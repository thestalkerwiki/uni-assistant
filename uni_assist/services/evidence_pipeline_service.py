"""Orchestrate the complete source-to-evidence pipeline."""

from typing import Any

from uni_assist.domain.evidence import EvidenceConfidence
from uni_assist.extraction.classification_join import (
    attach_classifications,
)
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.extraction.pipeline import (
    run_grounded_evidence_pipeline,
)
from uni_assist.services.classification_service import (
    EvidenceClassifier,
    classify_grounded_candidates,
)
from uni_assist.services.evidence_item_service import (
    EvidenceBuildResult,
    build_evidence_items,
)


def build_evidence_from_structured_blocks(
    structured_blocks: list[dict[str, Any]],
    source: EvidenceSourceContext,
    classifier: EvidenceClassifier,
    confidence: EvidenceConfidence,
) -> EvidenceBuildResult:
    """
    Transform structured source blocks into domain evidence.

    The function preserves the deterministic extraction chain,
    delegates only semantic categorization to the classifier,
    and keeps unresolved candidates separate.
    """

    grounded_candidates = run_grounded_evidence_pipeline(
        structured_blocks=structured_blocks,
        source=source,
    )

    classifications = classify_grounded_candidates(
        candidates=grounded_candidates,
        classifier=classifier,
    )

    classified_candidates = attach_classifications(
        candidates=grounded_candidates,
        classifications=classifications,
    )

    return build_evidence_items(
        candidates=classified_candidates,
        confidence=confidence,
    )

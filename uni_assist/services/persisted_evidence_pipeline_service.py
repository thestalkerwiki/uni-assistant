"""Orchestrate evidence building and persistence."""

from typing import Any

from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceConfidence
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.services.classification_service import (
    EvidenceClassifier,
)
from uni_assist.services.evidence_item_service import (
    EvidenceBuildResult,
)
from uni_assist.services.evidence_pipeline_service import (
    build_evidence_from_structured_blocks,
)
from uni_assist.storage.evidence_persistence import (
    persist_evidence_item,
)


def build_and_persist_evidence(
    db: Session,
    user_id: str,
    programme_id: str,
    structured_blocks: list[dict[str, Any]],
    source: EvidenceSourceContext,
    classifier: EvidenceClassifier,
    confidence: EvidenceConfidence,
) -> EvidenceBuildResult:
    """
    Build source-grounded evidence and persist resolved items.

    Transaction commit remains the caller's responsibility.
    """

    result = build_evidence_from_structured_blocks(
        structured_blocks=structured_blocks,
        source=source,
        classifier=classifier,
        confidence=confidence,
    )

    for evidence_item in result.evidence_items:
        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme_id,
            evidence_item=evidence_item,
        )

    return result

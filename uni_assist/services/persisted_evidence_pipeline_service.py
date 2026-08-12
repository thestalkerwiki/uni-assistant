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
from uni_assist.storage.repositories import (
    get_source_by_id,
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


def build_and_persist_evidence_for_source(
    db: Session,
    user_id: str,
    programme_id: str,
    source_id: str,
    classifier: EvidenceClassifier,
    confidence: EvidenceConfidence,
) -> EvidenceBuildResult:
    """Build and persist evidence from a stored source snapshot."""

    stored_source = get_source_by_id(
        db=db,
        source_id=source_id,
        user_id=user_id,
    )

    if stored_source is None:
        raise ValueError(
            "Source was not found for this user."
        )

    source_context = EvidenceSourceContext(
        source_id=stored_source.id,
        source_url=stored_source.url,
        source_type=stored_source.source_type,
        source_language=stored_source.source_language,
    )

    return build_and_persist_evidence(
        db=db,
        user_id=user_id,
        programme_id=programme_id,
        structured_blocks=stored_source.structured_blocks,
        source=source_context,
        classifier=classifier,
        confidence=confidence,
    )

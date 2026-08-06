"""Persist domain EvidenceItem objects without losing provenance."""

from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceItem
from uni_assist.storage.models import (
    EvidenceItemModel,
    generate_id,
)
from uni_assist.storage.repositories import (
    attach_source_to_programme,
    get_source_by_id,
)


def persist_evidence_item(
    db: Session,
    user_id: str,
    programme_id: str,
    evidence_item: EvidenceItem,
) -> EvidenceItemModel:
    """
    Persist one domain EvidenceItem.

    Source metadata must match the already persisted source snapshot.
    """

    source = get_source_by_id(
        db=db,
        source_id=evidence_item.source_id,
        user_id=user_id,
    )

    if source is None:
        raise ValueError(
            "Evidence source was not found for this user."
        )

    source_metadata_matches = (
        evidence_item.source_url == source.url
        and evidence_item.source_type == source.source_type
        and evidence_item.source_language
        == source.source_language
    )

    if not source_metadata_matches:
        raise ValueError(
            "Evidence provenance does not match "
            "the persisted source snapshot."
        )

    programme = attach_source_to_programme(
        db=db,
        source_id=source.id,
        programme_id=programme_id,
        user_id=user_id,
    )

    locator = (
        evidence_item.source_locator.model_dump(
            mode="json",
            exclude_none=True,
        )
        if evidence_item.source_locator is not None
        else None
    )

    evidence_model = EvidenceItemModel(
        id=evidence_item.id or generate_id(),
        programme=programme,
        session=source.session,
        source=source,
        category=evidence_item.category.value,
        label=evidence_item.label,
        value=evidence_item.value,
        applicant_action=evidence_item.applicant_action,
        stage=evidence_item.stage.value,
        importance=evidence_item.importance.value,
        confidence=evidence_item.confidence.value,
        source_url=source.url,
        source_type=source.source_type,
        source_language=source.source_language,
        raw_text=evidence_item.raw_text,
        source_locator=locator,
        created_at=evidence_item.created_at,
    )

    db.add(evidence_model)
    db.flush()

    return evidence_model

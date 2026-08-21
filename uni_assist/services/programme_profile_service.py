from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceImportance
from uni_assist.domain.programme import MissingInformationStatus
from uni_assist.services.programme_projection_service import (
    CATEGORY_TO_PROGRAMME_FIELD,
    ProgrammeProjection,
    build_programme_projection,
)
from uni_assist.storage.models import MissingInformationModel
from uni_assist.storage.repositories import get_programme_by_id


def _sync_conflicting_missing_information(
    programme,
    projection: ProgrammeProjection,
) -> None:
    managed_fields = set(
        CATEGORY_TO_PROGRAMME_FIELD.values()
    )

    for existing_item in list(programme.missing_information):
        if (
            existing_item.status
            == MissingInformationStatus.CONFLICTING.value
            and existing_item.field_name in managed_fields
        ):
            programme.missing_information.remove(existing_item)

    for field_name, conflicting_values in projection.conflicts.items():
        relevant_evidence = [
            evidence_item
            for evidence_item in programme.evidence_items
            if CATEGORY_TO_PROGRAMME_FIELD.get(
                evidence_item.category
            )
            == field_name
            and evidence_item.value in conflicting_values
        ]

        searched_source_ids = list(
            dict.fromkeys(
                evidence_item.source_id
                for evidence_item in relevant_evidence
            )
        )

        searched_source_urls = list(
            dict.fromkeys(
                evidence_item.source_url
                for evidence_item in relevant_evidence
            )
        )

        missing_item = MissingInformationModel(
            programme_id=programme.id,
            field_name=field_name,
            status=MissingInformationStatus.CONFLICTING.value,
            reason=(
                f"Conflicting evidence values for {field_name}: "
                + " | ".join(conflicting_values)
            ),
            importance=EvidenceImportance.MEDIUM.value,
            searched_source_ids=searched_source_ids,
            searched_source_urls=searched_source_urls,
        )

        programme.missing_information.append(missing_item)


def refresh_programme_from_evidence(
    db: Session,
    *,
    user_id: str,
    programme_id: str,
) -> ProgrammeProjection:
    """Rebuild programme-level fields from all persisted evidence."""

    programme = get_programme_by_id(
        db=db,
        programme_id=programme_id,
        user_id=user_id,
    )

    if programme is None:
        raise ValueError(
            f"Programme {programme_id!r} was not found "
            f"for user {user_id!r}"
        )

    projection = build_programme_projection(
        programme.evidence_items
    )

    programme.degree = projection.degree
    programme.duration = projection.duration
    programme.credits = projection.credits
    programme.language = projection.language

    _sync_conflicting_missing_information(
        programme,
        projection,
    )

    db.flush()

    return projection
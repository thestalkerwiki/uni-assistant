"""Database repositories for Uni-Assist v2."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from uni_assist.storage.models import (
    EvidenceItemModel,
    ProgrammeModel,
    SearchSessionModel,
    SourceModel,
    UserModel,
)


def get_or_create_user(
    db: Session,
    user_id: str,
) -> UserModel:
    """Return an existing user or create a new local user."""

    user = db.get(UserModel, user_id)

    if user is None:
        user = UserModel(id=user_id)
        db.add(user)
        db.flush()

    return user


def create_search_session(
    db: Session,
    user_id: str,
    query: str,
    urls: list[str],
    output_language: str,
    detected_intent: Optional[str] = None,
) -> SearchSessionModel:
    """Create and stage one persistent user search session."""

    get_or_create_user(db, user_id)

    search_session = SearchSessionModel(
        user_id=user_id,
        query=query,
        urls=urls,
        detected_intent=detected_intent,
        output_language=output_language,
    )

    db.add(search_session)
    db.flush()

    return search_session


def list_search_sessions_by_user(
    db: Session,
    user_id: str,
) -> list[SearchSessionModel]:
    """Return saved search sessions, newest first."""

    statement = (
        select(SearchSessionModel)
        .where(SearchSessionModel.user_id == user_id)
        .order_by(SearchSessionModel.created_at.desc())
    )

    return list(db.execute(statement).scalars().all())

def create_source_record(
    db: Session,
    session_id: str,
    user_id: str,
    url: str,
    normalized_url: str,
    source_type: str,
    source_language: str,
    clean_text: str,
    structured_blocks: list[dict],
    content_hash: Optional[str] = None,
) -> SourceModel:
    """Create one processed source snapshot for a search session."""

    statement = select(SearchSessionModel).where(
        SearchSessionModel.id == session_id,
        SearchSessionModel.user_id == user_id,
    )

    search_session = db.execute(statement).scalar_one_or_none()

    if search_session is None:
        raise ValueError(
            "Search session was not found for this user."
        )

    source = SourceModel(
        session=search_session,
        url=url,
        normalized_url=normalized_url,
        source_type=source_type,
        source_language=source_language,
        clean_text=clean_text,
        structured_blocks=structured_blocks,
        content_hash=content_hash,
    )

    db.add(source)
    db.flush()

    return source


def list_sources_by_session(
    db: Session,
    session_id: str,
    user_id: str,
) -> list[SourceModel]:
    """Return all source snapshots belonging to one user session."""

    statement = (
        select(SourceModel)
        .join(SearchSessionModel)
        .where(
            SourceModel.session_id == session_id,
            SearchSessionModel.user_id == user_id,
        )
        .order_by(SourceModel.created_at.asc())
    )

    return list(db.execute(statement).scalars().all())

def get_source_by_id(
    db: Session,
    source_id: str,
    user_id: str,
) -> Optional[SourceModel]:
    """Return one source only if it belongs to the given user."""

    statement = (
        select(SourceModel)
        .join(SearchSessionModel)
        .where(
            SourceModel.id == source_id,
            SearchSessionModel.user_id == user_id,
        )
    )

    return db.execute(statement).scalar_one_or_none()


def attach_source_to_programme(
    db: Session,
    source_id: str,
    programme_id: str,
    user_id: str,
) -> ProgrammeModel:
    """Attach a processed source to one saved programme."""

    source = get_source_by_id(
        db=db,
        source_id=source_id,
        user_id=user_id,
    )

    if source is None:
        raise ValueError("Source was not found for this user.")

    programme = get_programme_by_id(
        db=db,
        programme_id=programme_id,
        user_id=user_id,
    )

    if programme is None:
        raise ValueError("Programme was not found for this user.")

    if source not in programme.sources:
        programme.sources.append(source)

    db.flush()

    return programme


def create_evidence_item(
    db: Session,
    user_id: str,
    source_id: str,
    programme_id: str,
    category: str,
    label: str,
    value: str,
    stage: str,
    importance: str,
    confidence: str,
    raw_text: str,
    applicant_action: Optional[str] = None,
    source_locator: Optional[dict] = None,
) -> EvidenceItemModel:
    """Create source-grounded evidence for one programme."""

    source = get_source_by_id(
        db=db,
        source_id=source_id,
        user_id=user_id,
    )

    if source is None:
        raise ValueError("Source was not found for this user.")

    programme = attach_source_to_programme(
        db=db,
        source_id=source_id,
        programme_id=programme_id,
        user_id=user_id,
    )

    evidence = EvidenceItemModel(
        programme=programme,
        session=source.session,
        source=source,
        category=category,
        label=label,
        value=value,
        applicant_action=applicant_action,
        stage=stage,
        importance=importance,
        confidence=confidence,
        source_url=source.url,
        source_type=source.source_type,
        source_language=source.source_language,
        raw_text=raw_text,
        source_locator=source_locator,
    )

    db.add(evidence)
    db.flush()

    return evidence

def get_programme_by_id(
    db: Session,
    programme_id: str,
    user_id: str,
) -> Optional[ProgrammeModel]:
    """Return one programme with its accumulated state."""

    statement = (
        select(ProgrammeModel)
        .where(
            ProgrammeModel.id == programme_id,
            ProgrammeModel.user_id == user_id,
        )
        .options(
            selectinload(ProgrammeModel.sources),
            selectinload(ProgrammeModel.evidence_items),
            selectinload(ProgrammeModel.missing_information),
        )
    )

    return db.execute(statement).scalar_one_or_none()


def list_programmes_by_user(
    db: Session,
    user_id: str,
) -> list[ProgrammeModel]:
    """Return all saved programmes belonging to one user."""

    statement = (
        select(ProgrammeModel)
        .where(ProgrammeModel.user_id == user_id)
        .options(
            selectinload(ProgrammeModel.sources),
            selectinload(ProgrammeModel.evidence_items),
            selectinload(ProgrammeModel.missing_information),
        )
        .order_by(ProgrammeModel.updated_at.desc())
    )

    return list(db.execute(statement).scalars().all())
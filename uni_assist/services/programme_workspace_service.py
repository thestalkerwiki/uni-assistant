"""Application service for persistent programme workspaces."""

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceConfidence
from uni_assist.ingestion.web_loader import normalize_urls
from uni_assist.services.classification_service import (
    EvidenceClassifier,
)
from uni_assist.services.source_analysis_service import (
    SourceAnalysisResult,
    analyse_url,
)
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import (
    attach_source_to_programme,
    get_or_create_user,
    get_programme_by_id,
    list_programmes_by_user,
)


@dataclass(frozen=True)
class WorkspaceSourceAdditionResult:
    """
    Result of adding sources to one persistent programme workspace.
    """

    programme: ProgrammeModel
    analyses: list[SourceAnalysisResult]
    skipped_urls: list[str]


def create_programme_workspace(
    db: Session,
    *,
    user_id: str,
    title: str,
    programme_name: Optional[str] = None,
    institution_name: Optional[str] = None,
) -> ProgrammeModel:
    """
    Create and persist one programme workspace.

    The workspace is the long-lived container for sources,
    evidence, conflicts, missing information, and future questions.
    """

    normalized_title = title.strip()

    if not normalized_title:
        raise ValueError(
            "Programme title must not be empty."
        )

    get_or_create_user(
        db=db,
        user_id=user_id,
    )

    programme = ProgrammeModel(
        user_id=user_id,
        title=normalized_title,
        programme_name=(
            programme_name.strip()
            if programme_name
            else None
        ),
        institution_name=(
            institution_name.strip()
            if institution_name
            else None
        ),
    )

    db.add(programme)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(programme)

    return programme


def get_programme_workspace(
    db: Session,
    *,
    user_id: str,
    programme_id: str,
) -> ProgrammeModel:
    """
    Load one programme workspace with its accumulated state.
    """

    programme = get_programme_by_id(
        db=db,
        programme_id=programme_id,
        user_id=user_id,
    )

    if programme is None:
        raise ValueError(
            "Programme workspace was not found for this user."
        )

    return programme


def list_programme_workspaces(
    db: Session,
    *,
    user_id: str,
) -> list[ProgrammeModel]:
    """
    Return all saved programme workspaces for one user.
    """

    return list_programmes_by_user(
        db=db,
        user_id=user_id,
    )


def add_sources_to_programme_workspace(
    db: Session,
    *,
    user_id: str,
    programme_id: str,
    urls: list[str],
    query: str,
    output_language: str,
    classifier: EvidenceClassifier,
    confidence: EvidenceConfidence,
) -> WorkspaceSourceAdditionResult:
    """
    Add previously unknown sources to an existing workspace.

    Source identity is currently based on normalized_url.

    If a URL is already attached to the programme, it is skipped.
    New sources are analysed and contribute evidence to the same
    accumulated programme state.
    """

    programme = get_programme_workspace(
        db=db,
        user_id=user_id,
        programme_id=programme_id,
    )

    normalized_urls = normalize_urls(urls)

    if not normalized_urls:
        raise ValueError(
            "At least one valid source URL is required."
        )

    existing_normalized_urls = {
        source.normalized_url
        for source in programme.sources
    }

    analyses: list[SourceAnalysisResult] = []
    skipped_urls: list[str] = []

    for normalized_url in normalized_urls:
        if normalized_url in existing_normalized_urls:
            skipped_urls.append(
                normalized_url
            )
            continue

        analysis = analyse_url(
            db=db,
            user_id=user_id,
            programme_id=programme_id,
            url=normalized_url,
            query=query,
            output_language=output_language,
            classifier=classifier,
            confidence=confidence,
        )

        analyses.append(analysis)

        # A source belongs to the workspace even if it produced
        # no resolved evidence items.
        attach_source_to_programme(
            db=db,
            source_id=analysis.ingestion.source.id,
            programme_id=programme_id,
            user_id=user_id,
        )

        db.commit()

        existing_normalized_urls.add(
            analysis.ingestion.source.normalized_url
        )

    programme = get_programme_workspace(
        db=db,
        user_id=user_id,
        programme_id=programme_id,
    )

    return WorkspaceSourceAdditionResult(
        programme=programme,
        analyses=analyses,
        skipped_urls=skipped_urls,
    )
"""Application service for analysing web sources end to end."""

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceConfidence
from uni_assist.services.classification_service import (
    EvidenceClassifier,
)
from uni_assist.services.evidence_item_service import (
    EvidenceBuildResult,
)
from uni_assist.services.ingestion_service import (
    IngestionResult,
    ingest_url,
)
from uni_assist.services.persisted_evidence_pipeline_service import (
    build_and_persist_evidence_for_source,
)
from uni_assist.services.programme_profile_service import (
    refresh_programme_from_evidence,
)
from uni_assist.services.programme_projection_service import (
    ProgrammeProjection,
)


@dataclass(frozen=True)
class StoredSourceAnalysisResult:
    """Result of analysing one already persisted source."""

    evidence: EvidenceBuildResult
    programme_projection: ProgrammeProjection


@dataclass(frozen=True)
class SourceAnalysisResult:
    """Combined result of ingestion and source analysis."""

    ingestion: IngestionResult
    evidence: EvidenceBuildResult
    programme_projection: ProgrammeProjection


def analyse_stored_source(
    db: Session,
    *,
    user_id: str,
    programme_id: str,
    source_id: str,
    classifier: EvidenceClassifier,
    confidence: EvidenceConfidence,
) -> StoredSourceAnalysisResult:
    """
    Analyse one persisted source and refresh programme state.

    Evidence and programme changes are committed together.
    """

    try:
        evidence_result = build_and_persist_evidence_for_source(
            db=db,
            user_id=user_id,
            programme_id=programme_id,
            source_id=source_id,
            classifier=classifier,
            confidence=confidence,
        )

        programme_projection = refresh_programme_from_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme_id,
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return StoredSourceAnalysisResult(
        evidence=evidence_result,
        programme_projection=programme_projection,
    )


def analyse_url(
    db: Session,
    *,
    user_id: str,
    programme_id: str,
    url: str,
    query: str,
    output_language: str,
    classifier: EvidenceClassifier,
    confidence: EvidenceConfidence,
    detected_intent: Optional[str] = None,
) -> SourceAnalysisResult:
    """Ingest one URL and analyse its persisted snapshot."""

    ingestion_result = ingest_url(
        db=db,
        user_id=user_id,
        url=url,
        query=query,
        output_language=output_language,
        detected_intent=detected_intent,
    )

    analysis_result = analyse_stored_source(
        db=db,
        user_id=user_id,
        programme_id=programme_id,
        source_id=ingestion_result.source.id,
        classifier=classifier,
        confidence=confidence,
    )

    return SourceAnalysisResult(
        ingestion=ingestion_result,
        evidence=analysis_result.evidence,
        programme_projection=(
            analysis_result.programme_projection
        ),
    )
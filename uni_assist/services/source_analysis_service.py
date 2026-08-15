"""Application service for analysing one web source end to end."""

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


@dataclass(frozen=True)
class SourceAnalysisResult:
    """Combined result of source ingestion and evidence analysis."""

    ingestion: IngestionResult
    evidence: EvidenceBuildResult


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
    """Ingest one URL and build persisted evidence from its snapshot."""

    ingestion_result = ingest_url(
        db=db,
        user_id=user_id,
        url=url,
        query=query,
        output_language=output_language,
        detected_intent=detected_intent,
    )

    try:
        evidence_result = build_and_persist_evidence_for_source(
            db=db,
            user_id=user_id,
            programme_id=programme_id,
            source_id=ingestion_result.source.id,
            classifier=classifier,
            confidence=confidence,
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    return SourceAnalysisResult(
        ingestion=ingestion_result,
        evidence=evidence_result,
    )

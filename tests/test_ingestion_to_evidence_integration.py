"""Integration test from scraped HTML to persisted evidence."""

from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
)
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.ingestion.web_loader import ScrapedWebPage
from uni_assist.services.ingestion_service import ingest_url
from uni_assist.services.persisted_evidence_pipeline_service import (
    build_and_persist_evidence_for_source,
)
from uni_assist.storage.database import Base
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import get_programme_by_id


class FakeClassifier:
    """Resolve test evidence as duration/workload."""

    def classify(self, request):
        return [
            EvidenceClassification(
                candidate_id=candidate.candidate_id,
                status=ClassificationStatus.RESOLVED,
                category=EvidenceCategory.DURATION_WORKLOAD,
            )
            for candidate in request.candidates
        ]


def test_ingests_html_and_persists_evidence(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    html = """
    <html>
        <head>
            <title>Psychology</title>
        </head>
        <body>
            <main>
                <h1>Psychologie</h1>
                <h2>Studienübersicht</h2>

                <dl>
                    <dt>Studiendauer</dt>
                    <dd>6 Semester</dd>
                </dl>
            </main>
        </body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser")

    def fake_scrape_web_page(url: str) -> ScrapedWebPage:
        return ScrapedWebPage(
            requested_url=url,
            source_url=url,
            title="Psychology",
            soup=soup,
        )

    monkeypatch.setattr(
        "uni_assist.services.ingestion_service.scrape_web_page",
        fake_scrape_web_page,
    )

    with Session(engine) as db:
        user_id = "user-123"

        ingestion_result = ingest_url(
            db=db,
            user_id=user_id,
            url="https://example.edu/psychology",
            query="Psychology admission",
            output_language="en",
        )

        stored_source = ingestion_result.source

        assert stored_source.structured_blocks == [
            {
                "tag": "h1",
                "text": "Psychologie",
            },
            {
                "tag": "h2",
                "text": "Studienübersicht",
            },
            {
                "type": "label_value",
                "tag": "dl",
                "label": "Studiendauer",
                "value": "6 Semester",
                "text": "Studiendauer: 6 Semester",
            },
        ]

        programme = ProgrammeModel(
            user_id=user_id,
            title="Psychology",
        )

        db.add(programme)
        db.flush()

        result = build_and_persist_evidence_for_source(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            source_id=stored_source.id,
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert len(result.evidence_items) == 1
        assert result.unresolved_candidates == []

        db.commit()
        db.expire_all()

        loaded_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id=user_id,
        )

        assert loaded_programme is not None
        assert len(loaded_programme.evidence_items) == 1

        evidence = loaded_programme.evidence_items[0]

        assert evidence.category == "duration_workload"
        assert evidence.label == "Studiendauer"
        assert evidence.value == "6 Semester"
        assert evidence.raw_text == (
            "Studiendauer: 6 Semester"
        )
        assert evidence.source_id == stored_source.id

import pytest

from sqlalchemy import create_engine, select

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
from uni_assist.services.source_analysis_service import analyse_url
from uni_assist.storage.database import Base
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import get_programme_by_id
from uni_assist.storage.models import (
    EvidenceItemModel,
    ProgrammeModel,
    SourceModel,
)
from uni_assist.services.persisted_evidence_pipeline_service import (
    build_and_persist_evidence_for_source,
)
from uni_assist.storage.models import (
    EvidenceItemModel,
    ProgrammeModel,
    SourceModel,
)


class FakeClassifier:
    def classify(self, request):
        return [
            EvidenceClassification(
                candidate_id=candidate.candidate_id,
                status=ClassificationStatus.RESOLVED,
                category=EvidenceCategory.DURATION_WORKLOAD,
            )
            for candidate in request.candidates
        ]

class FailingClassifier:
    def classify(self, request):
        raise RuntimeError("Classifier failed")

def test_analyse_url_runs_complete_application_flow(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    html = """
    <html lang="de">
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
        programme = ProgrammeModel(
            user_id="user-123",
            title="Psychology",
        )

        db.add(programme)
        db.commit()

        result = analyse_url(
            db=db,
            user_id="user-123",
            programme_id=programme.id,
            url="https://example.edu/psychology",
            query="Psychology admission",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert result.ingestion.source.source_language == "de"
        assert len(result.evidence.evidence_items) == 1

        loaded_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id="user-123",
        )

        assert loaded_programme is not None
        assert len(loaded_programme.evidence_items) == 1
        assert (
            loaded_programme.evidence_items[0].value
            == "6 Semester"
        )

def test_source_remains_when_evidence_analysis_fails(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    html = """
    <html lang="de">
        <body>
            <main>
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
        programme = ProgrammeModel(
            user_id="user-123",
            title="Psychology",
        )

        db.add(programme)
        db.commit()

        with pytest.raises(
            RuntimeError,
            match="Classifier failed",
        ):
            analyse_url(
                db=db,
                user_id="user-123",
                programme_id=programme.id,
                url="https://example.edu/psychology",
                query="Psychology admission",
                output_language="en",
                classifier=FailingClassifier(),
                confidence=EvidenceConfidence.HIGH,
            )

        sources = list(
            db.execute(
                select(SourceModel)
            ).scalars().all()
        )

        evidence_items = list(
            db.execute(
                select(EvidenceItemModel)
            ).scalars().all()
        )

        assert len(sources) == 1
        assert sources[0].url == (
            "https://example.edu/psychology"
        )

        assert evidence_items == []
        
def test_failed_source_can_be_reanalysed_without_reingestion(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    html = """
    <html lang="de">
        <body>
            <main>
                <dl>
                    <dt>Studiendauer</dt>
                    <dd>6 Semester</dd>
                </dl>
            </main>
        </body>
    </html>
    """

    soup = BeautifulSoup(html, "html.parser")

    scrape_calls = 0

    def fake_scrape_web_page(url: str) -> ScrapedWebPage:
        nonlocal scrape_calls
        scrape_calls += 1

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
        programme = ProgrammeModel(
            user_id="user-123",
            title="Psychology",
        )

        db.add(programme)
        db.commit()

        with pytest.raises(RuntimeError):
            analyse_url(
                db=db,
                user_id="user-123",
                programme_id=programme.id,
                url="https://example.edu/psychology",
                query="Psychology admission",
                output_language="en",
                classifier=FailingClassifier(),
                confidence=EvidenceConfidence.HIGH,
            )

        stored_source = db.execute(
            select(SourceModel)
        ).scalar_one()

        result = build_and_persist_evidence_for_source(
            db=db,
            user_id="user-123",
            programme_id=programme.id,
            source_id=stored_source.id,
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        db.commit()

        assert scrape_calls == 1
        assert len(result.evidence_items) == 1

        evidence_items = list(
            db.execute(
                select(EvidenceItemModel)
            ).scalars().all()
        )

        assert len(evidence_items) == 1
        assert evidence_items[0].value == "6 Semester"
        assert evidence_items[0].source_id == stored_source.id
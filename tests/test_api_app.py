from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from uni_assist.api.app import (
    app,
    get_classifier,
    get_db,
    get_query_interpreter,
)
from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.domain.query import (
    QueryIntent,
    QueryInterpretation,
)
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.ingestion.web_loader import ScrapedWebPage
from uni_assist.storage.database import Base


TEST_HTML = """
<html lang="en">
    <body>
        <main>
            <h1>Computer Science</h1>

            <dl>
                <dt>Duration of study</dt>
                <dd>6 semesters</dd>

                <dt>ECTS credit points</dt>
                <dd>180</dd>

                <dt>Application deadline</dt>
                <dd>15 July</dd>
            </dl>
        </main>
    </body>
</html>
"""


class FakeClassifier:
    def classify(self, request):
        categories = {
            "Duration of study": (
                EvidenceCategory.DURATION_WORKLOAD
            ),
            "ECTS credit points": (
                EvidenceCategory.CREDIT_REQUIREMENT
            ),
            "Application deadline": (
                EvidenceCategory.DEADLINE
            ),
        }

        return [
            EvidenceClassification(
                candidate_id=candidate.candidate_id,
                status=ClassificationStatus.RESOLVED,
                category=categories[candidate.raw_label],
            )
            for candidate in request.candidates
        ]


class FakeQueryInterpreter:
    def __init__(
        self,
        intent: QueryIntent,
    ) -> None:
        self.intent = intent

    def interpret(
        self,
        question: str,
    ) -> QueryInterpretation:
        return QueryInterpretation(
            intent=self.intent
        )


def build_test_client(
    monkeypatch,
    intent: QueryIntent,
) -> TestClient:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    def override_get_db():
        db = TestingSessionLocal()

        try:
            yield db
        finally:
            db.close()

    soup = BeautifulSoup(
        TEST_HTML,
        "html.parser",
    )

    def fake_scrape_web_page(
        url: str,
    ) -> ScrapedWebPage:
        return ScrapedWebPage(
            requested_url=url,
            source_url=url,
            title="Computer Science",
            soup=soup,
        )

    monkeypatch.setattr(
        "uni_assist.services.ingestion_service.scrape_web_page",
        fake_scrape_web_page,
    )

    app.dependency_overrides[get_db] = (
        override_get_db
    )

    app.dependency_overrides[get_classifier] = (
        lambda: FakeClassifier()
    )

    app.dependency_overrides[
        get_query_interpreter
    ] = lambda: FakeQueryInterpreter(intent)

    return TestClient(app)


def test_v2_analyse_endpoint_runs_application_flow(
    monkeypatch,
) -> None:
    client = build_test_client(
        monkeypatch=monkeypatch,
        intent=QueryIntent.OVERVIEW,
    )

    try:
        response = client.post(
            "/api/v2/analyse",
            json={
                "urls": [
                    "https://example.edu/computer-science"
                ],
                "question": (
                    "What should an applicant know "
                    "about this programme?"
                ),
                "programme_title": "Computer Science",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["intent"] == "overview"
        assert data["source_language"] == "en"

        assert (
            data["programme"]["title"]
            == "Computer Science"
        )

        assert (
            data["programme"]["duration"]
            == "6 semesters"
        )

        assert (
            data["programme"]["credits"]
            == "180"
        )

        assert data["conflicts"] == {}

        assert len(data["evidence"]) == 3

        assert {
            item["category"]
            for item in data["evidence"]
        } == {
            "duration_workload",
            "credit_requirement",
            "deadline",
        }

    finally:
        app.dependency_overrides.clear()


def test_deadline_question_filters_response_evidence(
    monkeypatch,
) -> None:
    client = build_test_client(
        monkeypatch=monkeypatch,
        intent=QueryIntent.DEADLINES,
    )

    try:
        response = client.post(
            "/api/v2/analyse",
            json={
                "urls": [
                    "https://example.edu/computer-science"
                ],
                "question": (
                    "What are the application deadlines?"
                ),
                "programme_title": "Computer Science",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["intent"] == "deadlines"

        assert len(data["evidence"]) == 1

        assert (
            data["evidence"][0]["category"]
            == "deadline"
        )

        assert (
            data["evidence"][0]["label"]
            == "Application deadline"
        )

        assert (
            data["evidence"][0]["value"]
            == "15 July"
        )

        assert (
            data["programme"]["duration"]
            == "6 semesters"
        )

        assert (
            data["programme"]["credits"]
            == "180"
        )

    finally:
        app.dependency_overrides.clear()
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from uni_assist.api.app import (
    app,
    get_classifier,
    get_db,
    get_query_interpreter,
)
from uni_assist.domain.evidence import (
    EvidenceCategory,
)
from uni_assist.domain.query import (
    QueryIntent,
    QueryInterpretation,
)
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.ingestion.web_loader import (
    ScrapedWebPage,
)
from uni_assist.storage.database import Base


class FakeClassifier:
    """Deterministic replacement for OpenAI during API tests."""

    def classify(self, request):
        categories = {
            "Duration of study": (
                EvidenceCategory.DURATION_WORKLOAD
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
    """Deterministic query interpreter used instead of OpenAI."""

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
):
    """
    Build one isolated FastAPI application environment.

    The API is real, but network and OpenAI are replaced by fakes.
    """

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(
        engine
    )

    scrape_counter = {
        "calls": 0,
    }

    def override_get_db():
        db = Session(
            engine
        )

        try:
            yield db
        finally:
            db.close()

    def fake_scrape_web_page(
        url: str,
    ) -> ScrapedWebPage:
        scrape_counter["calls"] += 1

        if url == "https://example.edu/programme":
            label = "Duration of study"
            value = "6 semesters"

        elif url == "https://example.edu/deadlines":
            label = "Application deadline"
            value = "15 July"

        else:
            raise AssertionError(
                f"Unexpected URL: {url}"
            )

        html = f"""
        <html lang="en">
            <body>
                <main>
                    <dl>
                        <dt>{label}</dt>
                        <dd>{value}</dd>
                    </dl>
                </main>
            </body>
        </html>
        """

        return ScrapedWebPage(
            requested_url=url,
            source_url=url,
            title="Computer Science",
            soup=BeautifulSoup(
                html,
                "html.parser",
            ),
        )

    monkeypatch.setattr(
        "uni_assist.services.ingestion_service.scrape_web_page",
        fake_scrape_web_page,
    )

    app.dependency_overrides[
        get_db
    ] = override_get_db

    app.dependency_overrides[
        get_classifier
    ] = lambda: FakeClassifier()

    app.dependency_overrides[
        get_query_interpreter
    ] = lambda: FakeQueryInterpreter(
        QueryIntent.DEADLINES
    )

    client = TestClient(
        app
    )

    return client, scrape_counter


def test_workspace_api_accumulates_sources(
    monkeypatch,
) -> None:
    client, _ = build_test_client(
        monkeypatch
    )

    try:
        create_response = client.post(
            "/api/v2/programmes",
            json={
                "title": (
                    "Computer Science · "
                    "Example University"
                ),
                "programme_name": (
                    "Computer Science"
                ),
                "institution_name": (
                    "Example University"
                ),
            },
        )

        assert (
            create_response.status_code
            == 200
        )

        created_workspace = (
            create_response.json()
        )

        programme_id = (
            created_workspace[
                "programme"
            ]["id"]
        )

        assert (
            created_workspace["sources"]
            == []
        )

        assert (
            created_workspace["evidence"]
            == []
        )

        first_add_response = client.post(
            (
                f"/api/v2/programmes/"
                f"{programme_id}/sources"
            ),
            json={
                "urls": [
                    "https://example.edu/programme"
                ],
            },
        )

        assert (
            first_add_response.status_code
            == 200
        )

        first_state = (
            first_add_response.json()
            ["workspace"]
        )

        assert (
            first_state["programme"]["duration"]
            == "6 semesters"
        )

        assert (
            len(first_state["sources"])
            == 1
        )

        assert (
            len(first_state["evidence"])
            == 1
        )

        second_add_response = client.post(
            (
                f"/api/v2/programmes/"
                f"{programme_id}/sources"
            ),
            json={
                "urls": [
                    "https://example.edu/deadlines"
                ],
            },
        )

        assert (
            second_add_response.status_code
            == 200
        )

        second_state = (
            second_add_response.json()
            ["workspace"]
        )

        assert (
            second_state["programme"]["duration"]
            == "6 semesters"
        )

        assert (
            len(second_state["sources"])
            == 2
        )

        assert (
            len(second_state["evidence"])
            == 2
        )

        workspace_response = client.get(
            f"/api/v2/programmes/{programme_id}"
        )

        assert (
            workspace_response.status_code
            == 200
        )

        workspace = (
            workspace_response.json()
        )

        assert (
            workspace["programme"]["duration"]
            == "6 semesters"
        )

        assert (
            len(workspace["sources"])
            == 2
        )

        assert (
            len(workspace["evidence"])
            == 2
        )

    finally:
        app.dependency_overrides.clear()


def test_duplicate_source_is_visible_as_skipped(
    monkeypatch,
) -> None:
    client, scrape_counter = build_test_client(
        monkeypatch
    )

    try:
        create_response = client.post(
            "/api/v2/programmes",
            json={
                "title": "Computer Science",
            },
        )

        programme_id = (
            create_response.json()
            ["programme"]["id"]
        )

        first_response = client.post(
            (
                f"/api/v2/programmes/"
                f"{programme_id}/sources"
            ),
            json={
                "urls": [
                    (
                        "https://example.edu/"
                        "programme#overview"
                    )
                ],
            },
        )

        assert (
            first_response.status_code
            == 200
        )

        second_response = client.post(
            (
                f"/api/v2/programmes/"
                f"{programme_id}/sources"
            ),
            json={
                "urls": [
                    (
                        "https://example.edu/"
                        "programme#requirements"
                    )
                ],
            },
        )

        assert (
            second_response.status_code
            == 200
        )

        duplicate_result = (
            second_response.json()
        )

        assert (
            scrape_counter["calls"]
            == 1
        )

        assert (
            duplicate_result["added_sources"]
            == []
        )

        assert duplicate_result[
            "skipped_urls"
        ] == [
            "https://example.edu/programme"
        ]

        workspace = (
            duplicate_result["workspace"]
        )

        assert (
            len(workspace["sources"])
            == 1
        )

        assert (
            len(workspace["evidence"])
            == 1
        )

    finally:
        app.dependency_overrides.clear()


def test_question_reads_accumulated_state_without_scraping(
    monkeypatch,
) -> None:
    client, scrape_counter = build_test_client(
        monkeypatch
    )

    try:
        create_response = client.post(
            "/api/v2/programmes",
            json={
                "title": "Computer Science",
            },
        )

        programme_id = (
            create_response.json()
            ["programme"]["id"]
        )

        source_response = client.post(
            (
                f"/api/v2/programmes/"
                f"{programme_id}/sources"
            ),
            json={
                "urls": [
                    "https://example.edu/programme",
                    "https://example.edu/deadlines",
                ],
            },
        )

        assert (
            source_response.status_code
            == 200
        )

        assert (
            scrape_counter["calls"]
            == 2
        )

        question_response = client.post(
            (
                f"/api/v2/programmes/"
                f"{programme_id}/questions"
            ),
            json={
                "question": (
                    "What is the application deadline?"
                ),
            },
        )

        assert (
            question_response.status_code
            == 200
        )

        answer = (
            question_response.json()
        )

        assert (
            answer["intent"]
            == "deadlines"
        )

        assert (
            len(answer["evidence"])
            == 1
        )

        assert (
            answer["evidence"][0]["category"]
            == "deadline"
        )

        assert (
            answer["evidence"][0]["value"]
            == "15 July"
        )

        # Asking a question must not download sources again.
        assert (
            scrape_counter["calls"]
            == 2
        )

        workspace_response = client.get(
            f"/api/v2/programmes/{programme_id}"
        )

        workspace = (
            workspace_response.json()
        )

        assert (
            len(workspace["sources"])
            == 2
        )

        assert (
            len(workspace["evidence"])
            == 2
        )

    finally:
        app.dependency_overrides.clear()


def test_saved_programme_appears_in_programme_list(
    monkeypatch,
) -> None:
    client, _ = build_test_client(
        monkeypatch
    )

    try:
        create_response = client.post(
            "/api/v2/programmes",
            json={
                "title": "Computer Science",
            },
        )

        programme_id = (
            create_response.json()
            ["programme"]["id"]
        )

        list_response = client.get(
            "/api/v2/programmes"
        )

        assert (
            list_response.status_code
            == 200
        )

        programmes = (
            list_response.json()
        )

        assert (
            len(programmes)
            == 1
        )

        assert (
            programmes[0]["id"]
            == programme_id
        )

        assert (
            programmes[0]["title"]
            == "Computer Science"
        )

        assert (
            programmes[0]["source_count"]
            == 0
        )

        assert (
            programmes[0]["evidence_count"]
            == 0
        )

    finally:
        app.dependency_overrides.clear()
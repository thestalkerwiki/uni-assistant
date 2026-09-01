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
from uni_assist.services.programme_workspace_service import (
    add_sources_to_programme_workspace,
    create_programme_workspace,
    get_programme_workspace,
    list_programme_workspaces,
)
from uni_assist.storage.database import Base


class FakeClassifier:
    def classify(self, request):
        categories = {
            "Duration of study": (
                EvidenceCategory.DURATION_WORKLOAD
            ),
            "ECTS credit points": (
                EvidenceCategory.CREDIT_REQUIREMENT
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


def test_programme_workspace_survives_new_database_session(
    tmp_path,
) -> None:
    database_path = tmp_path / "workspace.db"

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    user_id = "user-123"

    with Session(engine) as first_session:
        created_programme = create_programme_workspace(
            db=first_session,
            user_id=user_id,
            title="Computer Science · TU Graz",
            programme_name="Computer Science",
            institution_name="TU Graz",
        )

        programme_id = created_programme.id

    with Session(engine) as second_session:
        loaded_programme = get_programme_workspace(
            db=second_session,
            user_id=user_id,
            programme_id=programme_id,
        )

        assert loaded_programme.id == programme_id

        assert (
            loaded_programme.title
            == "Computer Science · TU Graz"
        )

        assert (
            loaded_programme.programme_name
            == "Computer Science"
        )

        assert (
            loaded_programme.institution_name
            == "TU Graz"
        )


def test_lists_saved_programme_workspaces(
    tmp_path,
) -> None:
    database_path = tmp_path / "workspace-list.db"

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    user_id = "user-123"

    with Session(engine) as db:
        first_programme = create_programme_workspace(
            db=db,
            user_id=user_id,
            title="Computer Science · TU Graz",
        )

        second_programme = create_programme_workspace(
            db=db,
            user_id=user_id,
            title="Psychology · Uni Graz",
        )

        programmes = list_programme_workspaces(
            db=db,
            user_id=user_id,
        )

        programme_ids = {
            programme.id
            for programme in programmes
        }

        assert programme_ids == {
            first_programme.id,
            second_programme.id,
        }


def test_programme_workspace_is_private_to_user(
    tmp_path,
) -> None:
    database_path = tmp_path / "workspace-users.db"

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    with Session(engine) as db:
        programme = create_programme_workspace(
            db=db,
            user_id="user-123",
            title="Computer Science",
        )

        try:
            get_programme_workspace(
                db=db,
                user_id="different-user",
                programme_id=programme.id,
            )

        except ValueError:
            pass

        else:
            raise AssertionError(
                "A programme workspace must not be "
                "accessible by another user."
            )


def test_programme_accumulates_state_across_source_additions(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "workspace-accumulation.db"

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    duration_url = (
        "https://example.edu/computer-science/programme"
    )

    credits_url = (
        "https://example.edu/computer-science/credits"
    )

    duration_html = """
    <html lang="en">
        <body>
            <main>
                <dl>
                    <dt>Duration of study</dt>
                    <dd>6 semesters</dd>
                </dl>
            </main>
        </body>
    </html>
    """

    credits_html = """
    <html lang="en">
        <body>
            <main>
                <dl>
                    <dt>ECTS credit points</dt>
                    <dd>180</dd>
                </dl>
            </main>
        </body>
    </html>
    """

    def fake_scrape_web_page(
        url: str,
    ) -> ScrapedWebPage:
        if url == duration_url:
            html = duration_html
        elif url == credits_url:
            html = credits_html
        else:
            raise AssertionError(
                f"Unexpected test URL: {url}"
            )

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

    user_id = "user-123"

    with Session(engine) as first_session:
        programme = create_programme_workspace(
            db=first_session,
            user_id=user_id,
            title="Computer Science · TU Graz",
        )

        programme_id = programme.id

        first_result = add_sources_to_programme_workspace(
            db=first_session,
            user_id=user_id,
            programme_id=programme_id,
            urls=[duration_url],
            query="Programme overview",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert (
            first_result.programme.duration
            == "6 semesters"
        )

        assert first_result.programme.credits is None
        assert len(first_result.programme.sources) == 1
        assert len(first_result.programme.evidence_items) == 1

    with Session(engine) as second_session:
        second_result = add_sources_to_programme_workspace(
            db=second_session,
            user_id=user_id,
            programme_id=programme_id,
            urls=[credits_url],
            query="Programme overview",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert (
            second_result.programme.duration
            == "6 semesters"
        )

        assert (
            second_result.programme.credits
            == "180"
        )

        assert len(second_result.programme.sources) == 2
        assert len(second_result.programme.evidence_items) == 2

    with Session(engine) as third_session:
        persisted_programme = get_programme_workspace(
            db=third_session,
            user_id=user_id,
            programme_id=programme_id,
        )

        assert (
            persisted_programme.duration
            == "6 semesters"
        )

        assert persisted_programme.credits == "180"

        assert len(persisted_programme.sources) == 2
        assert len(persisted_programme.evidence_items) == 2
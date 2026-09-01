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
)
from uni_assist.storage.database import Base


class FakeClassifier:
    """Resolve deterministic test labels into evidence categories."""

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


def build_scraped_page(
    url: str,
    *,
    label: str,
    value: str,
) -> ScrapedWebPage:
    """Build one small deterministic university-like HTML page."""

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


def test_same_source_is_not_added_twice(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = (
        tmp_path / "same-source.db"
    )

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    canonical_url = (
        "https://example.edu/programme"
    )

    scrape_calls = 0

    def fake_scrape_web_page(
        url: str,
    ) -> ScrapedWebPage:
        nonlocal scrape_calls

        scrape_calls += 1

        assert url == canonical_url

        return build_scraped_page(
            url,
            label="Duration of study",
            value="6 semesters",
        )

    monkeypatch.setattr(
        "uni_assist.services.ingestion_service.scrape_web_page",
        fake_scrape_web_page,
    )

    with Session(engine) as db:
        programme = create_programme_workspace(
            db=db,
            user_id="user-123",
            title="Computer Science",
        )

        programme_id = programme.id

        first_result = add_sources_to_programme_workspace(
            db=db,
            user_id="user-123",
            programme_id=programme_id,
            urls=[
                (
                    "https://example.edu/"
                    "programme#overview"
                )
            ],
            query="Programme overview",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        second_result = add_sources_to_programme_workspace(
            db=db,
            user_id="user-123",
            programme_id=programme_id,
            urls=[
                (
                    "https://example.edu/"
                    "programme#requirements"
                )
            ],
            query="Programme overview",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert scrape_calls == 1

        assert len(first_result.analyses) == 1
        assert first_result.skipped_urls == []

        assert second_result.analyses == []
        assert second_result.skipped_urls == [
            canonical_url
        ]

        assert len(
            second_result.programme.sources
        ) == 1

        assert len(
            second_result.programme.evidence_items
        ) == 1

        assert (
            second_result.programme.duration
            == "6 semesters"
        )


def test_two_sources_can_confirm_same_fact(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = (
        tmp_path / "confirmed-fact.db"
    )

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    first_url = (
        "https://example.edu/programme"
    )

    second_url = (
        "https://example.edu/regulations"
    )

    def fake_scrape_web_page(
        url: str,
    ) -> ScrapedWebPage:
        if url not in {
            first_url,
            second_url,
        }:
            raise AssertionError(
                f"Unexpected URL: {url}"
            )

        return build_scraped_page(
            url,
            label="Duration of study",
            value="6 semesters",
        )

    monkeypatch.setattr(
        "uni_assist.services.ingestion_service.scrape_web_page",
        fake_scrape_web_page,
    )

    with Session(engine) as db:
        programme = create_programme_workspace(
            db=db,
            user_id="user-123",
            title="Computer Science",
        )

        result = add_sources_to_programme_workspace(
            db=db,
            user_id="user-123",
            programme_id=programme.id,
            urls=[
                first_url,
                second_url,
            ],
            query="Programme overview",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert (
            result.programme.duration
            == "6 semesters"
        )

        assert len(
            result.programme.sources
        ) == 2

        assert len(
            result.programme.evidence_items
        ) == 2

        assert (
            result.programme.missing_information
            == []
        )


def test_conflicting_sources_create_persistent_conflict(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = (
        tmp_path / "conflicting-fact.db"
    )

    engine = create_engine(
        f"sqlite+pysqlite:///{database_path}"
    )

    Base.metadata.create_all(engine)

    first_url = (
        "https://example.edu/programme"
    )

    second_url = (
        "https://example.edu/regulations"
    )

    def fake_scrape_web_page(
        url: str,
    ) -> ScrapedWebPage:
        if url == first_url:
            value = "6 semesters"

        elif url == second_url:
            value = "4 semesters"

        else:
            raise AssertionError(
                f"Unexpected URL: {url}"
            )

        return build_scraped_page(
            url,
            label="Duration of study",
            value=value,
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
            title="Computer Science",
        )

        programme_id = programme.id

        result = add_sources_to_programme_workspace(
            db=first_session,
            user_id=user_id,
            programme_id=programme_id,
            urls=[
                first_url,
                second_url,
            ],
            query="Programme overview",
            output_language="en",
            classifier=FakeClassifier(),
            confidence=EvidenceConfidence.HIGH,
        )

        assert result.programme.duration is None

        assert len(
            result.programme.sources
        ) == 2

        assert len(
            result.programme.evidence_items
        ) == 2

        conflicts = [
            item
            for item in (
                result.programme.missing_information
            )
            if (
                item.field_name == "duration"
                and item.status == "conflicting"
            )
        ]

        assert len(conflicts) == 1

    with Session(engine) as second_session:
        persisted_programme = (
            get_programme_workspace(
                db=second_session,
                user_id=user_id,
                programme_id=programme_id,
            )
        )

        assert persisted_programme.duration is None

        assert len(
            persisted_programme.sources
        ) == 2

        assert len(
            persisted_programme.evidence_items
        ) == 2

        conflicts = [
            item
            for item in (
                persisted_programme.missing_information
            )
            if (
                item.field_name == "duration"
                and item.status == "conflicting"
            )
        ]

        assert len(conflicts) == 1
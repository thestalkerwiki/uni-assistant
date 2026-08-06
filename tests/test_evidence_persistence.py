import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
    EvidenceItem,
    SourceLocator,
)
from uni_assist.storage.database import Base
from uni_assist.storage.evidence_persistence import (
    persist_evidence_item,
)
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import (
    create_search_session,
    create_source_record,
    get_programme_by_id,
)


def prepare_storage(db: Session):
    user_id = "user-123"

    search_session = create_search_session(
        db=db,
        user_id=user_id,
        query="Psychology admission",
        urls=["https://example.edu/psychology"],
        output_language="en",
    )

    source = create_source_record(
        db=db,
        session_id=search_session.id,
        user_id=user_id,
        url="https://example.edu/psychology",
        normalized_url="https://example.edu/psychology",
        source_type="webpage",
        source_language="de",
        clean_text="Studiendauer: 6 Semester",
        structured_blocks=[],
    )

    programme = ProgrammeModel(
        user_id=user_id,
        title="Psychology",
    )

    db.add(programme)
    db.flush()

    return user_id, source, programme


def test_persists_and_reads_domain_evidence() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        user_id, source, programme = prepare_storage(db)

        evidence_item = EvidenceItem(
            id="a" * 64,
            category=EvidenceCategory.DURATION_WORKLOAD,
            label="Studiendauer",
            value="6 Semester",
            confidence=EvidenceConfidence.HIGH,
            source_id=source.id,
            source_url=source.url,
            source_type=source.source_type,
            source_language=source.source_language,
            raw_text="Studiendauer: 6 Semester",
            source_locator=SourceLocator(
                heading="Studienübersicht",
                block_index=2,
            ),
        )

        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            evidence_item=evidence_item,
        )

        db.commit()
        db.expire_all()

        loaded_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id=user_id,
        )

        assert loaded_programme is not None
        assert len(loaded_programme.evidence_items) == 1

        stored = loaded_programme.evidence_items[0]

        assert stored.id == "a" * 64
        assert stored.category == "duration_workload"
        assert stored.label == "Studiendauer"
        assert stored.value == "6 Semester"
        assert stored.raw_text == (
            "Studiendauer: 6 Semester"
        )
        assert stored.source_id == source.id
        assert stored.source_locator == {
            "heading": "Studienübersicht",
            "block_index": 2,
        }


def test_rejects_mismatched_source_provenance() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        user_id, source, programme = prepare_storage(db)

        evidence_item = EvidenceItem(
            id="b" * 64,
            category=EvidenceCategory.DURATION_WORKLOAD,
            label="Studiendauer",
            value="6 Semester",
            confidence=EvidenceConfidence.HIGH,
            source_id=source.id,
            source_url="https://wrong.example/source",
            source_type=source.source_type,
            source_language=source.source_language,
            raw_text="Studiendauer: 6 Semester",
        )

        with pytest.raises(
            ValueError,
            match="provenance",
        ):
            persist_evidence_item(
                db=db,
                user_id=user_id,
                programme_id=programme.id,
                evidence_item=evidence_item,
            )

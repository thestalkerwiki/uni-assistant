from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
    EvidenceItem,
)
from uni_assist.services.programme_profile_service import (
    refresh_programme_from_evidence,
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


def make_duration_evidence(
    *,
    evidence_id: str,
    value: str,
    raw_text: str,
    source,
) -> EvidenceItem:
    return EvidenceItem(
        id=evidence_id,
        category=EvidenceCategory.DURATION_WORKLOAD,
        label="Studiendauer",
        value=value,
        confidence=EvidenceConfidence.HIGH,
        source_id=source.id,
        source_url=source.url,
        source_type=source.source_type,
        source_language=source.source_language,
        raw_text=raw_text,
    )


def test_refreshes_programme_from_persisted_evidence() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        user_id, source, programme = prepare_storage(db)

        evidence_item = make_duration_evidence(
            evidence_id="a" * 64,
            value="6 Semester",
            raw_text="Studiendauer: 6 Semester",
            source=source,
        )

        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            evidence_item=evidence_item,
        )

        projection = refresh_programme_from_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
        )

        db.commit()
        db.expire_all()

        loaded_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id=user_id,
        )

        assert loaded_programme is not None
        assert projection.duration == "6 Semester"
        assert projection.conflicts == {}
        assert loaded_programme.duration == "6 Semester"


def test_conflict_clears_previous_programme_value() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        user_id, source, programme = prepare_storage(db)

        first_evidence = make_duration_evidence(
            evidence_id="a" * 64,
            value="6 Semester",
            raw_text="Studiendauer: 6 Semester",
            source=source,
        )

        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            evidence_item=first_evidence,
        )

        first_projection = refresh_programme_from_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
        )

        assert first_projection.duration == "6 Semester"
        assert programme.duration == "6 Semester"

        conflicting_evidence = make_duration_evidence(
            evidence_id="b" * 64,
            value="4 Semester",
            raw_text="Studiendauer: 4 Semester",
            source=source,
        )

        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            evidence_item=conflicting_evidence,
        )

        conflict_projection = refresh_programme_from_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
        )

        db.commit()
        db.expire_all()

        loaded_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id=user_id,
        )

        assert loaded_programme is not None

        assert conflict_projection.duration is None
        assert conflict_projection.conflicts == {
            "duration": [
                "6 Semester",
                "4 Semester",
            ]
        }

        assert loaded_programme.duration is None
        
    assert len(loaded_programme.missing_information) == 1

    missing_item = loaded_programme.missing_information[0]

    assert missing_item.field_name == "duration"
    assert missing_item.status == "conflicting"
    assert missing_item.importance == "medium"

    assert missing_item.searched_source_ids == [
        source.id
    ]

    assert missing_item.searched_source_urls == [
        source.url
    ]

    assert "6 Semester" in missing_item.reason
    assert "4 Semester" in missing_item.reason
    
    
    
def test_repeated_refresh_does_not_duplicate_conflict() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        user_id, source, programme = prepare_storage(db)

        first_evidence = make_duration_evidence(
            evidence_id="a" * 64,
            value="6 Semester",
            raw_text="Studiendauer: 6 Semester",
            source=source,
        )

        second_evidence = make_duration_evidence(
            evidence_id="b" * 64,
            value="4 Semester",
            raw_text="Studiendauer: 4 Semester",
            source=source,
        )

        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            evidence_item=first_evidence,
        )

        persist_evidence_item(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            evidence_item=second_evidence,
        )

        refresh_programme_from_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
        )

        refresh_programme_from_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
        )

        db.commit()
        db.expire_all()

        loaded_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id=user_id,
        )

        assert loaded_programme is not None
        assert len(loaded_programme.missing_information) == 1
        assert (
            loaded_programme.missing_information[0].status
            == "conflicting"
        )
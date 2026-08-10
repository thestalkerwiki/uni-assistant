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
from uni_assist.extraction.evidence_grounding import (
    EvidenceSourceContext,
)
from uni_assist.services.persisted_evidence_pipeline_service import (
    build_and_persist_evidence,
)
from uni_assist.storage.database import Base
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import (
    create_search_session,
    create_source_record,
    get_programme_by_id,
)


class FakeClassifier:
    """Resolve every test candidate as duration/workload."""

    def classify(self, request):
        return [
            EvidenceClassification(
                candidate_id=candidate.candidate_id,
                status=ClassificationStatus.RESOLVED,
                category=EvidenceCategory.DURATION_WORKLOAD,
            )
            for candidate in request.candidates
        ]


def test_builds_and_persists_evidence_end_to_end() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    Base.metadata.create_all(engine)

    structured_blocks = [
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

    with Session(engine) as db:
        user_id = "user-123"

        search_session = create_search_session(
            db=db,
            user_id=user_id,
            query="Psychology admission",
            urls=["https://example.edu/psychology"],
            output_language="en",
        )

        stored_source = create_source_record(
            db=db,
            session_id=search_session.id,
            user_id=user_id,
            url="https://example.edu/psychology",
            normalized_url="https://example.edu/psychology",
            source_type="webpage",
            source_language="de",
            clean_text="Studiendauer: 6 Semester",
            structured_blocks=structured_blocks,
        )

        programme = ProgrammeModel(
            user_id=user_id,
            title="Psychology",
        )

        db.add(programme)
        db.flush()

        source_context = EvidenceSourceContext(
            source_id=stored_source.id,
            source_url=stored_source.url,
            source_type=stored_source.source_type,
            source_language=stored_source.source_language,
        )

        result = build_and_persist_evidence(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            structured_blocks=structured_blocks,
            source=source_context,
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

        stored_evidence = loaded_programme.evidence_items[0]

        assert stored_evidence.category == "duration_workload"
        assert stored_evidence.label == "Studiendauer"
        assert stored_evidence.value == "6 Semester"
        assert stored_evidence.raw_text == (
            "Studiendauer: 6 Semester"
        )
        assert stored_evidence.source_id == stored_source.id

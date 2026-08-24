"""Manual live smoke test for the complete source-analysis flow."""

import argparse
from dataclasses import asdict

from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceConfidence
from uni_assist.integrations.openai_classifier import (
    OpenAIEvidenceClassifier,
)
from uni_assist.services.source_analysis_service import analyse_url
from uni_assist.storage.database import (
    SessionLocal,
    init_db,
)
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import (
    get_or_create_user,
    get_programme_by_id,
)


USER_ID = "live-smoke-user"


def create_smoke_programme(
    db: Session,
    title: str,
) -> ProgrammeModel:
    get_or_create_user(
        db=db,
        user_id=USER_ID,
    )

    programme = ProgrammeModel(
        user_id=USER_ID,
        title=title,
    )

    db.add(programme)
    db.commit()

    return programme


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "url",
        help="Official university page to analyse.",
    )
    parser.add_argument(
        "--programme",
        default="Live Smoke Programme",
    )
    parser.add_argument(
        "--query",
        default="What should an applicant know about this programme?",
    )

    args = parser.parse_args()

    init_db()

    with SessionLocal() as db:
        programme = create_smoke_programme(
            db=db,
            title=args.programme,
        )

        print()
        print("=== LIVE SOURCE ANALYSIS ===")
        print(f"URL: {args.url}")
        print(f"Programme ID: {programme.id}")
        print()

        result = analyse_url(
            db=db,
            user_id=USER_ID,
            programme_id=programme.id,
            url=args.url,
            query=args.query,
            output_language="en",
            classifier=OpenAIEvidenceClassifier(
                model_name="gpt-5.5",
            ),
            confidence=EvidenceConfidence.HIGH,
        )

        print("=== SOURCE ===")
        print(f"id: {result.ingestion.source.id}")
        print(f"url: {result.ingestion.source.url}")
        print(
            "language:",
            result.ingestion.source.source_language,
        )
        print(
            "type:",
            result.ingestion.source.source_type,
        )
        print(
            "structured blocks:",
            len(result.ingestion.source.structured_blocks),
        )
        print()

        print("=== RESOLVED EVIDENCE ===")

        if not result.evidence.evidence_items:
            print("(none)")

        for evidence_item in result.evidence.evidence_items:
            print()
            print(
                evidence_item.model_dump_json(
                    indent=2,
                )
            )

        print()
        print("=== UNRESOLVED ===")

        if not result.evidence.unresolved_candidates:
            print("(none)")

        for candidate in result.evidence.unresolved_candidates:
            print()
            print(
                candidate.model_dump_json(
                    indent=2,
                )
            )

        print()
        print("=== PROGRAMME PROJECTION ===")
        print(
            asdict(
                result.programme_projection
            )
        )

        persisted_programme = get_programme_by_id(
            db=db,
            programme_id=programme.id,
            user_id=USER_ID,
        )

        if persisted_programme is None:
            raise RuntimeError(
                "Programme disappeared after analysis."
            )

        print()
        print("=== PERSISTED PROGRAMME ===")
        print(f"title: {persisted_programme.title}")
        print(f"degree: {persisted_programme.degree}")
        print(f"duration: {persisted_programme.duration}")
        print(f"credits: {persisted_programme.credits}")
        print(f"language: {persisted_programme.language}")
        print(
            "evidence items:",
            len(persisted_programme.evidence_items),
        )
        print(
            "missing information:",
            len(persisted_programme.missing_information),
        )

        if not result.evidence.evidence_items:
            raise RuntimeError(
                "The live pipeline completed, but no resolved "
                "EvidenceItem was produced. Try a programme page "
                "containing explicit structured facts."
            )

        print()
        print("LIVE SOURCE ANALYSIS: PASS")


if __name__ == "__main__":
    main()
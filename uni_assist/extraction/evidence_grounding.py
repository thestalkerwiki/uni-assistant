"""Attach source provenance and identity to evidence candidates."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from uni_assist.domain.evidence import SourceLocator
from uni_assist.extraction.evidence_candidate import (
    EvidenceCandidate,
)


class EvidenceSourceContext(BaseModel):
    """
    Metadata of the source from which evidence was extracted.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_language: str = Field(min_length=1)


class GroundedEvidenceCandidate(EvidenceCandidate):
    """
    An evidence candidate permanently connected to its source.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    candidate_id: str = Field(
        min_length=64,
        max_length=64,
    )

    source_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_language: str = Field(min_length=1)

    source_locator: SourceLocator


def _build_candidate_id(
    candidate: EvidenceCandidate,
    source: EvidenceSourceContext,
) -> str:
    """
    Build a deterministic identity for one factual occurrence.

    The same candidate inside the same source record produces
    the same identifier.
    """

    identity_payload = {
        "source_id": source.source_id,
        "source_index": candidate.source_index,
        "raw_label": candidate.raw_label,
        "raw_value": candidate.raw_value,
        "raw_text": candidate.raw_text,
        "section_path": candidate.section_path,
    }

    serialized_payload = json.dumps(
        identity_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized_payload.encode("utf-8")
    ).hexdigest()


def ground_evidence_candidates(
    candidates: list[EvidenceCandidate],
    source: EvidenceSourceContext,
) -> list[GroundedEvidenceCandidate]:
    """
    Attach source metadata, identity and locator to candidates.

    No semantic category is inferred here.
    """

    grounded_candidates: list[
        GroundedEvidenceCandidate
    ] = []

    for candidate in candidates:
        heading = (
            candidate.section_path[-1]
            if candidate.section_path
            else None
        )

        locator = SourceLocator(
            heading=heading,
            block_index=candidate.source_index,
        )

        grounded_candidates.append(
            GroundedEvidenceCandidate(
                **candidate.model_dump(),
                candidate_id=_build_candidate_id(
                    candidate=candidate,
                    source=source,
                ),
                source_id=source.source_id,
                source_url=source.source_url,
                source_type=source.source_type,
                source_language=source.source_language,
                source_locator=locator,
            )
        )

    return grounded_candidates

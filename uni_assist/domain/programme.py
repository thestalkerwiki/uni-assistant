"""Domain models for structured programme profiles."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from uni_assist.domain.evidence import (
    EvidenceImportance,
    EvidenceItem,
)


class MissingInformationStatus(str, Enum):
    """Reason why an Admission Core field is unresolved."""

    NOT_FOUND = "not_found"
    CONFLICTING = "conflicting"
    AMBIGUOUS = "ambiguous"
    EXTERNAL_SOURCE_REQUIRED = "external_source_required"


class MissingInformationItem(BaseModel):
    """An unresolved or uncertain Admission Core field."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    field: str = Field(min_length=1, max_length=100)
    status: MissingInformationStatus
    reason: str = Field(min_length=1)

    importance: EvidenceImportance = EvidenceImportance.MEDIUM

    searched_source_ids: List[str] = Field(default_factory=list)
    searched_source_urls: List[str] = Field(default_factory=list)


class ProgrammeProfile(BaseModel):
    """A persistent programme-level view built from validated evidence."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: Optional[str] = None
    user_id: str = Field(min_length=1)

    title: str = Field(min_length=1, max_length=200)

    programme_name: Optional[str] = None
    institution_name: Optional[str] = None
    degree: Optional[str] = None
    duration: Optional[str] = None
    credits: Optional[str] = None
    language: Optional[str] = None
    country: Optional[str] = None

    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    missing_information: List[MissingInformationItem] = Field(
        default_factory=list
    )

    source_ids: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
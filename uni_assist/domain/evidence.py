"""Domain models for source-grounded admission evidence."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvidenceCategory(str, Enum):
    """Language-neutral categories used throughout Uni-Assist."""

    PROGRAMME_IDENTITY = "programme_identity"
    DEGREE_QUALIFICATION = "degree_qualification"
    DURATION_WORKLOAD = "duration_workload"
    CREDIT_REQUIREMENT = "credit_requirement"
    LANGUAGE_OF_INSTRUCTION = "language_of_instruction"
    ADMISSION_ELIGIBILITY = "admission_eligibility"
    DOCUMENT_REQUIREMENT = "document_requirement"
    LANGUAGE_REQUIREMENT = "language_requirement"
    DEADLINE = "deadline"
    SELECTION_PROCESS = "selection_process"
    TEST_OR_ASSESSMENT = "test_or_assessment"
    FINANCIAL_REQUIREMENT = "financial_requirement"
    SPECIAL_CONDITION = "special_condition"
    MISSING_INFORMATION = "missing_information"
    NEXT_ACTION = "next_action"


class EvidenceStage(str, Enum):
    """Application stage at which the evidence matters."""

    PROGRAMME_INFORMATION = "programme_information"
    PRE_APPLICATION = "pre_application"
    APPLICATION = "application"
    SELECTION = "selection"
    ADMISSION = "admission"
    ENROLLMENT = "enrollment"
    POST_ENROLLMENT = "post_enrollment"
    NOT_SPECIFIED = "not_specified"


class EvidenceImportance(str, Enum):
    """Practical importance of the evidence for the applicant."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceConfidence(str, Enum):
    """Confidence that the evidence is supported by the source."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceLocator(BaseModel):
    """Location of the supporting fragment inside a source."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    heading: Optional[str] = None
    block_index: Optional[int] = Field(default=None, ge=0)
    page_number: Optional[int] = Field(default=None, ge=1)
    css_selector: Optional[str] = None
    character_start: Optional[int] = Field(default=None, ge=0)
    character_end: Optional[int] = Field(default=None, ge=0)


class EvidenceItem(BaseModel):
    """A normalized admission fact supported by one source fragment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: Optional[str] = None

    category: EvidenceCategory
    label: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1)
    applicant_action: Optional[str] = None

    stage: EvidenceStage = EvidenceStage.NOT_SPECIFIED
    importance: EvidenceImportance = EvidenceImportance.MEDIUM
    confidence: EvidenceConfidence

    source_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    source_locator: Optional[SourceLocator] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
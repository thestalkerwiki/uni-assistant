"""Domain models for user query intent."""

from enum import Enum

from pydantic import BaseModel, ConfigDict


class QueryIntent(str, Enum):
    OVERVIEW = "overview"
    ADMISSION_REQUIREMENTS = "admission_requirements"
    DEADLINES = "deadlines"
    DOCUMENTS = "documents"
    PREPARATION = "preparation"
    DIRECT_QUESTION = "direct_question"


class QueryInterpretation(BaseModel):
    """Language-neutral interpretation of one user question."""

    model_config = ConfigDict(
        extra="forbid",
    )

    intent: QueryIntent
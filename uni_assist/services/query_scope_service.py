"""Deterministic evidence scope for interpreted user queries."""

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.domain.query import QueryIntent


QUERY_INTENT_CATEGORIES = {
    QueryIntent.ADMISSION_REQUIREMENTS: (
        EvidenceCategory.ADMISSION_ELIGIBILITY,
        EvidenceCategory.LANGUAGE_REQUIREMENT,
        EvidenceCategory.SELECTION_PROCESS,
        EvidenceCategory.TEST_OR_ASSESSMENT,
        EvidenceCategory.SPECIAL_CONDITION,
    ),
    QueryIntent.DEADLINES: (
        EvidenceCategory.DEADLINE,
    ),
    QueryIntent.DOCUMENTS: (
        EvidenceCategory.DOCUMENT_REQUIREMENT,
    ),
    QueryIntent.PREPARATION: (
        EvidenceCategory.ADMISSION_ELIGIBILITY,
        EvidenceCategory.DOCUMENT_REQUIREMENT,
        EvidenceCategory.LANGUAGE_REQUIREMENT,
        EvidenceCategory.DEADLINE,
        EvidenceCategory.SELECTION_PROCESS,
        EvidenceCategory.TEST_OR_ASSESSMENT,
        EvidenceCategory.FINANCIAL_REQUIREMENT,
        EvidenceCategory.SPECIAL_CONDITION,
        EvidenceCategory.NEXT_ACTION,
    ),
}


def get_evidence_categories_for_intent(
    intent: QueryIntent,
) -> tuple[EvidenceCategory, ...]:
    """
    Return evidence categories relevant to one query intent.

    OVERVIEW and DIRECT_QUESTION remain unrestricted because
    their relevance cannot be safely narrowed from intent alone.
    """

    if intent in {
        QueryIntent.OVERVIEW,
        QueryIntent.DIRECT_QUESTION,
    }:
        return tuple(EvidenceCategory)

    return QUERY_INTENT_CATEGORIES[intent]
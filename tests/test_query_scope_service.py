from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.domain.query import QueryIntent
from uni_assist.services.query_scope_service import (
    get_evidence_categories_for_intent,
)


def test_deadline_intent_only_selects_deadline() -> None:
    categories = get_evidence_categories_for_intent(
        QueryIntent.DEADLINES
    )

    assert categories == (
        EvidenceCategory.DEADLINE,
    )


def test_documents_intent_only_selects_document_requirement() -> None:
    categories = get_evidence_categories_for_intent(
        QueryIntent.DOCUMENTS
    )

    assert categories == (
        EvidenceCategory.DOCUMENT_REQUIREMENT,
    )


def test_admission_requirements_do_not_include_deadlines() -> None:
    categories = get_evidence_categories_for_intent(
        QueryIntent.ADMISSION_REQUIREMENTS
    )

    assert EvidenceCategory.ADMISSION_ELIGIBILITY in categories
    assert EvidenceCategory.LANGUAGE_REQUIREMENT in categories

    assert EvidenceCategory.DEADLINE not in categories
    assert EvidenceCategory.DOCUMENT_REQUIREMENT not in categories


def test_preparation_includes_actionable_categories() -> None:
    categories = get_evidence_categories_for_intent(
        QueryIntent.PREPARATION
    )

    assert EvidenceCategory.DOCUMENT_REQUIREMENT in categories
    assert EvidenceCategory.DEADLINE in categories
    assert EvidenceCategory.LANGUAGE_REQUIREMENT in categories
    assert EvidenceCategory.TEST_OR_ASSESSMENT in categories


def test_overview_remains_unrestricted() -> None:
    categories = get_evidence_categories_for_intent(
        QueryIntent.OVERVIEW
    )

    assert set(categories) == set(EvidenceCategory)


def test_direct_question_remains_unrestricted() -> None:
    categories = get_evidence_categories_for_intent(
        QueryIntent.DIRECT_QUESTION
    )

    assert set(categories) == set(EvidenceCategory)
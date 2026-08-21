from types import SimpleNamespace

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.services.programme_projection_service import (
    build_programme_projection,
)


def make_evidence(
    category: EvidenceCategory,
    value: str,
):
    return SimpleNamespace(
        category=category.value,
        value=value,
    )


def test_projects_single_evidence_value() -> None:
    evidence_items = [
        make_evidence(
            EvidenceCategory.DURATION_WORKLOAD,
            "6 Semester",
        )
    ]

    projection = build_programme_projection(evidence_items)

    assert projection.duration == "6 Semester"
    assert projection.conflicts == {}


def test_same_value_from_multiple_evidence_is_not_conflict() -> None:
    evidence_items = [
        make_evidence(
            EvidenceCategory.LANGUAGE_OF_INSTRUCTION,
            "English",
        ),
        make_evidence(
            EvidenceCategory.LANGUAGE_OF_INSTRUCTION,
            "English",
        ),
    ]

    projection = build_programme_projection(evidence_items)

    assert projection.language == "English"
    assert projection.conflicts == {}


def test_conflicting_values_are_not_projected() -> None:
    evidence_items = [
        make_evidence(
            EvidenceCategory.DURATION_WORKLOAD,
            "4 Semester",
        ),
        make_evidence(
            EvidenceCategory.DURATION_WORKLOAD,
            "6 Semester",
        ),
    ]

    projection = build_programme_projection(evidence_items)

    assert projection.duration is None
    assert projection.conflicts == {
        "duration": [
            "4 Semester",
            "6 Semester",
        ]
    }


def test_unrelated_evidence_is_ignored() -> None:
    evidence_items = [
        make_evidence(
            EvidenceCategory.DEADLINE,
            "15 July",
        )
    ]

    projection = build_programme_projection(evidence_items)

    assert projection.degree is None
    assert projection.duration is None
    assert projection.credits is None
    assert projection.language is None
    assert projection.conflicts == {}
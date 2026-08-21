from dataclasses import dataclass, field
from typing import Optional, Sequence

from uni_assist.domain.evidence import EvidenceCategory
from uni_assist.storage.models import EvidenceItemModel


CATEGORY_TO_PROGRAMME_FIELD = {
    EvidenceCategory.DEGREE_QUALIFICATION.value: "degree",
    EvidenceCategory.DURATION_WORKLOAD.value: "duration",
    EvidenceCategory.CREDIT_REQUIREMENT.value: "credits",
    EvidenceCategory.LANGUAGE_OF_INSTRUCTION.value: "language",
}


@dataclass(frozen=True)
class ProgrammeProjection:
    degree: Optional[str] = None
    duration: Optional[str] = None
    credits: Optional[str] = None
    language: Optional[str] = None

    conflicts: dict[str, list[str]] = field(default_factory=dict)


def build_programme_projection(
    evidence_items: Sequence[EvidenceItemModel],
) -> ProgrammeProjection:
    values_by_field: dict[str, list[str]] = {}

    for evidence_item in evidence_items:
        target_field = CATEGORY_TO_PROGRAMME_FIELD.get(
            evidence_item.category
        )

        if target_field is None:
            continue

        values_by_field.setdefault(target_field, []).append(
            evidence_item.value
        )

    resolved: dict[str, Optional[str]] = {
        "degree": None,
        "duration": None,
        "credits": None,
        "language": None,
    }
    conflicts: dict[str, list[str]] = {}

    for field_name, values in values_by_field.items():
        unique_values = list(dict.fromkeys(values))

        if len(unique_values) == 1:
            resolved[field_name] = unique_values[0]
        else:
            conflicts[field_name] = unique_values

    return ProgrammeProjection(
        degree=resolved["degree"],
        duration=resolved["duration"],
        credits=resolved["credits"],
        language=resolved["language"],
        conflicts=conflicts,
    )
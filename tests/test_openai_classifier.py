from uni_assist.domain.evidence import (
    EvidenceCategory,
)
from uni_assist.extraction.classification_contract import (
    ClassificationStatus,
    EvidenceClassification,
)
from uni_assist.extraction.classification_input import (
    EvidenceClassificationInput,
)
from uni_assist.extraction.classification_request import (
    build_classification_request,
)
from uni_assist.integrations.openai_classifier import (
    EvidenceClassificationBatch,
    OpenAIEvidenceClassifier,
)


class FakeStructuredModel:
    def __init__(self) -> None:
        self.received_input = None

    def invoke(
        self,
        input: object,
    ) -> EvidenceClassificationBatch:
        self.received_input = input

        return EvidenceClassificationBatch(
            classifications=[
                EvidenceClassification(
                    candidate_id="a" * 64,
                    status=(
                        ClassificationStatus.RESOLVED
                    ),
                    category=(
                        EvidenceCategory.DURATION_WORKLOAD
                    ),
                )
            ]
        )


def build_request():
    classification_input = (
        EvidenceClassificationInput(
            candidate_id="a" * 64,
            raw_label="Studiendauer",
            raw_value="6 Semester",
            raw_text=(
                "Studiendauer: 6 Semester"
            ),
            section_path=[
                "Studienübersicht",
            ],
            source_language="de",
        )
    )

    return build_classification_request(
        inputs=[classification_input],
    )


def test_openai_adapter_returns_structured_results() -> None:
    fake_model = FakeStructuredModel()

    classifier = OpenAIEvidenceClassifier(
        structured_model=fake_model,
    )

    results = classifier.classify(
        request=build_request(),
    )

    assert len(results) == 1
    assert results[0].candidate_id == "a" * 64
    assert results[0].status == (
        ClassificationStatus.RESOLVED
    )
    assert results[0].category == (
        EvidenceCategory.DURATION_WORKLOAD
    )

    assert fake_model.received_input is not None


def test_openai_adapter_marks_source_data_as_untrusted() -> None:
    fake_model = FakeStructuredModel()

    classifier = OpenAIEvidenceClassifier(
        structured_model=fake_model,
    )

    classifier.classify(
        request=build_request(),
    )

    messages = fake_model.received_input
    system_message = messages[0].content
    human_message = messages[1].content

    assert "untrusted source data" in (
        system_message.casefold()
    )

    assert "Studiendauer" in human_message
    assert "6 Semester" in human_message
    assert "a" * 64 in human_message

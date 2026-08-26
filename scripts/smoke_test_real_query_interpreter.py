"""Manual smoke test for multilingual query interpretation."""

from uni_assist.domain.query import QueryIntent
from uni_assist.integrations.openai_query_interpreter import (
    OpenAIQueryInterpreter,
)
from uni_assist.services.query_interpretation_service import (
    interpret_user_query,
)


TEST_CASES = [
    (
        "What are the application deadlines?",
        QueryIntent.DEADLINES,
    ),
    (
        "Welche Fristen gelten für die Bewerbung?",
        QueryIntent.DEADLINES,
    ),
    (
        "Какие сроки подачи заявления?",
        QueryIntent.DEADLINES,
    ),
]


def main() -> None:
    interpreter = OpenAIQueryInterpreter(
        model_name="gpt-5.5",
    )

    for question, expected_intent in TEST_CASES:
        result = interpret_user_query(
            question=question,
            interpreter=interpreter,
        )

        print()
        print("QUESTION:", question)
        print("INTENT:", result.intent.value)

        if result.intent != expected_intent:
            raise RuntimeError(
                "Unexpected query intent. "
                f"Expected {expected_intent.value!r}, "
                f"got {result.intent.value!r}."
            )

    print()
    print("REAL QUERY INTERPRETER: PASS")


if __name__ == "__main__":
    main()
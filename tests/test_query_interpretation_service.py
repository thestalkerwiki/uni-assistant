import pytest

from uni_assist.domain.query import (
    QueryIntent,
    QueryInterpretation,
)
from uni_assist.services.query_interpretation_service import (
    interpret_user_query,
)


class FakeQueryInterpreter:
    def __init__(
        self,
        intent: QueryIntent,
    ) -> None:
        self.intent = intent
        self.received_questions = []

    def interpret(
        self,
        question: str,
    ) -> QueryInterpretation:
        self.received_questions.append(
            question
        )

        return QueryInterpretation(
            intent=self.intent
        )


def test_interprets_user_question() -> None:
    interpreter = FakeQueryInterpreter(
        QueryIntent.DEADLINES
    )

    result = interpret_user_query(
        question="Какие сроки подачи?",
        interpreter=interpreter,
    )

    assert result.intent == QueryIntent.DEADLINES
    assert interpreter.received_questions == [
        "Какие сроки подачи?"
    ]


def test_question_whitespace_is_removed() -> None:
    interpreter = FakeQueryInterpreter(
        QueryIntent.DOCUMENTS
    )

    interpret_user_query(
        question="  Welche Dokumente brauche ich?  ",
        interpreter=interpreter,
    )

    assert interpreter.received_questions == [
        "Welche Dokumente brauche ich?"
    ]


def test_empty_question_is_rejected() -> None:
    interpreter = FakeQueryInterpreter(
        QueryIntent.OVERVIEW
    )

    with pytest.raises(
        ValueError,
        match="Question must not be empty",
    ):
        interpret_user_query(
            question="   ",
            interpreter=interpreter,
        )
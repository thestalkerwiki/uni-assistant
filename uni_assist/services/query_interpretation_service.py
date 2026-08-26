"""Orchestrate language-neutral interpretation of user queries."""

from typing import Protocol

from uni_assist.domain.query import QueryInterpretation


class QueryInterpreter(Protocol):
    """Interface implemented by semantic query interpreters."""

    def interpret(
        self,
        question: str,
    ) -> QueryInterpretation:
        ...


def interpret_user_query(
    question: str,
    interpreter: QueryInterpreter,
) -> QueryInterpretation:
    """Interpret one non-empty user question."""

    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError(
            "Question must not be empty."
        )

    return interpreter.interpret(
        normalized_question
    )
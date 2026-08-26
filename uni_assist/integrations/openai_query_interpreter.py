"""OpenAI adapter for constrained user-query interpretation."""

import os
from typing import Optional, Protocol

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from uni_assist.domain.query import QueryInterpretation


SYSTEM_PROMPT = """
You are a constrained query-intent classifier inside Uni-Assist.

Your only task is to identify what the user is asking for.

The user may write in any language.

Intent meanings:

- overview:
  A broad programme overview or general summary.

- admission_requirements:
  Admission eligibility, entry requirements, language requirements,
  tests, selection conditions, or other admission conditions.

- deadlines:
  Application deadlines, submission periods, dates, or time limits.

- documents:
  Required documents, certificates, forms, proofs, or attachments.

- preparation:
  What the applicant should prepare, do, plan, or focus on before
  or during the application process.

- direct_question:
  A specific question that cannot be safely reduced to one of the
  intents above.

Important rules:
- Do not answer the user's question.
- Do not generate or infer university facts.
- Do not rewrite the user's question.
- Classify only the user's intent.
- Return exactly one valid intent.
""".strip()


class StructuredQueryModel(Protocol):
    """Minimal interface required from a structured LLM."""

    def invoke(
        self,
        input: object,
    ) -> QueryInterpretation:
        ...


class OpenAIQueryInterpreter:
    """Interpret user questions using OpenAI structured output."""

    def __init__(
        self,
        model_name: str = "gpt-5.5",
        structured_model: Optional[
            StructuredQueryModel
        ] = None,
    ) -> None:
        if structured_model is not None:
            self._structured_model = structured_model
            return

        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY was not found in the environment."
            )

        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=60,
            max_retries=2,
        )

        self._structured_model = (
            llm.with_structured_output(
                QueryInterpretation,
                method="json_schema",
            )
        )

    def interpret(
        self,
        question: str,
    ) -> QueryInterpretation:
        messages = [
            SystemMessage(
                content=SYSTEM_PROMPT
            ),
            HumanMessage(
                content=question
            ),
        ]

        response = self._structured_model.invoke(
            messages
        )

        if not isinstance(
            response,
            QueryInterpretation,
        ):
            raise TypeError(
                "Structured query interpreter returned "
                "an unexpected response type."
            )

        return response
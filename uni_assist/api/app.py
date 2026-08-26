"""FastAPI entry point for Uni-Assist v2."""

from contextlib import asynccontextmanager
from typing import Generator, Optional

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from uni_assist.domain.evidence import EvidenceConfidence
from uni_assist.integrations.openai_classifier import (
    OpenAIEvidenceClassifier,
)
from uni_assist.integrations.openai_query_interpreter import (
    OpenAIQueryInterpreter,
)
from uni_assist.services.classification_service import (
    EvidenceClassifier,
)
from uni_assist.services.query_interpretation_service import (
    QueryInterpreter,
    interpret_user_query,
)
from uni_assist.services.query_scope_service import (
    get_evidence_categories_for_intent,
)
from uni_assist.services.source_analysis_service import analyse_url
from uni_assist.storage.database import (
    SessionLocal,
    init_db,
)
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import get_or_create_user


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Uni-Assist API",
    version="2.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


class AnalyseSourceRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    programme_title: str = Field(
        default="Untitled Programme",
        min_length=1,
        max_length=200,
    )


class EvidenceResponse(BaseModel):
    id: str
    category: str
    label: str
    value: str
    raw_text: str
    source_url: str


class ProgrammeResponse(BaseModel):
    id: str
    title: str
    degree: Optional[str] = None
    duration: Optional[str] = None
    credits: Optional[str] = None
    language: Optional[str] = None


class AnalyseSourceResponse(BaseModel):
    source_id: str
    source_url: str
    source_language: str
    intent: str

    evidence: list[EvidenceResponse]
    unresolved_count: int

    programme: ProgrammeResponse
    conflicts: dict[str, list[str]]


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def get_query_interpreter() -> QueryInterpreter:
    return OpenAIQueryInterpreter(
        model_name="gpt-5.5",
    )


def get_classifier() -> EvidenceClassifier:
    return OpenAIEvidenceClassifier(
        model_name="gpt-5.5",
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Uni-Assist v2 API is running",
    }


@app.get(
    "/demo",
    include_in_schema=False,
)
def demo_page():
    return FileResponse("static/demo.html")


@app.post(
    "/api/v2/analyse",
    response_model=AnalyseSourceResponse,
)
def analyse_source(
    request: AnalyseSourceRequest,
    db: Session = Depends(get_db),
    classifier: EvidenceClassifier = Depends(get_classifier),
    query_interpreter: QueryInterpreter = Depends(
        get_query_interpreter
    ),
) -> AnalyseSourceResponse:
    user_id = "demo-user"

    query_interpretation = interpret_user_query(
        question=request.question,
        interpreter=query_interpreter,
    )

    allowed_categories = set(
        get_evidence_categories_for_intent(
            query_interpretation.intent
        )
    )

    get_or_create_user(
        db=db,
        user_id=user_id,
    )

    programme = ProgrammeModel(
        user_id=user_id,
        title=request.programme_title,
    )

    db.add(programme)
    db.commit()

    evidence_items = []
    unresolved_count = 0
    analysed_results = []

    for url in request.urls:
        result = analyse_url(
            db=db,
            user_id=user_id,
            programme_id=programme.id,
            url=url,
            query=request.question,
            output_language="en",
            classifier=classifier,
            confidence=EvidenceConfidence.HIGH,
        )

        analysed_results.append(result)

        evidence_items.extend(
            result.evidence.evidence_items
        )

        unresolved_count += len(
            result.evidence.unresolved_candidates
        )

    response_evidence_items = [
        item
        for item in evidence_items
        if item.category in allowed_categories
    ]

    final_result = analysed_results[-1]

    return AnalyseSourceResponse(
        source_id=final_result.ingestion.source.id,
        source_url=final_result.ingestion.source.url,
        source_language=(
            final_result.ingestion.source.source_language
        ),
        intent=query_interpretation.intent.value,
        evidence=[
            EvidenceResponse(
                id=item.id,
                category=item.category.value,
                label=item.label,
                value=item.value,
                raw_text=item.raw_text,
                source_url=item.source_url,
            )
            for item in response_evidence_items
        ],
        unresolved_count=unresolved_count,
        programme=ProgrammeResponse(
            id=programme.id,
            title=programme.title,
            degree=(
                final_result.programme_projection.degree
            ),
            duration=(
                final_result.programme_projection.duration
            ),
            credits=(
                final_result.programme_projection.credits
            ),
            language=(
                final_result.programme_projection.language
            ),
        ),
        conflicts=(
            final_result.programme_projection.conflicts
        ),
    )
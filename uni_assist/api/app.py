"""FastAPI entry point for Uni-Assist v2."""

from contextlib import asynccontextmanager
from typing import Generator, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from uni_assist.domain.evidence import (
    EvidenceCategory,
    EvidenceConfidence,
)
from uni_assist.integrations.openai_classifier import (
    OpenAIEvidenceClassifier,
)
from uni_assist.integrations.openai_query_interpreter import (
    OpenAIQueryInterpreter,
)
from uni_assist.services.classification_service import (
    EvidenceClassifier,
)
from uni_assist.services.programme_workspace_service import (
    add_sources_to_programme_workspace,
    create_programme_workspace,
    get_programme_workspace,
    list_programme_workspaces,
)
from uni_assist.services.query_interpretation_service import (
    QueryInterpreter,
    interpret_user_query,
)
from uni_assist.services.query_scope_service import (
    get_evidence_categories_for_intent,
)
from uni_assist.services.source_analysis_service import (
    analyse_url,
)
from uni_assist.storage.database import (
    SessionLocal,
    init_db,
)
from uni_assist.storage.models import ProgrammeModel
from uni_assist.storage.repositories import (
    get_or_create_user,
)


DEMO_USER_ID = "demo-user"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize persistent storage when the API starts."""

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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateProgrammeRequest(BaseModel):
    """User input for creating one persistent programme workspace."""

    title: str = Field(
        min_length=1,
        max_length=200,
    )

    programme_name: Optional[str] = Field(
        default=None,
        max_length=200,
    )

    institution_name: Optional[str] = Field(
        default=None,
        max_length=200,
    )


class AddProgrammeSourcesRequest(BaseModel):
    """New official sources to add to an existing programme."""

    urls: list[str] = Field(
        min_length=1,
    )

    query: str = Field(
        default="Programme overview",
        min_length=1,
    )

    output_language: str = Field(
        default="en",
        min_length=1,
        max_length=50,
    )


class AskProgrammeQuestionRequest(BaseModel):
    """A question asked against already accumulated programme state."""

    question: str = Field(
        min_length=1,
    )


class AnalyseSourceRequest(BaseModel):
    """
    Compatibility model for the original one-shot demo flow.

    The persistent programme endpoints are now the preferred path.
    """

    urls: list[str] = Field(
        min_length=1,
    )

    question: str = Field(
        min_length=1,
    )

    programme_title: str = Field(
        default="Untitled Programme",
        min_length=1,
        max_length=200,
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class SourceResponse(BaseModel):
    """Public API representation of one persisted source."""

    id: str
    url: str
    normalized_url: str
    source_language: str


class EvidenceResponse(BaseModel):
    """Public API representation of one source-grounded fact."""

    id: str
    category: str
    label: str
    value: str
    raw_text: str
    source_url: str


class MissingInformationResponse(BaseModel):
    """Public representation of one unresolved programme field."""

    field: str
    status: str
    reason: str
    importance: str


class ProgrammeResponse(BaseModel):
    """Current projected state of one programme."""

    id: str
    title: str

    programme_name: Optional[str] = None
    institution_name: Optional[str] = None

    degree: Optional[str] = None
    duration: Optional[str] = None
    credits: Optional[str] = None
    language: Optional[str] = None


class ProgrammeWorkspaceResponse(BaseModel):
    """
    Complete accumulated state of one persistent programme workspace.
    """

    programme: ProgrammeResponse

    sources: list[SourceResponse]
    evidence: list[EvidenceResponse]
    missing_information: list[MissingInformationResponse]


class ProgrammeListItemResponse(BaseModel):
    """Compact representation used when listing saved programmes."""

    id: str
    title: str

    programme_name: Optional[str] = None
    institution_name: Optional[str] = None

    source_count: int
    evidence_count: int


class AddProgrammeSourcesResponse(BaseModel):
    """Updated workspace after adding new sources."""

    workspace: ProgrammeWorkspaceResponse

    added_sources: list[SourceResponse]
    skipped_urls: list[str]


class ProgrammeQuestionResponse(BaseModel):
    """
    Relevant existing evidence selected for one user question.
    """

    programme: ProgrammeResponse

    question: str
    intent: str

    evidence: list[EvidenceResponse]
    missing_information: list[MissingInformationResponse]


class AnalyseSourceResponse(BaseModel):
    """Response model for the original one-shot demo endpoint."""

    source_id: str
    source_url: str
    source_language: str
    intent: str

    evidence: list[EvidenceResponse]
    unresolved_count: int

    programme: ProgrammeResponse
    conflicts: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """Provide one database session for one API request."""

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def get_query_interpreter() -> QueryInterpreter:
    """Provide the real query-intent adapter."""

    return OpenAIQueryInterpreter(
        model_name="gpt-5.5",
    )


def get_classifier() -> EvidenceClassifier:
    """Provide the real evidence-classification adapter."""

    return OpenAIEvidenceClassifier(
        model_name="gpt-5.5",
    )


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def build_programme_response(
    programme: ProgrammeModel,
) -> ProgrammeResponse:
    """Convert stored programme state into the public API shape."""

    return ProgrammeResponse(
        id=programme.id,
        title=programme.title,
        programme_name=programme.programme_name,
        institution_name=programme.institution_name,
        degree=programme.degree,
        duration=programme.duration,
        credits=programme.credits,
        language=programme.language,
    )


def build_source_response(
    source,
) -> SourceResponse:
    """Convert one stored source into the public API shape."""

    return SourceResponse(
        id=source.id,
        url=source.url,
        normalized_url=source.normalized_url,
        source_language=source.source_language,
    )


def build_evidence_response(
    evidence,
) -> EvidenceResponse:
    """Convert stored or domain evidence into the public API shape."""

    category = evidence.category

    if isinstance(
        category,
        EvidenceCategory,
    ):
        category_value = category.value

    else:
        category_value = str(category)

    return EvidenceResponse(
        id=evidence.id,
        category=category_value,
        label=evidence.label,
        value=evidence.value,
        raw_text=evidence.raw_text,
        source_url=evidence.source_url,
    )


def build_missing_information_response(
    item,
) -> MissingInformationResponse:
    """Convert stored uncertainty into the public API shape."""

    return MissingInformationResponse(
        field=item.field_name,
        status=item.status,
        reason=item.reason,
        importance=item.importance,
    )


def build_workspace_response(
    programme: ProgrammeModel,
) -> ProgrammeWorkspaceResponse:
    """
    Build one complete snapshot of the programme's accumulated state.
    """

    return ProgrammeWorkspaceResponse(
        programme=build_programme_response(
            programme
        ),
        sources=[
            build_source_response(source)
            for source in programme.sources
        ],
        evidence=[
            build_evidence_response(evidence)
            for evidence in programme.evidence_items
        ],
        missing_information=[
            build_missing_information_response(item)
            for item in programme.missing_information
        ],
    )


def get_workspace_or_404(
    *,
    db: Session,
    programme_id: str,
) -> ProgrammeModel:
    """Load a demo-user workspace or return an HTTP 404."""

    try:
        return get_programme_workspace(
            db=db,
            user_id=DEMO_USER_ID,
            programme_id=programme_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


# ---------------------------------------------------------------------------
# Basic pages
# ---------------------------------------------------------------------------


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
    return FileResponse(
        "static/demo.html"
    )


# ---------------------------------------------------------------------------
# Persistent Programme Workspace API
# ---------------------------------------------------------------------------


@app.post(
    "/api/v2/programmes",
    response_model=ProgrammeWorkspaceResponse,
)
def create_programme(
    request: CreateProgrammeRequest,
    db: Session = Depends(get_db),
) -> ProgrammeWorkspaceResponse:
    """Create one persistent programme workspace."""

    programme = create_programme_workspace(
        db=db,
        user_id=DEMO_USER_ID,
        title=request.title,
        programme_name=request.programme_name,
        institution_name=request.institution_name,
    )

    programme = get_workspace_or_404(
        db=db,
        programme_id=programme.id,
    )

    return build_workspace_response(
        programme
    )


@app.get(
    "/api/v2/programmes",
    response_model=list[ProgrammeListItemResponse],
)
def list_programmes(
    db: Session = Depends(get_db),
) -> list[ProgrammeListItemResponse]:
    """List all saved programme workspaces."""

    programmes = list_programme_workspaces(
        db=db,
        user_id=DEMO_USER_ID,
    )

    return [
        ProgrammeListItemResponse(
            id=programme.id,
            title=programme.title,
            programme_name=programme.programme_name,
            institution_name=programme.institution_name,
            source_count=len(
                programme.sources
            ),
            evidence_count=len(
                programme.evidence_items
            ),
        )
        for programme in programmes
    ]


@app.get(
    "/api/v2/programmes/{programme_id}",
    response_model=ProgrammeWorkspaceResponse,
)
def get_programme(
    programme_id: str,
    db: Session = Depends(get_db),
) -> ProgrammeWorkspaceResponse:
    """Open the complete accumulated state of one programme."""

    programme = get_workspace_or_404(
        db=db,
        programme_id=programme_id,
    )

    return build_workspace_response(
        programme
    )


@app.post(
    "/api/v2/programmes/{programme_id}/sources",
    response_model=AddProgrammeSourcesResponse,
)
def add_programme_sources(
    programme_id: str,
    request: AddProgrammeSourcesRequest,
    db: Session = Depends(get_db),
    classifier: EvidenceClassifier = Depends(
        get_classifier
    ),
) -> AddProgrammeSourcesResponse:
    """
    Add previously unknown sources to an existing workspace.

    Known normalized URLs are skipped.
    """

    get_workspace_or_404(
        db=db,
        programme_id=programme_id,
    )

    try:
        result = add_sources_to_programme_workspace(
            db=db,
            user_id=DEMO_USER_ID,
            programme_id=programme_id,
            urls=request.urls,
            query=request.query,
            output_language=request.output_language,
            classifier=classifier,
            confidence=EvidenceConfidence.HIGH,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    added_sources = [
        analysis.ingestion.source
        for analysis in result.analyses
    ]

    return AddProgrammeSourcesResponse(
        workspace=build_workspace_response(
            result.programme
        ),
        added_sources=[
            build_source_response(source)
            for source in added_sources
        ],
        skipped_urls=result.skipped_urls,
    )


@app.post(
    "/api/v2/programmes/{programme_id}/questions",
    response_model=ProgrammeQuestionResponse,
)
def ask_programme_question(
    programme_id: str,
    request: AskProgrammeQuestionRequest,
    db: Session = Depends(get_db),
    query_interpreter: QueryInterpreter = Depends(
        get_query_interpreter
    ),
) -> ProgrammeQuestionResponse:
    """
    Select relevant facts from existing accumulated programme state.

    No source is downloaded again and no new evidence is created.
    """

    programme = get_workspace_or_404(
        db=db,
        programme_id=programme_id,
    )

    interpretation = interpret_user_query(
        question=request.question,
        interpreter=query_interpreter,
    )

    allowed_categories = set(
        get_evidence_categories_for_intent(
            interpretation.intent
        )
    )

    relevant_evidence = [
        evidence
        for evidence in programme.evidence_items
        if EvidenceCategory(
            evidence.category
        )
        in allowed_categories
    ]

    return ProgrammeQuestionResponse(
        programme=build_programme_response(
            programme
        ),
        question=request.question,
        intent=interpretation.intent.value,
        evidence=[
            build_evidence_response(evidence)
            for evidence in relevant_evidence
        ],
        missing_information=[
            build_missing_information_response(item)
            for item in programme.missing_information
        ],
    )


# ---------------------------------------------------------------------------
# Original one-shot demo endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/api/v2/analyse",
    response_model=AnalyseSourceResponse,
)
def analyse_source(
    request: AnalyseSourceRequest,
    db: Session = Depends(get_db),
    classifier: EvidenceClassifier = Depends(
        get_classifier
    ),
    query_interpreter: QueryInterpreter = Depends(
        get_query_interpreter
    ),
) -> AnalyseSourceResponse:
    """
    Original one-shot analysis flow.

    It stays temporarily available while the browser UI is migrated
    to persistent programme workspaces.
    """

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
        user_id=DEMO_USER_ID,
    )

    programme = ProgrammeModel(
        user_id=DEMO_USER_ID,
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
            user_id=DEMO_USER_ID,
            programme_id=programme.id,
            url=url,
            query=request.question,
            output_language="en",
            classifier=classifier,
            confidence=EvidenceConfidence.HIGH,
        )

        analysed_results.append(
            result
        )

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
            build_evidence_response(item)
            for item in response_evidence_items
        ],
        unresolved_count=unresolved_count,
        programme=ProgrammeResponse(
            id=programme.id,
            title=programme.title,
            programme_name=programme.programme_name,
            institution_name=programme.institution_name,
            degree=final_result.programme_projection.degree,
            duration=final_result.programme_projection.duration,
            credits=final_result.programme_projection.credits,
            language=final_result.programme_projection.language,
        ),
        conflicts=final_result.programme_projection.conflicts,
    )
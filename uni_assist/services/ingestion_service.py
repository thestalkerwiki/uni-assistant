"""Application service for loading and persisting web sources."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Optional

from sqlalchemy.orm import Session

from uni_assist.ingestion.html_cleaner import clean_extracted_text
from uni_assist.ingestion.structure_extractor import (
    extract_structured_blocks,
)
from uni_assist.ingestion.web_loader import (
    ScrapedWebPage,
    scrape_web_page,
)
from uni_assist.storage.models import (
    SearchSessionModel,
    SourceModel,
)
from uni_assist.storage.repositories import (
    create_search_session,
    create_source_record,
)


UNKNOWN_SOURCE_TYPE = "webpage"
UNKNOWN_SOURCE_LANGUAGE = "unknown"


@dataclass(frozen=True)
class IngestionResult:
    """Persistent result of processing one web source."""

    search_session: SearchSessionModel
    source: SourceModel
    loaded_page: ScrapedWebPage


def build_content_hash(text: str) -> str:
    """Build a stable SHA-256 fingerprint for cleaned source text."""

    return sha256(text.encode("utf-8")).hexdigest()

def get_declared_source_language(soup) -> str:
    """Return the language explicitly declared by the HTML document."""

    html_tag = soup.find("html")

    if html_tag is None:
        return UNKNOWN_SOURCE_LANGUAGE

    raw_language = html_tag.get("lang")

    if not raw_language:
        return UNKNOWN_SOURCE_LANGUAGE

    language = str(raw_language).strip()

    return language or UNKNOWN_SOURCE_LANGUAGE


def ingest_url(
    db: Session,
    *,
    user_id: str,
    url: str,
    query: str,
    output_language: str,
    detected_intent: Optional[str] = None,
) -> IngestionResult:
    """
    Load, clean and persist one web source.

    Transaction ownership belongs to this service:
    all records are committed together or rolled back together.
    """

    loaded_page = scrape_web_page(url)

    structured_blocks = extract_structured_blocks(
        loaded_page.soup
    )

    if not structured_blocks:
        raise ValueError(
            "The loaded web page contained no structured content."
        )

    structured_text = "\n".join(
        block["text"]
        for block in structured_blocks
    )

    clean_text = clean_extracted_text(
        structured_text
    )

    if not clean_text:
        raise ValueError(
            "The loaded web page contained no usable clean text."
        )

    content_hash = build_content_hash(clean_text)
    
    source_language = get_declared_source_language(
    loaded_page.soup
    )

    try:
        search_session = create_search_session(
            db=db,
            user_id=user_id,
            query=query,
            urls=[loaded_page.requested_url],
            output_language=output_language,
            detected_intent=detected_intent,
        )

        source = create_source_record(
            db=db,
            session_id=search_session.id,
            user_id=user_id,
            url=loaded_page.requested_url,
            normalized_url=loaded_page.source_url,
            source_type=UNKNOWN_SOURCE_TYPE,
            source_language=source_language,
            clean_text=clean_text,
            structured_blocks=structured_blocks,
            content_hash=content_hash,
        )

        db.commit()

        db.refresh(search_session)
        db.refresh(source)

    except Exception:
        db.rollback()
        raise

    return IngestionResult(
        search_session=search_session,
        source=source,
        loaded_page=loaded_page,
    )
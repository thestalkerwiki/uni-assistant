"""Neutral web-page loading for Uni-Assist v2."""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import bs4
from langchain_community.document_loaders import WebBaseLoader


DEFAULT_USER_AGENT = (
    "UniAssist/2.0 "
    "(educational application intelligence project)"
)

CONTENT_TAGS = [
    "title",
    "main",
    "article",
    "h1",
    "h2",
    "h3",
    "p",
    "li",
]


class WebPageLoadError(RuntimeError):
    """Raised when a web page cannot be loaded into usable text."""


@dataclass(frozen=True)
class LoadedWebPage:
    """Unprocessed text and metadata extracted from one web page."""

    requested_url: str
    source_url: str
    title: Optional[str]
    extracted_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
@dataclass(frozen=True)
class ScrapedWebPage:
    """A web page preserved as a parsed HTML document."""

    requested_url: str
    source_url: str
    title: Optional[str]
    soup: Any


def normalize_urls(urls: list[str]) -> list[str]:
    """Normalize, validate and deduplicate user-supplied HTTP URLs."""

    normalized_urls: list[str] = []

    for item in urls:
        if not item:
            continue

        candidates = re.split(r"\s+", item.strip())

        for candidate in candidates:
            normalized_url = candidate.strip()

            if not normalized_url:
                continue

            if not normalized_url.startswith(
                ("http://", "https://")
            ):
                continue

            if normalized_url not in normalized_urls:
                normalized_urls.append(normalized_url)

    return normalized_urls


def scrape_web_page(url: str) -> ScrapedWebPage:
    """Load one URL while preserving its full HTML structure."""

    normalized_urls = normalize_urls([url])

    if not normalized_urls:
        raise ValueError(
            "A valid HTTP or HTTPS URL is required."
        )

    normalized_url = normalized_urls[0]

    loader = WebBaseLoader(
        normalized_url,
        header_template={
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )

    try:
        soup = loader.scrape()
    except Exception as error:
        raise WebPageLoadError(
            f"Could not scrape web page: {normalized_url}"
        ) from error

    if soup is None:
        raise WebPageLoadError(
            f"Web page returned no HTML document: {normalized_url}"
        )

    raw_title = (
        soup.title.get_text(" ", strip=True)
        if soup.title is not None
        else None
    )

    title = raw_title or None

    return ScrapedWebPage(
        requested_url=normalized_url,
        source_url=normalized_url,
        title=title,
        soup=soup,
    )


def load_web_page(url: str) -> LoadedWebPage:
    """Load one URL and return its extracted text and metadata."""

    normalized_urls = normalize_urls([url])

    if not normalized_urls:
        raise ValueError(
            "A valid HTTP or HTTPS URL is required."
        )

    normalized_url = normalized_urls[0]

    loader = WebBaseLoader(
        normalized_url,
        header_template={
            "User-Agent": DEFAULT_USER_AGENT,
        },
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(CONTENT_TAGS),
        },
        bs_get_text_kwargs={
            "separator": "\n",
            "strip": True,
        },
    )

    try:
        documents = loader.load()
    except Exception as error:
        raise WebPageLoadError(
            f"Could not load web page: {normalized_url}"
        ) from error

    extracted_parts = [
        document.page_content.strip()
        for document in documents
        if document.page_content.strip()
    ]

    if not extracted_parts:
        raise WebPageLoadError(
            f"Web page contained no usable text: {normalized_url}"
        )

    extracted_text = "\n\n".join(extracted_parts)

    metadata = (
        dict(documents[0].metadata)
        if documents
        else {}
    )

    source_url = str(
        metadata.get("source") or normalized_url
    )

    raw_title = metadata.get("title")
    title = (
        str(raw_title).strip()
        if raw_title
        else None
    )

    return LoadedWebPage(
        requested_url=normalized_url,
        source_url=source_url,
        title=title,
        extracted_text=extracted_text,
        metadata=metadata,
    )
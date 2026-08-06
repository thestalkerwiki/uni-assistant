"""SQLAlchemy persistence models for Uni-Assist v2."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from uni_assist.storage.database import Base


def generate_id() -> str:
    """Generate a portable string identifier."""

    return str(uuid4())


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


programme_sources = Table(
    "programme_sources",
    Base.metadata,
    Column(
        "programme_id",
        String(36),
        ForeignKey("programmes.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "source_id",
        String(36),
        ForeignKey("sources.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class UserModel(Base):
    """Locally persisted Uni-Assist user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    search_sessions: Mapped[list["SearchSessionModel"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    programmes: Mapped[list["ProgrammeModel"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class SearchSessionModel(Base):
    """A persisted user search or analysis request."""

    __tablename__ = "search_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )
    user_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    urls: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    detected_intent: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    output_language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="en",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    user: Mapped["UserModel"] = relationship(
        back_populates="search_sessions",
    )
    sources: Mapped[list["SourceModel"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )
    evidence_items: Mapped[list["EvidenceItemModel"]] = relationship(
        back_populates="session",
    )


class SourceModel(Base):
    """An immutable processed snapshot of one supplied source."""

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("search_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    normalized_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    source_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    source_language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    clean_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    structured_blocks: Mapped[list[dict]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    content_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    session: Mapped["SearchSessionModel"] = relationship(
        back_populates="sources",
    )
    programmes: Mapped[list["ProgrammeModel"]] = relationship(
        secondary=programme_sources,
        back_populates="sources",
    )
    evidence_items: Mapped[list["EvidenceItemModel"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )


class ProgrammeModel(Base):
    """A persistent programme profile accumulated over time."""

    __tablename__ = "programmes"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )
    user_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    programme_name: Mapped[Optional[str]] = mapped_column(
        String(300),
        nullable=True,
    )
    institution_name: Mapped[Optional[str]] = mapped_column(
        String(300),
        nullable=True,
    )
    degree: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )
    duration: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )
    credits: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    language: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )
    country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    user: Mapped["UserModel"] = relationship(
        back_populates="programmes",
    )
    sources: Mapped[list["SourceModel"]] = relationship(
        secondary=programme_sources,
        back_populates="programmes",
    )
    evidence_items: Mapped[list["EvidenceItemModel"]] = relationship(
        back_populates="programme",
    )
    missing_information: Mapped[list["MissingInformationModel"]] = (
        relationship(
            back_populates="programme",
            cascade="all, delete-orphan",
        )
    )


class EvidenceItemModel(Base):
    """A persisted admission fact with source provenance."""

    __tablename__ = "evidence_items"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=generate_id,
    )

    programme_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("programmes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("search_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    label: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    applicant_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    stage: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    importance: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    confidence: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    source_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    source_language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    raw_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    source_locator: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    programme: Mapped[Optional["ProgrammeModel"]] = relationship(
        back_populates="evidence_items",
    )
    session: Mapped["SearchSessionModel"] = relationship(
        back_populates="evidence_items",
    )
    source: Mapped["SourceModel"] = relationship(
        back_populates="evidence_items",
    )


class MissingInformationModel(Base):
    """A persisted unresolved Admission Core field."""

    __tablename__ = "missing_information_items"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_id,
    )
    programme_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("programmes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    field_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    importance: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    searched_source_ids: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )
    searched_source_urls: Mapped[list[str]] = mapped_column(
        JSON,
        default=list,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    programme: Mapped["ProgrammeModel"] = relationship(
        back_populates="missing_information",
    )
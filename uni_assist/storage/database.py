"""SQLite database configuration for Uni-Assist v2."""

from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from uni_assist.core.config import settings


settings.data_dir.mkdir(parents=True, exist_ok=True)

connect_args = {}

if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False


engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(
        dbapi_connection,
        _connection_record,
    ) -> None:
        """Enable SQLite foreign-key constraints for every connection."""

        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy persistence models."""

    pass


def get_db() -> Generator[Session, None, None]:
    """Provide a database session and always close it afterwards."""

    database_session = SessionLocal()

    try:
        yield database_session
    finally:
        database_session.close()


def init_db() -> None:
    """Create all database tables registered in SQLAlchemy metadata."""

    from uni_assist.storage import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
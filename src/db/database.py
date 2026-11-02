"""Engine and thread-safe session factory (SQLite locally, PostgreSQL in production).

The backend is selected purely by ``DATABASE_URL``::

    sqlite:///./data/churn.db                       # default, zero setup
    postgresql+psycopg://user:pass@host:5432/churn  # production (uv sync --extra postgres)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from src.config import get_settings
from src.core.exceptions import DatabaseError
from src.core.logger import get_logger

log = get_logger("db")


def _normalise_url(url: str) -> str:
    # Many PaaS providers still hand out the legacy ``postgres://`` scheme.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgres://")
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def build_engine(url: str | None = None, echo: bool | None = None) -> Engine:
    """Create an engine with backend-appropriate pooling and resilience settings."""
    s = get_settings()
    url = _normalise_url(url or s.database_url)
    echo = s.db_echo if echo is None else echo
    if url.startswith("sqlite"):
        in_memory = url in ("sqlite://", "sqlite:///:memory:")
        if not in_memory:
            db_path = Path(url.split("///", 1)[-1])
            db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if in_memory else None,
        )

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            if not in_memory:
                cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        return engine
    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,  # transparently recycle connections dropped by the server
        pool_recycle=1800,
        pool_size=s.db_pool_size,
        max_overflow=s.db_max_overflow,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide engine singleton."""
    return build_engine()


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables if they do not exist (idempotent)."""
    from src.db import models  # noqa: F401  (register tables on the metadata)

    engine = engine or get_engine()
    try:
        SQLModel.metadata.create_all(engine)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Could not initialise database: {exc}") from exc
    return engine


def make_session_factory(engine: Engine | None = None) -> scoped_session[Session]:
    """Thread-local scoped session registry (safe under Streamlit's threaded runtime)."""
    factory = sessionmaker(bind=engine or get_engine(), class_=Session, expire_on_commit=False)
    return scoped_session(factory)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Transactional unit of work: commit on success, rollback and wrap on failure."""
    session = Session(engine or get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        log.error("Database transaction failed", extra={"ctx": {"error": str(exc)}})
        raise DatabaseError(str(exc)) from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

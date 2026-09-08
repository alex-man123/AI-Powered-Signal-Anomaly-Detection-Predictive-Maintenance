"""TASK 2.1 — centralized SQLAlchemy configuration (engine, session factory, Base).

Schema creation only, via `Base.metadata.create_all()` — no Alembic exists yet anywhere
in this project, and neither blueprint.md nor backlog.md requires it for this task
(see backlog TASK 2.1: "modele definite, migrate, testate" is satisfied by a real,
tested schema-creation call, not by introducing a migration framework). If Alembic is
introduced later, this mechanism should be replaced, not layered underneath it.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

# Used only when settings.database_url is unset (e.g. no .env present yet) — this
# project's own .env.example intentionally ships DATABASE_URL empty (TASK 1.1), so a
# local default is needed for the app/tests to have somewhere to create the schema.
DEFAULT_DATABASE_URL = "sqlite:///./mafaulda.db"


class Base(DeclarativeBase):
    pass


def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().database_url or DEFAULT_DATABASE_URL
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), autoflush=False, autocommit=False)


def create_schema(engine: Engine) -> None:
    """Creates all tables registered on `Base.metadata` (datasets, signals,
    signal_windows) if they don't already exist. Does not populate any data.

    Imports `app.models` first (local import, avoids a circular import with
    app.models.signal importing `Base` from this module) -- SQLAlchemy declarative
    models only register onto `Base.metadata` when their module has actually been
    imported, so calling this without that import would silently create zero tables.
    """
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)

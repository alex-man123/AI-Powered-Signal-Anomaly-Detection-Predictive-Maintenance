"""Importing this package registers every model onto `app.core.database.Base.metadata`
-- required before `create_schema()` can create their tables (SQLAlchemy declarative
models only register themselves when their module is actually imported)."""

from app.models.signal import Dataset, Signal, SignalLabel, SignalRecord, SignalWindow

__all__ = ["Dataset", "Signal", "SignalLabel", "SignalRecord", "SignalWindow"]
